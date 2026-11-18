import math
import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from map_msgs.msg import OccupancyGridUpdate
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from tf2_ros import Buffer, TransformListener

ARC_VX = float(os.environ.get("ARC_VX", "0.3"))
ARC_WZ = float(os.environ.get("ARC_WZ", "0.8"))
REACH = 0.62


class TurnArc:
    def __init__(self, n):
        self.pub = n.create_publisher(Twist, "cmd_vel_nav", 10)
        self.grid = None
        self.pose = None
        self.side = 0.0
        self.last_turn = -1.0e9
        self.n = n
        n.create_subscription(Twist, "cmd_vel_ctrl", self.on_cmd, 10)
        n.create_subscription(Odometry, "odom", self.on_odom, 10)
        n.create_subscription(OccupancyGrid, "local_costmap/costmap", self.on_grid, 1)
        n.create_subscription(OccupancyGridUpdate, "local_costmap/costmap_updates", self.on_update, 5)
        n.create_subscription(Path, "plan", self.on_plan, 1)
        self.plan = None
        self.tf = Buffer()
        TransformListener(self.tf, n)

    def on_plan(self, m):
        self.plan = [(p.pose.position.x, p.pose.position.y) for p in m.poses]

    def path_error(self):
        if not self.plan:
            return None
        try:
            t = self.tf.lookup_transform("map", "base_link", rclpy.time.Time())
        except Exception:
            return None
        x, y = t.transform.translation.x, t.transform.translation.y
        q = t.transform.rotation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        k = min(range(len(self.plan)), key=lambda i: math.hypot(self.plan[i][0] - x, self.plan[i][1] - y))
        for px, py in self.plan[k:]:
            if math.hypot(px - x, py - y) > 0.6:
                e = math.atan2(py - y, px - x) - yaw
                return math.atan2(math.sin(e), math.cos(e))
        return None

    def on_odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

    def on_grid(self, m):
        self.grid = (np.array(m.data, dtype=np.int16).reshape(m.info.height, m.info.width), m.info.resolution,
                     m.info.origin.position.x, m.info.origin.position.y)

    def on_update(self, u):
        if self.grid is None:
            return
        g = self.grid[0]
        g[u.y:u.y + u.height, u.x:u.x + u.width] = np.array(u.data, dtype=np.int16).reshape(u.height, u.width)

    def side_cost(self, sign):
        if self.grid is None or self.pose is None:
            return 0.0
        g, res, ox, oy = self.grid
        x, y, yaw = self.pose
        r = ARC_VX / min(ARC_WZ, 0.6)
        cx, cy = x - sign * r * math.sin(yaw), y + sign * r * math.cos(yaw)
        worst = 0
        for a in np.linspace(0, 2 * math.pi, 48, endpoint=False):
            for d in (r + REACH * 0.5, r + REACH):
                px, py = cx + d * math.cos(a), cy + d * math.sin(a)
                i, j = int((py - oy) / res), int((px - ox) / res)
                v = g[i, j] if 0 <= i < g.shape[0] and 0 <= j < g.shape[1] else 100
                worst = max(worst, 100 if v < 0 else int(v))
        return worst

    def on_cmd(self, m):
        now = self.n.get_clock().now().nanoseconds * 1e-9
        if abs(m.linear.x) < 0.05 and abs(m.linear.y) < 0.05 and abs(m.angular.z) >= 0.1:
            if now - self.last_turn > 1.0:
                e = self.path_error()
                if e is not None and abs(e) < 2.6:
                    self.side = math.copysign(1.0, e)
                    why = f"path error {math.degrees(e):.0f} deg"
                    near, far = self.side_cost(self.side), self.side_cost(-self.side)
                    if near >= 99 and far < near:
                        self.side = -self.side
                        why += f", flipped, cost {near} vs {far}"
                else:
                    left, right = self.side_cost(1.0), self.side_cost(-1.0)
                    self.side = 1.0 if left < right else (-1.0 if right < left else math.copysign(1.0, m.angular.z))
                    why = f"u-turn, cost left {left}, right {right}"
                self.n.get_logger().info(f"arc turn {'left' if self.side > 0 else 'right'} ({why})")
            self.last_turn = now
            if self.side_cost(self.side) >= 99:
                m.linear.x = 0.0
            else:
                m.linear.x = ARC_VX
                m.angular.z = self.side * ARC_WZ
        self.pub.publish(m)


def main():
    rclpy.init()
    n = rclpy.create_node("turn_arc")
    TurnArc(n)
    rclpy.spin(n)


if __name__ == "__main__":
    main()
