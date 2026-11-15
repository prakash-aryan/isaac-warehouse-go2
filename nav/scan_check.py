import math
import time

import rclpy
from nav_msgs.msg import Odometry
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan

rclpy.init()
n = rclpy.create_node("scan_check")
got = {}
n.create_subscription(LaserScan, "/scan", lambda m: got.__setitem__("scan", m), qos_profile_sensor_data)
n.create_subscription(Odometry, "/odom", lambda m: got.__setitem__("odom", m), 10)
t0 = time.time()
while time.time() - t0 < 10 and not ("scan" in got and "odom" in got):
    rclpy.spin_once(n, timeout_sec=0.2)
s, o = got["scan"], got["odom"]
q = o.pose.pose.orientation
print("odom yaw deg", round(math.degrees(2 * math.atan2(q.z, q.w)), 1), "pos", round(o.pose.pose.position.x, 2), round(o.pose.pose.position.y, 2))
print("frame", s.header.frame_id, "amin", round(math.degrees(s.angle_min), 2), "amax", round(math.degrees(s.angle_max), 2),
      "inc", round(math.degrees(s.angle_increment), 4), "n", len(s.ranges), "rmin", s.range_min, "rmax", s.range_max)
for deg in (0, 45, 90, 135, 180, -135, -90, -45):
    i = int(round((math.radians(deg) - s.angle_min) / s.angle_increment)) % len(s.ranges)
    w = [r for r in s.ranges[max(0, i - 2):i + 3]]
    print(f"{deg:5d} deg range {min(w):.2f}")
