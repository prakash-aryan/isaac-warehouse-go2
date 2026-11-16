import warp as wp

wp.config.enable_backward = False

import argparse
import collections
import json
import math
import os
import random
import sys
import time

import torch

from isaaclab_rl.entrypoints.backends import play_rsl_rl as P

ap = argparse.ArgumentParser(add_help=False)
ap.add_argument("--scene", required=True)
ap.add_argument("--robots", type=int, default=8)
ap.add_argument("--ros", action="store_true")
ap.add_argument("--lidars", default="hero", choices=["none", "hero", "all"])
ap.add_argument("--steps", type=int, default=0)
ap.add_argument("--stats", default="")
ap.add_argument("--speed", type=float, default=0.8)
ap.add_argument("--drop_every", type=float, default=4.0)
ap.add_argument("--pace", action="store_true")
ap.add_argument("--falls", default="")
ap.add_argument("--floor_drop", type=float, default=0.0)
ap.add_argument("--render_every", type=int, default=1)
own, rest = ap.parse_known_args(sys.argv[1:])
LAYOUT = json.load(open(own.scene.replace(".usda", "_layout.json")))
N = own.robots
DRAW = [True]


def yaw_quat(yaw):
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


def configure(env_cfg):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg

    env_cfg.scene.num_envs = N
    env_cfg.scene.env_spacing = 2.0
    env_cfg.episode_length_s = 1.0e6
    env_cfg.sim.render_interval = env_cfg.decimation * own.render_every
    env_cfg.scene.warehouse = AssetBaseCfg(prim_path="/World/Warehouse", spawn=sim_utils.UsdFileCfg(usd_path=own.scene),
                                           init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -own.floor_drop)))
    for k, t in enumerate(LAYOUT["boxes"]):
        setattr(env_cfg.scene, f"dyn_{k}", RigidObjectCfg(prim_path=f"/World/Warehouse/Dyn/{t}_.*", spawn=None))
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.42)
    ev = env_cfg.events
    ev.push_robot = None
    ev.reset_base.params["pose_range"] = {}
    ev.reset_base.params["velocity_range"] = {}
    ev.reset_robot_joints.params["position_range"] = (1.0, 1.0)
    c = env_cfg.commands.base_velocity
    c.resampling_time_range = (1.0e9, 1.0e9)
    c.heading_command = False
    c.rel_heading_envs = 0.0
    c.rel_standing_envs = 0.0
    c.ranges.heading = None
    c.debug_vis = False
    env_cfg.terminations.base_contact = None
    terrain = env_cfg.scene.terrain
    if getattr(terrain, "terrain_generator", None) is not None:
        terrain.terrain_type = "plane"
        terrain.terrain_generator = None
        if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
            env_cfg.curriculum.terrain_levels = None
    for obj in (getattr(env_cfg.sim, "physics", None), getattr(env_cfg.sim, "physx", None)):
        if obj is not None and hasattr(obj, "gpu_collision_stack_size"):
            obj.gpu_collision_stack_size = 2**30
        if obj is not None and hasattr(obj, "gpu_max_rigid_patch_count"):
            obj.gpu_max_rigid_patch_count = 2**21


def blocked(pos, heading, others, dist, cone=40.0):
    for o in others:
        d = math.dist(pos, o)
        if d < 1e-3 or d > dist:
            continue
        ang = math.degrees(math.atan2(o[1] - pos[1], o[0] - pos[0]) - heading)
        if abs((ang + 180.0) % 360.0 - 180.0) < cone:
            return True
    return False


def ahead(pos, heading, others, dist, half):
    c, s = math.cos(heading), math.sin(heading)
    for o in others:
        dx, dy = o[0] - pos[0], o[1] - pos[1]
        f = dx * c + dy * s
        if 0.05 < f < dist and abs(dy * c - dx * s) < half:
            return True
    return False


class Movers:
    def __init__(self, stage):
        from pxr import UsdGeom

        self.items, self.spins = [], []
        for m in LAYOUT["movers"]:
            prim = stage.GetPrimAtPath(m["prim"])
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps() if prim else []
            if len(ops) < 2:
                print("[wh] mover not found", m["prim"], flush=True)
                continue
            if m["mode"] == "spin":
                self.spins.append((ops[1], m["speed"]))
                continue
            pts = m["path"] + ([m["path"][0]] if m["mode"] == "loop" else [])
            segs = [(pts[i], pts[i + 1], math.dist(pts[i], pts[i + 1])) for i in range(len(pts) - 1)]
            self.items.append({"half": m.get("half", [0.5, 0.5]), "top": ops[0], "rop": ops[1], "segs": segs,
                               "total": sum(x[2] for x in segs),
                               "speed": m["speed"], "mode": m["mode"], "s": m.get("offset", 0.0),
                               "yoff": m.get("yaw_offset", 0.0), "pos": tuple(pts[0]), "head": 0.0, "wait": 0})
        self.where = [it["pos"] for it in self.items]
        print("[wh] movers", len(self.items), "fans", len(self.spins), flush=True)

    def _pose(self, it, s):
        back = False
        if it["mode"] == "pingpong":
            s = s % (2 * it["total"])
            back = s > it["total"]
            s = 2 * it["total"] - s if back else s
        else:
            s = s % it["total"]
        for a, b, length in it["segs"]:
            if s <= length or (a, b, length) == it["segs"][-1]:
                f = min(1.0, s / length) if length else 0.0
                x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
                return (x, y), math.atan2(b[1] - a[1], b[0] - a[0]) + (math.pi if back else 0.0)
            s -= length

    def update(self, t, dt, others):
        from pxr import Gf

        for rop, rate in self.spins if DRAW[0] else ():
            rop.Set((rate * t) % 360.0)
        for k, it in enumerate(self.items):
            stop = blocked(it["pos"], it["head"], others, 2.6, 35.0)
            it["wait"] = it["wait"] + 1 if stop else 0
            if not stop or it["wait"] > 1500:
                it["s"] += it["speed"] * dt
            (x, y), head = self._pose(it, it["s"])
            it["pos"], it["head"] = (x, y), head
            if DRAW[0]:
                it["top"].Set(Gf.Vec3d(x, y, 0.0))
                it["rop"].Set(math.degrees(head) + it["yoff"])
        self.where = [it["pos"] for it in self.items]
        return self.where

    def boxes(self):
        return [(it["pos"], it["head"], it["half"], it["yoff"]) for it in self.items]


class Shuttle:
    def __init__(self, stage):
        from pxr import UsdGeom

        self.cfg = LAYOUT.get("shuttle")
        if not self.cfg:
            self.ok = False
            return
        c = self.cfg
        self.f_ops = UsdGeom.Xformable(stage.GetPrimAtPath(c["forklift"])).GetOrderedXformOps()
        self.p_ops = UsdGeom.Xformable(stage.GetPrimAtPath(c["pallet"])).GetOrderedXformOps()
        fork = stage.GetPrimAtPath(c["fork"])
        self.fork_op = next((op for op in UsdGeom.Xformable(fork).GetOrderedXformOps() if "lift" in op.GetOpName()), None)
        self.ok = len(self.f_ops) >= 2 and len(self.p_ops) >= 2
        x, a, b, fa, so = c["x"], c["a"], c["b"], c["fork_ahead"], c["standoff"]
        south, north = -math.pi / 2, math.pi / 2
        eng_a, app_a = (x, a + fa), (x, a + fa + so)
        eng_b, app_b = (x, b - fa), (x, b - fa - so)
        carry_ab = [("turn", south), ("move", eng_a, False), ("lift", c["lift"]), ("move", app_a, True),
                    ("turn", north), ("move", eng_b, False), ("lift", 0.0), ("move", app_b, True)]
        carry_ba = [("move", eng_b, False), ("lift", c["lift"]), ("move", app_b, True), ("turn", south),
                    ("move", eng_a, False), ("lift", 0.0), ("move", app_a, True), ("turn", north)]
        self.plan = [("move", app_b, False), ("turn", south), ("move", app_a, False)] + carry_ab + carry_ba
        self.plan = self.plan[2:]
        self.k = 0
        self.pos = (x, (app_a[1] + app_b[1]) / 2)
        self.head = south
        self.lift = 0.0
        self.carry = False
        self.pallet = (x, a)
        self.wait = 0
        self.ahead = fa
        print("[wh] shuttle", "ok" if self.ok else "missing parts", "fork op", self.fork_op is not None, flush=True)

    def update(self, dt, others):
        from pxr import Gf

        if not self.ok:
            return []
        kind, arg, *rest = self.plan[self.k]
        done = False
        if kind == "move":
            reverse = rest[0]
            d = math.dist(self.pos, arg)
            mv_head = self.head + (math.pi if reverse else 0.0)
            stop = blocked(self.pos, mv_head, others, 2.4, 35.0) and not reverse
            self.wait = self.wait + 1 if stop else 0
            if not stop or self.wait > 1500:
                step = min(d, self.cfg["speed"] * (0.5 if reverse else 1.0) * dt)
                if d > 1e-6:
                    self.pos = (self.pos[0] + step * (arg[0] - self.pos[0]) / d,
                                self.pos[1] + step * (arg[1] - self.pos[1]) / d)
            done = math.dist(self.pos, arg) < 1e-3
        elif kind == "turn":
            err = math.atan2(math.sin(arg - self.head), math.cos(arg - self.head))
            step = max(-0.6 * dt, min(0.6 * dt, err))
            self.head += step
            done = abs(err) < 1e-3
        elif kind == "lift":
            err = arg - self.lift
            self.lift += max(-0.12 * dt, min(0.12 * dt, err))
            done = abs(err) < 1e-4
            if done:
                self.carry = arg > 0.0
        if done:
            self.k = (self.k + 1) % len(self.plan)
        fx, fy = self.pos[0] + self.ahead * math.cos(self.head), self.pos[1] + self.ahead * math.sin(self.head)
        if self.carry or (kind == "lift" and arg > 0.0):
            self.pallet = (fx, fy)
            if DRAW[0]:
                self.p_ops[0].Set(Gf.Vec3d(fx, fy, self.lift))
                self.p_ops[1].Set(math.degrees(self.head))
        if DRAW[0]:
            self.f_ops[0].Set(Gf.Vec3d(self.pos[0], self.pos[1], 0.0))
            self.f_ops[1].Set(math.degrees(self.head) + 90.0)
            if self.fork_op is not None:
                self.fork_op.Set(Gf.Vec3d(0.0, 0.0, self.lift))
        return [self.pos]

    def boxes(self):
        if not self.ok:
            return []
        c = self.cfg
        return [[self.pos[0], self.pos[1], c["half"][0], c["half"][1], math.degrees(self.head) + 90.0],
                [self.pallet[0], self.pallet[1], c["pallet_half"][0], c["pallet_half"][1], math.degrees(self.head)]]


class People:
    CLIP_SPEED = 1.08

    def __init__(self, stage):
        from pxr import Usd, UsdGeom, UsdSkel

        self.items, clips = [], {}
        for m in LAYOUT.get("people", []):
            if m["clip"] not in clips:
                cs = Usd.Stage.Open(m["clip"])
                a = UsdSkel.Animation(cs.GetDefaultPrim())
                times = a.GetRotationsAttr().GetTimeSamples()
                tr, sc = a.GetTranslationsAttr(), a.GetScalesAttr()
                frames = [(tr.Get(t), a.GetRotationsAttr().Get(t), sc.Get(t) if sc.GetTimeSamples() else sc.Get())
                          for t in times]
                fps = (len(times) - 1) * cs.GetTimeCodesPerSecond() / (times[-1] - times[0])
                clips[m["clip"]] = (frames, fps, cs)
            frames, fps, _ = clips[m["clip"]]
            prim = stage.GetPrimAtPath(m["prim"])
            anim = UsdSkel.Animation(stage.GetPrimAtPath(m["anim"]))
            ops = UsdGeom.Xformable(prim).GetOrderedXformOps() if prim else []
            if len(ops) < 2 or not anim:
                print("[wh] person not found", m["prim"], flush=True)
                continue
            pts = m.get("path") or []
            if m["mode"] == "loop":
                pts = pts + [pts[0]]
            segs = [(pts[i], pts[i + 1], math.dist(pts[i], pts[i + 1])) for i in range(len(pts) - 1)]
            self.items.append(dict(top=ops[0], rop=ops[1], segs=segs, total=sum(x[2] for x in segs), m=m,
                                   frames=frames, fps=fps, last=-1, s=m.get("offset", 0.0), phase=0.0,
                                   attrs=(anim.GetTranslationsAttr(), anim.GetRotationsAttr(), anim.GetScalesAttr()),
                                   pos=tuple(pts[0]) if pts else None, head=0.0, wait=0))
        self.clips = clips
        self.where = [it["pos"] for it in self.items if it["pos"] is not None]
        print("[wh] people", len(self.items), flush=True)

    def update(self, t, dt, others):
        from pxr import Gf

        where = []
        for it in self.items:
            m = it["m"]
            rate = 1.0
            if m["mode"] in ("pingpong", "loop") and it["total"] > 0:
                moving = True
                it["s"] += m["speed"] * dt
                s, back = it["s"], False
                if m["mode"] == "pingpong":
                    s = s % (2 * it["total"])
                    back = s > it["total"]
                    s = 2 * it["total"] - s if back else s
                else:
                    s = s % it["total"]
                for a, b, length in it["segs"]:
                    if s <= length or (a, b, length) == it["segs"][-1]:
                        f = min(1.0, s / length) if length else 0.0
                        x, y = a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1])
                        head = math.atan2(b[1] - a[1], b[0] - a[0]) + (math.pi if back else 0.0)
                        break
                    s -= length
                it["pos"], it["head"] = (x, y), head
                if DRAW[0]:
                    it["top"].Set(Gf.Vec3d(x, y, 0.0))
                    it["rop"].Set(math.degrees(head) + m.get("yaw_offset", 0.0))
                where.append((x, y))
                rate = m["speed"] / self.CLIP_SPEED if moving else 0.0
            it["phase"] += dt * rate * it["fps"]
            n = len(it["frames"])
            k = int((it["phase"] + m.get("phase", 0.0)) % n)
            if k != it["last"] and DRAW[0]:
                tr, r, sc = it["frames"][k]
                it["attrs"][0].Set(tr)
                it["attrs"][1].Set(r)
                if sc is not None:
                    it["attrs"][2].Set(sc)
                it["last"] = k
        self.where = where
        return where


class Boxes:
    def __init__(self, u):
        self.objs = [u.scene[f"dyn_{k}"] for k in range(len(LAYOUT["boxes"]))]
        self.slots = [(g, i) for g, o in enumerate(self.objs) for i in range(o.num_instances)]
        random.shuffle(self.slots)
        self.zones = LAYOUT["drop_zones"]
        self.dev = u.device
        self.next = 0
        self.last = -1e9
        print("[wh] dynamic boxes", len(self.slots), "in", len(self.objs), "groups", flush=True)

    def drop(self, t, batch):
        if not self.slots or t - self.last < own.drop_every:
            return
        self.last = t
        groups = {}
        for k in range(batch):
            g, i = self.slots[(self.next + k) % len(self.slots)]
            groups.setdefault(g, []).append(i)
        self.next = (self.next + batch) % len(self.slots)
        for g, ids in groups.items():
            poses = []
            for k, i in enumerate(ids):
                z = self.zones[(i + g) % len(self.zones)]
                q = yaw_quat(random.uniform(-math.pi, math.pi))
                poses.append([random.uniform(z[0], z[2]), random.uniform(z[1], z[3]), 2.5 + 0.5 * (k % 5), *q])
            idx = torch.tensor(ids, device=self.dev, dtype=torch.int32)
            self.objs[g].write_root_pose_to_sim_index(root_pose=torch.tensor(poses, device=self.dev), env_ids=idx)
            self.objs[g].write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=self.dev),
                                                          env_ids=idx)


class Scan2D:
    def __init__(self, n=720, rmin=0.15, rmax=25.0, rate=10.0):
        import numpy as np

        self.np = np
        self.n, self.rmin, self.rmax, self.period = n, rmin, rmax, 1.0 / rate
        self.ang = np.linspace(-math.pi, math.pi, n, endpoint=False)
        self.static = np.array(LAYOUT.get("footprints", []), dtype=np.float64).reshape(-1, 5)
        self.next = 0.0
        self.rng = np.random.default_rng(3)
        print("[wh] scan2d static boxes", len(self.static), flush=True)

    def _boxes(self, ox, oy, d, boxes):
        np = self.np
        if len(boxes) == 0:
            return np.full(len(d), np.inf)
        cx, cy, hx, hy, yaw = (boxes[:, k][None, :] for k in range(5))
        c, s = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        px, py = ox - cx, oy - cy
        lx, ly = c * px + s * py, -s * px + c * py
        dx = c * d[:, 0:1] + s * d[:, 1:2]
        dy = -s * d[:, 0:1] + c * d[:, 1:2]
        with np.errstate(divide="ignore", invalid="ignore"):
            t1x, t2x = (-hx - lx) / dx, (hx - lx) / dx
            t1y, t2y = (-hy - ly) / dy, (hy - ly) / dy
        tmin = np.maximum(np.minimum(t1x, t2x), np.minimum(t1y, t2y))
        tmax = np.minimum(np.maximum(t1x, t2x), np.maximum(t1y, t2y))
        hit = (tmax >= tmin) & (tmax > 0) & (tmin > 0)
        t = np.where(hit, tmin, np.inf)
        return np.nan_to_num(t, nan=np.inf).min(axis=1)

    def _circles(self, ox, oy, d, circ):
        np = self.np
        if len(circ) == 0:
            return np.full(len(d), np.inf)
        cx, cy, r = circ[:, 0][None, :], circ[:, 1][None, :], circ[:, 2][None, :]
        fx, fy = ox - cx, oy - cy
        b = d[:, 0:1] * fx + d[:, 1:2] * fy
        cc = fx * fx + fy * fy - r * r
        disc = b * b - cc
        t = -b - np.sqrt(np.maximum(disc, 0.0))
        t = np.where((disc >= 0) & (t > 0), t, np.inf)
        return t.min(axis=1)

    def ranges(self, ox, oy, yaw, dyn_boxes, circles):
        np = self.np
        a = self.ang + yaw
        d = np.stack([np.cos(a), np.sin(a)], 1)
        t = np.minimum(self._boxes(ox, oy, d, self.static), self._boxes(ox, oy, d, np.array(dyn_boxes).reshape(-1, 5)))
        t = np.minimum(t, self._circles(ox, oy, d, np.array(circles).reshape(-1, 3)))
        t = t + self.rng.normal(0.0, 0.01, t.shape)
        t[(t > self.rmax) | ~np.isfinite(t)] = np.inf
        t[t < self.rmin] = np.inf
        return t


class Ros:
    def __init__(self, u, start):
        import rclpy
        from geometry_msgs.msg import TransformStamped, Twist
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from rosgraph_msgs.msg import Clock
        from tf2_msgs.msg import TFMessage

        rclpy.init()
        self.rclpy = rclpy
        self.node = rclpy.create_node("warehouse_sim")
        self.Odometry, self.TransformStamped, self.TFMessage, self.Clock = Odometry, TransformStamped, TFMessage, Clock
        self.clock = self.node.create_publisher(Clock, "/clock", 10)
        self.odom = self.node.create_publisher(Odometry, "/odom", 10)
        self.tf = self.node.create_publisher(TFMessage, "/tf", 50)
        static_qos = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.tf_static = self.node.create_publisher(TFMessage, "/tf_static", static_qos)
        from sensor_msgs.msg import LaserScan

        self.LaserScan = LaserScan
        self.scan = self.node.create_publisher(LaserScan, "/scan", 10)
        self.cmd = (0.0, 0.0, 0.0)
        self.cmd_wall = 0.0
        self.node.create_subscription(Twist, "/cmd_vel", self._on_cmd, 10)
        self.x0, self.y0, self.yaw0 = start
        lz = LAYOUT["lidar_height"]
        st = self._tf("base_link", "laser", LAYOUT["lidar_forward"], 0.0, lz, 0.0, None)
        self.tf_static.publish(TFMessage(transforms=[st]))
        print("[wh] ros node up", flush=True)

    def _on_cmd(self, msg):
        self.cmd = (msg.linear.x, msg.linear.y, msg.angular.z)
        self.cmd_wall = time.monotonic()

    def command(self):
        self.rclpy.spin_once(self.node, timeout_sec=0.0)
        if time.monotonic() - self.cmd_wall > 0.5:
            return (0.0, 0.0, 0.0)
        return self.cmd

    def _stamp(self, t):
        from builtin_interfaces.msg import Time

        sec = int(t)
        return Time(sec=sec, nanosec=int((t - sec) * 1e9))

    def _tf(self, parent, child, x, y, z, yaw, t):
        m = self.TransformStamped()
        if t is not None:
            m.header.stamp = self._stamp(t)
        m.header.frame_id, m.child_frame_id = parent, child
        m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = x, y, z
        m.transform.rotation.z, m.transform.rotation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        return m

    def publish_scan(self, t, ranges, sc):
        m = self.LaserScan()
        m.header.stamp = self._stamp(t)
        m.header.frame_id = "laser"
        m.angle_min, m.angle_max = -math.pi, math.pi - 2 * math.pi / sc.n
        m.angle_increment = 2 * math.pi / sc.n
        m.scan_time, m.time_increment = sc.period, 0.0
        m.range_min, m.range_max = sc.rmin, sc.rmax
        m.ranges = [float(r) for r in ranges]
        self.scan.publish(m)

    def publish(self, t, pos, yaw, vel_b, wz):
        c, s = math.cos(-self.yaw0), math.sin(-self.yaw0)
        dx, dy = pos[0] - self.x0, pos[1] - self.y0
        ox, oy = c * dx - s * dy, s * dx + c * dy
        oyaw = math.atan2(math.sin(yaw - self.yaw0), math.cos(yaw - self.yaw0))
        stamp = self._stamp(t)
        self.clock.publish(self.Clock(clock=stamp))
        o = self.Odometry()
        o.header.stamp = stamp
        o.header.frame_id, o.child_frame_id = "odom", "base_link"
        o.pose.pose.position.x, o.pose.pose.position.y = ox, oy
        o.pose.pose.orientation.z, o.pose.pose.orientation.w = math.sin(oyaw / 2), math.cos(oyaw / 2)
        o.twist.twist.linear.x, o.twist.twist.linear.y = vel_b[0], vel_b[1]
        o.twist.twist.angular.z = wz
        self.odom.publish(o)
        self.tf.publish(self.TFMessage(transforms=[self._tf("odom", "base_link", ox, oy, 0.0, oyaw, t)]))


def make_lidars(stage, hero_only):
    from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor
    from pxr import UsdGeom

    rigs, sensors = [], []
    count = 1 if hero_only else N
    for i in range(count):
        path = f"/World/LidarRigs/rig_{i:02d}"
        xf = UsdGeom.Xform.Define(stage, path)
        rigs.append((xf.AddTranslateOp(), xf.AddRotateZOp()))
        cfg = "Example_Rotary_2D" if i == 0 else "Example_Rotary"
        extra = {"accumulate_outputs": True, "attributes": {"omni:sensor:Core:nearRangeM": 0.15}} if i == 0 else {}
        lidar = Lidar.create(path=f"{path}/lidar", config=cfg, tick_rate=10.0, translations=[[0.0, 0.0, 0.0]], **extra)
        sensors.append((lidar, LidarSensor(lidar, annotators=["generic-model-output"])))
    return rigs, sensors


def attach_scan_writer(lidar, sensor):
    import isaacsim.core.experimental.utils.prim as prim_utils

    p = prim_utils.get_prim_at_path(lidar.paths[0])
    rate = float(p.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get())
    fire = int(p.GetAttribute("omni:sensor:Core:patternFiringRateHz").Get())
    sensor.attach_writer("RtxLidarROS2PublishLaserScan", topicName="scan", frameId="laser", horizontalFov=360.0,
                         horizontalResolution=360.0 * rate / fire,
                         depthRange=[float(p.GetAttribute("omni:sensor:Core:nearRangeM").Get()),
                                     float(p.GetAttribute("omni:sensor:Core:farRangeM").Get())],
                         rotationRate=rate, azimuthRange=[-180.0, 180.0])


TURN = {}


def controller(pos, yaw, wps, k, i=0, look=0.8):
    if math.dist(pos, wps[k]) < 0.8:
        k = (k + 1) % len(wps)
    (ax, ay), (bx, by) = wps[k - 1], wps[k]
    dx, dy = bx - ax, by - ay
    seg = math.hypot(dx, dy)
    if seg > 1e-6:
        t = ((pos[0] - ax) * dx + (pos[1] - ay) * dy) / (seg * seg)
        t = min(1.0, max(0.0, t) + look / seg)
    else:
        t = 1.0
    tx, ty = ax + t * dx, ay + t * dy
    err = math.atan2(ty - pos[1], tx - pos[0]) - yaw
    err = math.atan2(math.sin(err), math.cos(err))
    if abs(err) > 2.4:
        err = math.copysign(abs(err), TURN.get(i, 1.0))
    else:
        TURN[i] = 1.0 if err >= 0 else -1.0
    vx = own.speed * max(0.0, math.cos(err)) ** 2
    wz = max(-0.8, min(0.8, 1.8 * err))
    return [vx, 0.0, wz], k


def main():
    args_cli = P._parse_args(rest)
    installed = P.check_rsl_rl_version()
    env_cfg, agent_cfg = P.resolve_task_config(args_cli.task, args_cli.agent, play_mode=True)
    with P.launch_simulation(env_cfg, args_cli):
        agent_cfg = P.cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
        agent_cfg = P.handle_deprecated_rsl_rl_cfg(agent_cfg, installed)
        P.apply_env_overrides(args_cli, env_cfg)
        env_cfg.seed = agent_cfg.seed
        configure(env_cfg)
        log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
        ckpt = P._resolve_checkpoint(args_cli, agent_cfg, env_cfg, log_root)
        env_cfg.log_dir = os.path.dirname(ckpt)
        env = P.create_isaaclab_env(args_cli.task, env_cfg, args_cli, convert_marl_to_single_agent=False)
        env = P.RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = P.create_rsl_rl_runner(env, agent_cfg)
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        u = env.unwrapped
        import omni.kit.app
        import omni.usd

        em = omni.kit.app.get_app().get_extension_manager()
        exts = ["isaacsim.core.experimental.utils", "isaacsim.sensors.experimental.rtx"]
        if own.ros:
            exts.append("isaacsim.ros2.bridge")
        for e in exts:
            em.set_extension_enabled_immediate(e, True)
            print("[wh] extension", e, em.is_extension_enabled(e), flush=True)
        from isaacsim.core.simulation_manager import SimulationManager
        from pxr import Gf, UsdGeom

        stage = omni.usd.get_context().get_stage()
        for prim in stage.Traverse():
            if str(prim.GetPath()).startswith("/World/ground") and prim.IsA(UsdGeom.Mesh):
                UsdGeom.Imageable(prim).MakeInvisible()

        robots = LAYOUT["robots"][:N]
        if len(robots) < N:
            raise SystemExit(f"layout has {len(robots)} robot slots, {N} requested")
        origins = u.scene.env_origins
        for i, r in enumerate(robots):
            origins[i, 0], origins[i, 1], origins[i, 2] = r["start"][0], r["start"][1], 0.0
        obs, _ = env.reset()
        robot = u.scene["robot"]
        dev = u.device

        def set_yaw(ids):
            p = robot.data.root_pos_w.torch[ids].clone()
            q = torch.tensor([yaw_quat(math.radians(robots[i]["start"][2])) for i in ids], device=dev)
            robot.write_root_pose_to_sim_index(root_pose=torch.cat([p, q], dim=1),
                                               env_ids=torch.tensor(ids, device=dev, dtype=torch.int32))

        set_yaw(list(range(N)))
        obs = env.get_observations()

        safe = [collections.deque(maxlen=6) for _ in range(N)]
        trail = [collections.deque(maxlen=30) for _ in range(N)]
        low = [0] * N
        fall_t = [[] for _ in range(N)]
        fall_log = []

        def recover(ids, t, ts, pos, gz):
            rows = []
            for i in ids:
                fall_t[i] = [s for s in fall_t[i] if ts - s < 60.0] + [ts]
                if safe[i] and len(fall_t[i]) < 3:
                    x, y, yw = safe[i][0]
                else:
                    x, y, yw = robots[i]["start"][0], robots[i]["start"][1], math.radians(robots[i]["start"][2])
                rows.append([x, y, 0.42, *yaw_quat(yw)])
                ev = {"robot": i, "step": t, "sim_s": round(ts, 2), "at": [round(v, 2) for v in pos[i]],
                      "gz": round(gz[i], 2), "to": [round(x, 2), round(y, 2), round(math.degrees(yw), 1)],
                      "recent": len(fall_t[i]), "trail": list(trail[i])}
                for key, pts in (("robot_m", [p[:2] for j, p in enumerate(pos) if j != i]), ("person_m", people.where),
                                 ("mover_m", movers.where), ("shuttle_m", [shuttle.pos] if shuttle.ok else [])):
                    ev[key] = round(min((math.dist(pos[i][:2], q) for q in pts), default=-1.0), 2)
                fall_log.append(ev)
                print("[wh] fall", json.dumps({k: v for k, v in ev.items() if k != "trail"}), flush=True)
                safe[i].clear()
                trail[i].clear()
                low[i] = 0
            idx = torch.tensor(ids, device=dev, dtype=torch.int32)
            sel = torch.tensor(ids, device=dev, dtype=torch.long)
            robot.write_root_pose_to_sim_index(root_pose=torch.tensor(rows, device=dev), env_ids=idx)
            robot.write_root_velocity_to_sim_index(root_velocity=torch.zeros(len(ids), 6, device=dev), env_ids=idx)
            jp = robot.data.default_joint_pos.torch[sel].clone()
            robot.write_joint_position_to_sim_index(position=jp, env_ids=idx)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(jp), env_ids=idx)
            u.action_manager.reset(env_ids=ids)
            dn = torch.zeros(N, dtype=torch.bool, device=dev)
            dn[sel] = True
            policy.reset(dn)
            if own.falls:
                json.dump(fall_log, open(own.falls, "w"))

        import omni.timeline

        tl = omni.timeline.get_timeline_interface()
        tl.set_looping(False)
        tl.set_end_time(1.0e7)
        movers = Movers(stage)
        people = People(stage)
        shuttle = Shuttle(stage)
        shuttle_where = []
        boxes = Boxes(u)
        rigs, sensors = ([], [])
        if own.lidars != "none":
            rigs, sensors = make_lidars(stage, own.lidars == "hero")
        lf = LAYOUT["lidar_forward"]
        ros = None
        scan2d = None
        if own.ros:
            r0 = robots[0]["start"]
            ros = Ros(u, (r0[0], r0[1], math.radians(r0[2])))
            scan2d = Scan2D()
        dt = float(u.step_dt)
        term = u.command_manager.get_term("base_velocity")
        wps = [[tuple(w) for w in r["route"]] for r in robots]
        ks = [0] * N
        rwait = [0] * N
        hero_cmd = [0.0, 0.0, 0.0]
        lf, lz = LAYOUT["lidar_forward"], LAYOUT["lidar_height"]
        stats, t_wall0, last_t, last_wall = [], time.monotonic(), 0, time.monotonic()
        n_steps = own.steps if own.steps > 0 else 10**9
        print("[wh] running", N, "robots", flush=True)
        with torch.inference_mode():
            for t in range(n_steps):
                ts = t * dt
                DRAW[0] = (t + 1) % own.render_every == 0
                pos = robot.data.root_pos_w.torch[:, :3].tolist()
                quat = robot.data.root_quat_w.torch.tolist()
                yaws = [math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] * q[1] + q[2] * q[2])) for q in quat]
                gz = robot.data.projected_gravity_b.torch[:, 2].tolist()
                for i in range(N):
                    if pos[i][2] > 0.25 and gz[i] < -0.8 and t % 25 == 0:
                        safe[i].append((pos[i][0], pos[i][1], yaws[i]))
                    low[i] = low[i] + 1 if pos[i][2] < 0.22 or gz[i] > -0.5 else 0
                fallen = [i for i in range(N) if low[i] >= 25]
                if fallen:
                    recover(fallen, t, ts, pos, gz)
                    if 0 in fallen:
                        hero_cmd = [0.0, 0.0, 0.0]
                    obs = env.get_observations()
                cmds = []
                xy = [tuple(p[:2]) for p in pos]
                follow = [[j for j in range(N) if j != i and ahead(xy[i], yaws[i], [xy[j]], 1.3, 0.55)] for i in range(N)]
                for i in range(N):
                    if i == 0 and ros is not None:
                        vx, vy, wz = ros.command()
                        tgt = [max(-0.3, min(0.9, vx)), 0.0, max(-0.7, min(0.7, wz))]
                        for j in range(3):
                            hero_cmd[j] += 0.08 * (tgt[j] - hero_cmd[j])
                        cmds.append(list(hero_cmd))
                    else:
                        cmd, ks[i] = controller(pos[i][:2], yaws[i], wps[i], ks[i], i)
                        rb = [j for j in follow[i] if not (i in follow[j] and i < j)]
                        rwait[i] = rwait[i] + 1 if rb else 0
                        if rwait[i] > 300:
                            rwait[i] = 0
                        if ahead(xy[i], yaws[i], people.where, 1.7, 0.6) or (rb and rwait[i] < 200):
                            cmd = [0.0, 0.0, 0.0]
                        cmds.append(cmd)
                term.vel_command_b[:] = torch.tensor(cmds, device=dev)
                if t % 5 == 0:
                    for i in range(N):
                        trail[i].append([round(ts, 2), *(round(v, 3) for v in pos[i]), round(math.degrees(yaws[i]), 1),
                                         round(gz[i], 3), *(round(v, 2) for v in cmds[i])])
                bots = [tuple(p[:2]) for p in pos]
                walkers = people.where
                mv = movers.update(ts, dt, bots + walkers)
                sh = shuttle.update(dt, bots + walkers)
                shuttle_where = sh
                people.update(ts, dt, bots + mv + sh)
                boxes.drop(ts, LAYOUT.get("drop_batch", 24))
                for i, (top, rop) in enumerate(rigs if DRAW[0] else ()):
                    c, s = math.cos(yaws[i]), math.sin(yaws[i])
                    top.Set(Gf.Vec3d(pos[i][0] + lf * c, pos[i][1] + lf * s, lz))
                    rop.Set(math.degrees(yaws[i]))
                t_step = time.monotonic()
                obs, _, dones, _ = env.step(policy(obs))
                if own.pace:
                    left = dt - (time.monotonic() - t_step)
                    if left > 0:
                        time.sleep(left)
                policy.reset(dones)
                if ros is not None:
                    vb = robot.data.root_lin_vel_b.torch[0].tolist()
                    wz = robot.data.root_ang_vel_b.torch[0, 2].item()
                    tsim = SimulationManager.get_simulation_time()
                    ros.publish(tsim, pos[0], yaws[0], vb, wz)
                    if tsim >= scan2d.next:
                        scan2d.next = tsim + scan2d.period
                        p0 = robot.data.root_pos_w.torch[0].tolist()
                        q0 = robot.data.root_quat_w.torch[0].tolist()
                        y0 = math.atan2(2 * (q0[3] * q0[2] + q0[0] * q0[1]), 1 - 2 * (q0[1] ** 2 + q0[2] ** 2))
                        lx, ly = p0[0] + lf * math.cos(y0), p0[1] + lf * math.sin(y0)
                        dyn = [[x, y, h[0], h[1], math.degrees(hd) + yo] for (x, y), hd, h, yo in movers.boxes()]
                        dyn += shuttle.boxes()
                        circ = [[x, y, 0.28] for x, y in people.where] + [[p[0], p[1], 0.32] for p in pos[1:]]
                        ros.publish_scan(tsim, scan2d.ranges(lx, ly, y0, dyn, circ), scan2d)
                d = dones.nonzero().flatten().tolist()
                if d:
                    set_yaw(d)
                    for i in d:
                        ks[i] = 0
                    print("[wh] reset", d, "at step", t, flush=True)
                if t % 250 == 0:
                    now = time.monotonic()
                    rate = (t - last_t) / max(1e-6, now - last_wall)
                    last_t, last_wall = t, now
                    st = {"step": t, "sim_s": round(ts, 2), "steps_per_s": round(rate, 2),
                          "rtf": round(rate * dt, 3), "simtime": round(SimulationManager.get_simulation_time(), 3),
                          "hero": [round(v, 2) for v in pos[0]], "hero_cmd": [round(v, 2) for v in cmds[0]]}
                    st["timeline"] = [round(tl.get_current_time(), 3), tl.is_playing()]
                    st["falls"] = len(fall_log)
                    if sensors:
                        try:
                            d, _ = sensors[0][1].get_data("generic-model-output")
                            st["lidar_pts"] = 0 if d is None else int(d.size)
                        except Exception as e:
                            st["lidar_pts"] = str(e)[:80]
                    if shuttle.ok:
                        st["shuttle"] = [round(shuttle.pos[0], 2), round(shuttle.pos[1], 2), shuttle.k,
                                         round(shuttle.lift, 2), shuttle.carry]
                    stats.append(st)
                    print("[wh]", json.dumps(st), flush=True)
                    if own.stats:
                        json.dump(stats, open(own.stats, "w"))
        print("[wh] done wall", round(time.monotonic() - t_wall0, 1), flush=True)
        P.close_env(env)


if __name__ == "__main__":
    main()
