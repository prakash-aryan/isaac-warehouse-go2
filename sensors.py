import argparse
import collections
import math
import time

import numpy as np

sp = argparse.ArgumentParser(add_help=False)
sp.add_argument("--port", type=int, default=47556)
sp.add_argument("--go2", default="")
sp.add_argument("--num_lidars", type=int, default=0)
sp.add_argument("--rate", type=float, default=10.0)
sp.add_argument("--beams", type=int, default=720)
sp.add_argument("--stride", type=int, default=4)
sp.add_argument("--fabric", type=int, default=1)
sp.add_argument("--mount_x", type=float, default=0.38)
sp.add_argument("--mount_z", type=float, default=0.0)
sp.add_argument("--rear_mask", type=float, default=0.75)
sp.add_argument("--near", type=float, default=0.03)
sp.add_argument("--fire_hz", type=float, default=0.0)
sown, _ = sp.parse_known_args()

import run_warehouse as W
from isaaclab.app import AppLauncher, add_launcher_args

lp = argparse.ArgumentParser(add_help=False)
add_launcher_args(lp)
largs, _ = lp.parse_known_args()
app = AppLauncher(largs).app

import omni.kit.app
import omni.timeline
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics

import mirror
import stream

em = omni.kit.app.get_app().get_extension_manager()
for e in ("isaacsim.core.experimental.utils", "isaacsim.sensors.experimental.rtx"):
    em.set_extension_enabled_immediate(e, True)
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor, parse_generic_model_output_data


def strip_physics(root):
    joints = bodies = 0
    for p in Usd.PrimRange(root):
        if p.IsA(UsdPhysics.Joint):
            p.SetActive(False)
            joints += 1
            continue
        if p.HasAPI(UsdPhysics.ArticulationRootAPI):
            p.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            p.RemoveAPI(UsdPhysics.RigidBodyAPI)
            bodies += 1
    return joints, bodies


class Scans:
    def __init__(self, n):
        import rclpy
        from sensor_msgs.msg import LaserScan

        rclpy.init()
        self.rclpy, self.LaserScan = rclpy, LaserScan
        self.node = rclpy.create_node("warehouse_lidars")
        self.pubs = [self.node.create_publisher(LaserScan, f"/go2_{i}/scan", 10) for i in range(n)]
        self.res = 2 * math.pi / sown.beams
        ang = -math.pi + self.res * np.arange(sown.beams)
        self.rear = np.abs(ang) > math.radians(115.0)

    def bins(self, az_deg, rng):
        out = np.full(sown.beams, np.inf, dtype=np.float32)
        k = np.floor((np.radians(az_deg) + math.pi) / self.res).astype(np.int64) % sown.beams
        order = np.lexsort((rng, k))
        k, rng = k[order], rng[order]
        first = np.ones(len(k), dtype=bool)
        first[1:] = k[1:] != k[:-1]
        out[k[first]] = rng[first]
        return out

    def publish(self, i, stamp, bins, rmin, rmax):
        from builtin_interfaces.msg import Time

        bins = bins.copy()
        bins[bins > rmax] = np.inf
        bins[self.rear & (bins < sown.rear_mask)] = np.nan
        m = self.LaserScan()
        sec = int(stamp)
        m.header.stamp = Time(sec=sec, nanosec=int((stamp - sec) * 1e9))
        m.header.frame_id = "laser"
        m.angle_min, m.angle_max = -math.pi, math.pi - self.res
        m.angle_increment = self.res
        m.scan_time, m.time_increment = 1.0 / sown.rate, 0.0
        m.range_min, m.range_max = float(rmin), float(rmax)
        m.ranges = bins.tolist()
        self.pubs[i].publish(m)


def main():
    ctx = omni.usd.get_context()
    ctx.new_stage()
    for _ in range(5):
        app.update()
    stage = ctx.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    world = UsdGeom.Xform.Define(stage, "/World").GetPrim()
    stage.SetDefaultPrim(world)
    stage.DefinePrim("/World/Warehouse", "Xform").GetReferences().AddReference(W.own.scene)
    path = mirror.go2_path(sown.go2)
    for i in range(W.N):
        stage.DefinePrim(f"/World/Robots/R{i}", "Xform").GetReferences().AddReference(path)
    for _ in range(20):
        app.update()
    joints, bodies = strip_physics(stage.GetPrimAtPath("/World/Robots"))
    print(f"[sensors] stage ready, robot physics removed: {joints} joints, {bodies} rigid bodies", flush=True)
    view = mirror.View(stage, W.N, W, fabric=sown.fabric, animate=False)
    k = W.N if sown.num_lidars <= 0 else min(sown.num_lidars, W.N)
    lf = sown.mount_x
    sensors = []
    for i in range(k):
        lidar = Lidar.create(path=f"/World/Robots/R{i}/base/lidar", config="Example_Rotary_2D", tick_rate=sown.rate,
                             translations=[[lf, 0.0, sown.mount_z]], accumulate_outputs=True,
                             attributes={"omni:sensor:Core:nearRangeM": sown.near})
        prim = stage.GetPrimAtPath(f"/World/Robots/R{i}/base/lidar")
        if sown.fire_hz > 0:
            prim.GetAttribute("omni:sensor:Core:patternFiringRateHz").Set(int(sown.fire_hz))
        if i == 0:
            em = [a.GetName() for a in prim.GetAttributes() if "emitterState" in a.GetName() and a.GetName().endswith("azimuthDeg")]
            n_em = len(prim.GetAttribute(em[0]).Get() or []) if em else -1
            print("[sensors] lidar fire", prim.GetAttribute("omni:sensor:Core:patternFiringRateHz").Get(), "Hz, scan",
                  prim.GetAttribute("omni:sensor:Core:scanRateBaseHz").Get(), "Hz, emitters", n_em, "far",
                  prim.GetAttribute("omni:sensor:Core:farRangeM").Get(), flush=True)
        sensors.append(LidarSensor(lidar, annotators=["generic-model-output"]))
    scans = Scans(k)
    tl = omni.timeline.get_timeline_interface()
    tl.set_looping(False)
    tl.set_end_time(1.0e7)
    tl.play()
    rx = stream.Receiver(sown.port)
    hist = collections.deque(maxlen=600)
    acc = [None] * k
    frames, published, t0, last = 0, 0, time.monotonic(), None
    tm = {"apply": 0.0, "update": 0.0, "lidar": 0.0}
    print(f"[sensors] {k} lidars at {sown.rate} Hz, {sown.beams} beams, state port {sown.port}", flush=True)
    while app.is_running():
        c0 = time.monotonic()
        d = rx.latest()
        if d is not None:
            view.apply(d)
            last = d["clock"]
        if last is not None:
            hist.append((tl.get_current_time(), last))
        c1 = time.monotonic()
        app.update()
        c2 = time.monotonic()
        tm["apply"] += c1 - c0
        tm["update"] += c2 - c1
        frames += 1
        if not hist:
            continue
        for i, s in enumerate(sensors):
            data, _ = s.get_data("generic-model-output")
            if data is None:
                continue
            g = parse_generic_model_output_data(data)
            n = int(g.numElements)
            ts = int(g.timestampNs)
            if n == 0:
                continue
            az = np.array(g.x, copy=True)[:n:sown.stride]
            rng = np.array(g.z, copy=True)[:n:sown.stride]
            part = scans.bins(az, rng)
            rmax = float(g.maxRangeM) if g.maxRangeM else 30.0
            if acc[i] is not None and ts != acc[i][0]:
                tsec = acc[i][0] / 1e9
                ref = next(((a, c) for a, c in reversed(hist) if a <= tsec), hist[0])
                stamp = ref[1] + max(0.0, tsec - ref[0])
                if i == 0 and published % 50 == 0:
                    print(f"[sensors] scan pieces={acc[i][2]} bins filled {int(np.isfinite(acc[i][1]).sum())}/{sown.beams}", flush=True)
                scans.publish(i, stamp, acc[i][1], sown.near, rmax)
                published += 1
                acc[i] = None
            if acc[i] is None:
                acc[i] = [ts, part, 1]
            else:
                np.minimum(acc[i][1], part, out=acc[i][1])
                acc[i][2] += 1
        now = time.monotonic()
        tm["lidar"] += now - c2
        if now - t0 > 5.0:
            print("[sensors] ms per frame " + " ".join(f"{k}={v / max(1, frames) * 1000:.1f}" for k, v in tm.items()), flush=True)
            tm = {k: 0.0 for k in tm}
            print(f"[sensors] fps={frames / (now - t0):.1f} scans/s={published / (now - t0):.1f} timeline={tl.get_current_time():.1f} clock={last}", flush=True)
            frames, published, t0 = 0, 0, now


main()
app.close()
