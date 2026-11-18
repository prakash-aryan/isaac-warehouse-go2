import math
import os

from pxr import Gf, UsdGeom


def go2_path(override=""):
    if override:
        return override
    from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

    url = UNITREE_GO2_CFG.spawn.usd_path
    cached = "/tmp/" + url.replace("://", "/", 1)
    return cached if os.path.exists(cached) else url


def fabric_matrix(path):
    import omni.usd
    import usdrt
    from usdrt import Rt

    stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
    prim = stage.GetPrimAtPath(path)
    x = Rt.Xformable(prim)
    x.CreateFabricHierarchyWorldMatrixAttr()
    x.CreateFabricHierarchyLocalMatrixAttr()
    x.SetLocalXformFromUsd()
    x.SetWorldXformFromUsd()
    return x.GetFabricHierarchyLocalMatrixAttr()


def pose_matrix(x, y, z, qx, qy, qz, qw):
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz, wx, wy, wz = qx * qy, qx * qz, qy * qz, qw * qx, qw * qy, qw * qz
    return (1 - 2 * (yy + zz), 2 * (xy + wz), 2 * (xz - wy), 0.0,
            2 * (xy - wz), 1 - 2 * (xx + zz), 2 * (yz + wx), 0.0,
            2 * (xz + wy), 2 * (yz - wx), 1 - 2 * (xx + yy), 0.0,
            x, y, z, 1.0)


class View:
    def __init__(self, stage, n, W, fabric=False, animate=True):
        self.stage = stage
        self.n = n
        self.fabric = fabric
        self.animate = animate
        self.links = None
        self.mv = W.Movers(stage)
        self.pp = W.People(stage)
        self.sh = W.Shuttle(stage)

    def bind(self, names):
        self.links = []
        for i in range(self.n):
            ops = []
            for name in names:
                path = f"/World/Robots/R{i}/{name}"
                prim = self.stage.GetPrimAtPath(path)
                if not prim:
                    ops.append(None)
                elif self.fabric:
                    ops.append(fabric_matrix(path))
                else:
                    ops.append(UsdGeom.Xformable(prim).MakeMatrixXform())
            self.links.append(ops)
        print("[view] bound", self.n, "robots x", len(names), "links", "fabric" if self.fabric else "usd", flush=True)

    def apply(self, d):
        if self.links is None:
            self.bind(d["names"])
        b = d["b"]
        for i in range(min(self.n, b.shape[0])):
            for j, op in enumerate(self.links[i]):
                if op is None:
                    continue
                x, y, z, qx, qy, qz, qw = (float(v) for v in b[i, j])
                if self.fabric:
                    import usdrt

                    op.Set(usdrt.Gf.Matrix4d(*pose_matrix(x, y, z, qx, qy, qz, qw)))
                    continue
                m = Gf.Matrix4d().SetRotate(Gf.Quatd(qw, qx, qy, qz))
                m.SetTranslateOnly(Gf.Vec3d(x, y, z))
                op.Set(m)
        t = d["t"]
        for rop, rate in self.mv.spins:
            rop.Set((rate * t) % 360.0)
        for it, (x, y, deg) in zip(self.mv.items, d["m"]):
            it["top"].Set(Gf.Vec3d(x, y, 0.0))
            it["rop"].Set(deg)
        for it, (x, y, deg, ph) in zip(self.pp.items, d["p"]):
            if x is not None:
                it["top"].Set(Gf.Vec3d(x, y, 0.0))
                it["rop"].Set(deg)
            k = int((ph + it["m"].get("phase", 0.0)) % len(it["frames"]))
            if self.animate and k != it["last"]:
                tr, r, sc = it["frames"][k]
                it["attrs"][0].Set(tr)
                it["attrs"][1].Set(r)
                if sc is not None:
                    it["attrs"][2].Set(sc)
                it["last"] = k
        s = d["s"]
        if s is not None and self.sh.ok:
            x, y, head, lift, vis, fx, fy = s
            if vis:
                self.sh.p_ops[0].Set(Gf.Vec3d(fx, fy, lift))
                self.sh.p_ops[1].Set(math.degrees(head))
            self.sh.f_ops[0].Set(Gf.Vec3d(x, y, 0.0))
            self.sh.f_ops[1].Set(math.degrees(head) + 90.0)
            if self.sh.fork_op is not None:
                self.sh.fork_op.Set(Gf.Vec3d(0.0, 0.0, lift))
