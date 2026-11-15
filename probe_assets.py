import glob
import json
import os
import sys

from pxr import Usd, UsdGeom, UsdPhysics

root, names, out_path = sys.argv[1], [l.strip() for l in open(sys.argv[2]) if l.strip()], sys.argv[3]
out = {}
for name in names:
    usd = sorted(glob.glob(os.path.join(root, name, "simready_usd", "sm_*.usd*")))
    if not usd:
        print(f"{name:45s} MISSING")
        continue
    stage = Usd.Stage.Open(usd[0])
    dp = stage.GetDefaultPrim()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    r = cache.ComputeWorldBound(dp).ComputeAlignedRange()
    vs = dp.GetVariantSets()
    variants = {k: vs.GetVariantSet(k).GetVariantNames() for k in vs.GetNames()}
    selected = {k: vs.GetVariantSet(k).GetVariantSelection() for k in variants}
    rb = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI))
    col = sum(1 for p in stage.Traverse() if p.HasAPI(UsdPhysics.CollisionAPI))
    mass = None
    for p in stage.Traverse():
        if p.HasAPI(UsdPhysics.MassAPI):
            mass = UsdPhysics.MassAPI(p).GetMassAttr().Get()
            break
    out[name] = {"usd": os.path.relpath(usd[0], root), "prim": str(dp.GetPath()), "up": UsdGeom.GetStageUpAxis(stage),
                 "mpu": UsdGeom.GetStageMetersPerUnit(stage), "min": list(r.GetMin()), "max": list(r.GetMax()),
                 "variants": variants, "selected": selected, "rigid": rb, "colliders": col, "mass": mass}
    size = [round(b - a, 2) for a, b in zip(r.GetMin(), r.GetMax())]
    print(f"{name:42s} size={size} min={[round(v, 2) for v in r.GetMin()]} rb={rb} col={col} m={mass} {selected}")
json.dump(out, open(out_path, "w"), indent=1)
