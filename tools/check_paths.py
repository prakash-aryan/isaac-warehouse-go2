import json
import math
import sys

import numpy as np
from pxr import Usd, UsdGeom

scene, out = sys.argv[1], sys.argv[2]
Z0, Z1, RES = 0.3, 1.8, 0.1
SKIP = ("/Warehouse/Movers", "/Warehouse/People", "/Warehouse/Dyn", "/Warehouse/Lights", "/Warehouse/Fans")
stage = Usd.Stage.Open(scene)
lay = json.load(open(scene.replace(".usda", "_layout.json")))
W, D = lay["size"]
nx, ny = int((W + 4) / RES), int((D + 4) / RES)
occ = np.zeros((nx, ny), dtype=bool)


def mark(pts):
    m = (pts[:, 2] >= Z0) & (pts[:, 2] <= Z1)
    p = pts[m]
    ix = np.floor((p[:, 0] + 2) / RES).astype(int)
    iy = np.floor((p[:, 1] + 2) / RES).astype(int)
    k = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    occ[ix[k], iy[k]] = True


meshes = 0
for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
    path = str(prim.GetPath())
    if not prim.IsA(UsdGeom.Mesh) or path.startswith(SKIP):
        continue
    if UsdGeom.Imageable(prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
        continue
    mesh = UsdGeom.Mesh(prim)
    pts = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    idx = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not counts or not idx:
        continue
    xf = np.array(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))
    P = np.asarray(pts, dtype=np.float64)
    P = P @ xf[:3, :3] + xf[3, :3]
    if P[:, 2].max() < Z0 or P[:, 2].min() > Z1:
        continue
    counts = np.asarray(counts)
    idx = np.asarray(idx)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    tris = []
    for c in np.unique(counts):
        s = starts[counts == c]
        for j in range(1, c - 1):
            tris.append(np.stack([idx[s], idx[s + j], idx[s + j + 1]], axis=1))
    T = np.concatenate(tris)
    A, B, C = P[T[:, 0]], P[T[:, 1]], P[T[:, 2]]
    zmin = np.minimum(np.minimum(A[:, 2], B[:, 2]), C[:, 2])
    zmax = np.maximum(np.maximum(A[:, 2], B[:, 2]), C[:, 2])
    keep = (zmax >= Z0) & (zmin <= Z1)
    A, B, C = A[keep], B[keep], C[keep]
    if not len(A):
        continue
    meshes += 1
    span = np.maximum(np.maximum(np.linalg.norm(B - A, axis=1), np.linalg.norm(C - A, axis=1)), np.linalg.norm(C - B, axis=1))
    n = np.clip(np.ceil(span / (RES * 0.7)).astype(int), 1, 400)
    for k in np.unique(n):
        sel = n == k
        u, v = np.meshgrid(np.arange(k + 1), np.arange(k + 1))
        u, v = u.ravel() / k, v.ravel() / k
        g = u + v <= 1
        u, v = u[g], v[g]
        a, b, c = A[sel], B[sel], C[sel]
        pts_s = a[:, None, :] + u[None, :, None] * (b - a)[:, None, :] + v[None, :, None] * (c - a)[:, None, :]
        mark(pts_s.reshape(-1, 3))

np.savez_compressed(out + ".npz", occ=occ, res=RES, origin=-2.0)


def clear(x, y, r):
    i0, i1 = int((x - r + 2) / RES), int((x + r + 2) / RES)
    j0, j1 = int((y - r + 2) / RES), int((y + r + 2) / RES)
    for i in range(max(0, i0), min(nx, i1 + 1)):
        for j in range(max(0, j0), min(ny, j1 + 1)):
            if occ[i, j] and math.hypot((i + 0.5) * RES - 2 - x, (j + 0.5) * RES - 2 - y) < r:
                return False
    return True


def check(name, pts, r, loop=False):
    pts = [tuple(p) for p in pts] + ([tuple(pts[0])] if loop else [])
    bad = []
    for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        L = math.hypot(bx - ax, by - ay)
        for s in np.arange(0, L + 1e-6, 0.1):
            x, y = ax + (bx - ax) * s / max(L, 1e-9), ay + (by - ay) * s / max(L, 1e-9)
            if not clear(x, y, r):
                bad.append((round(float(x), 1), round(float(y), 1)))
    print(f"{name:14s} {'OK' if not bad else 'HIT %d samples, e.g. %s' % (len(bad), bad[:: max(1, len(bad) // 6)][:6])}")


print("meshes rasterised", meshes, "occupied cells", int(occ.sum()))
for m in lay["people"]:
    check("person " + m["prim"].split("/")[-1], m["path"], 0.3)
for i, r in enumerate(lay["robots"]):
    check(f"robot {i}", r["route"], 0.3, loop=True)
for m in lay["movers"]:
    if m.get("mode") != "spin":
        check("mover " + m["prim"].split("/")[-1], m["path"], max(m.get("half", [0.5, 0.5])))
