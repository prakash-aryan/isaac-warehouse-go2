import json
import math
import os
import random
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdSkel

INFO = json.load(open(sys.argv[1]))
SIMREADY = os.path.abspath(os.environ.get("SIMREADY_DIR", "assets/simready"))
for _a in INFO.values():
    _a["usd"] = os.path.join(SIMREADY, _a["usd"])
OUT = sys.argv[2]
NBOX = int(sys.argv[3]) if len(sys.argv) > 3 else 0
random.seed(7)
CELL, NX, NY = 10.0, 5, 4
W, D = CELL * NX, CELL * NY
DYN_TYPES = ["box_cardboard_b01", "box_cardboard_c01", "box_cardboard_d01", "box_corrugated_brown_b07",
             "box_corrugated_white_a08", "crate_blue_a01", "crate_yellow_a01", "warehouse_box_cube_a07"]

stage = Usd.Stage.CreateNew(OUT)
UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Warehouse").GetPrim())
for g in ("Building", "Racks", "Staging", "Process", "Props", "Movers", "People", "Dyn", "Lights"):
    UsdGeom.Xform.Define(stage, f"/Warehouse/{g}")
count = {}
foot = []
NO_FOOT = ("tape_", "lighting_", "fireProtection_", "ventilation_")


def place(name, x, y, yaw=0.0, z=0.0, parent="Props", anchor="center", physics=False, sx=1.0):
    a = INFO[name]
    lo, hi = Gf.Vec3d(*a["min"]), Gf.Vec3d(*a["max"])
    if anchor == "center":
        c = (lo + hi) * 0.5
        c = Gf.Vec3d(c[0] * sx, c[1], c[2])
        r = math.radians(yaw)
        ox = math.cos(r) * c[0] - math.sin(r) * c[1]
        oy = math.sin(r) * c[0] + math.cos(r) * c[1]
        t = Gf.Vec3d(x - ox, y - oy, z - lo[2])
    else:
        t = Gf.Vec3d(x, y, z)
    i = count.get(name, 0)
    count[name] = i + 1
    xf = UsdGeom.Xform.Define(stage, f"/Warehouse/{parent}/{name}_{i:03d}")
    xf.AddTranslateOp().Set(t)
    xf.AddRotateZOp().Set(yaw)
    if sx != 1.0:
        xf.AddScaleOp().Set(Gf.Vec3f(sx, 1.0, 1.0))
    prim = xf.GetPrim()
    if parent in ("Racks", "Staging", "Process", "Props") and anchor == "center" and not name.startswith(NO_FOOT):
        foot.append([round(x, 3), round(y, 3), round((hi[0] - lo[0]) * sx / 2, 3), round((hi[1] - lo[1]) / 2, 3), yaw])
    prim.GetReferences().AddReference(a["usd"])
    if "PhysX" in a["variants"]:
        prim.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Enabled" if physics else "Disabled")
    return prim


for i in range(NX):
    for j in range(NY):
        ox, oy = CELL * i, CELL * j
        west, east, south, north = i == 0, i == NX - 1, j == 0, j == NY - 1
        if (west or east) and (south or north):
            yaw, t = {(1, 0, 0, 1): (0.0, (ox, oy + CELL)), (1, 0, 1, 0): (90.0, (ox, oy)),
                      (0, 1, 1, 0): (180.0, (ox + CELL, oy)), (0, 1, 0, 1): (-90.0, (ox + CELL, oy + CELL))}[
                (west, east, south, north)]
            place("modular_warehouse_h10m_corner_in_a01", *t, yaw=yaw, parent="Building", anchor="origin",
                  z=-1.3 - INFO["modular_warehouse_h10m_corner_in_a01"]["min"][2])
        elif west or east or south or north:
            if west:
                yaw, t = 0.0, (ox, oy + CELL)
            elif east:
                yaw, t = 180.0, (ox + CELL, oy)
            elif south:
                yaw, t = 90.0, (ox, oy)
            else:
                yaw, t = -90.0, (ox + CELL, oy + CELL)
            name = "modular_warehouse_h10m_straight_a04"
            place(name, *t, yaw=yaw, parent="Building", anchor="origin", z=-1.3 - INFO[name]["min"][2])
        else:
            place("modular_warehouse_h10m_center_a01", ox, oy + CELL, parent="Building", anchor="origin",
                  z=-1.3 - INFO["modular_warehouse_h10m_center_a01"]["min"][2])

RACK_X0, RACK_N = 3.0, 11
RACK_W = 2.45
COMPOS = ["rack_composition_a04", "rack_composition_a05", "rack_composition_a06", "rack_composition_a08"]


def rack_row(y, yaw):
    for k in range(RACK_N):
        place(random.choice(COMPOS), RACK_X0 + RACK_W * (k + 0.5), y, yaw=yaw, parent="Racks")


rows = [(1.2, 0.0)]
aisles = []
y = 1.85
for _ in range(4):
    aisles.append(y + 1.8)
    y += 3.6
    rows += [(y + 0.65, 180.0), (y + 1.97, 0.0)]
    y += 2.62
aisles.append(y + 1.8)
y += 3.6
rows.append((y + 0.65, 180.0))
for ry, yaw in rows:
    rack_row(ry, yaw)
north_aisle = y + 1.3 + 1.5
for k in range(6):
    place("rack_large_empty_a01", 5.0 + 4.1 * k, D - 1.3, yaw=0.0, parent="Racks")
place("shelve_steel_heavyDuty_a06", 29.0, D - 0.6)
place("shelve_steel_heavyDuty_a06", 30.2, D - 0.6)
place("ladder_rolling_yellow_a02", 27.6, D - 2.9, yaw=20)

PAL = ["pallet_assembly_a02", "pallet_assembly_a05", "pallet_assembly_b01", "pallet_assembly_c01",
       "pallet_assembly_d01", "pallet_assembly_red_a01", "pallet_assembly_yellow_a01",
       "pallet_assembly_transparent_a01"]
stage_cols = [36.8, 39.4, 42.0, 44.6]
stage_rows = [3.6, 7.0, 10.0, 13.4]
for sx_ in stage_cols:
    for sy_ in stage_rows:
        place(random.choice(PAL), sx_, sy_, yaw=random.choice([0.0, 90.0]), parent="Staging")
for k in range(4):
    place("pallet_a01", 36.0 + 2.4 * k, 16.2, yaw=0.0, parent="Staging")
for k in range(8):
    place("barrier_fence_metal_a02", 35.0 + 1.25 * k, 17.6, parent="Staging")

conv_y = 23.6
for k in range(6):
    place("warehouse_conveyorBelt_a05", 35.4 + 2.0 * k, conv_y, parent="Process", anchor="center")
for x in (37.0, 40.0, 43.0):
    place("warehouse_packingTable_heavyDuty_c02", x, 21.0, parent="Process")
place("workstation_factory_a01", 47.6, 20.6, yaw=0, parent="Process")
place("vehicle_forklift_red_b01", 46.5, 37.4, yaw=90, parent="Process")
for k, n in enumerate(["drum_steel_a01", "drum_steel_a01", "warehouse_barrel_plastic_a01",
                       "warehouse_barrel_plastic_a01", "wasteManagement_trashBin_wheeled_a01"]):
    place(n, 36.0 + 0.8 * k, 38.8)
place("handTruck_convertible_steel_orange_a01", 41.2, 38.6, yaw=30)
place("tiltTruck_red_a04", 43.0, 38.4, yaw=90)
place("equipment_robot_trolley_c01", 38.0, 33.5, yaw=0)
place("sign_floor_movable_a01", 30.6, 18.0)
place("sign_warning_a04", 34.4, 20.5, yaw=90)
for x, yy in ((0.25, 12.0), (0.25, 28.0), (W - 0.25, 6.0), (W - 0.25, 30.0), (25.0, D - 0.25)):
    place("fireProtection_extinguisher_portable_a01", x, yy, z=1.2)
for yy in (0.4, 12.4, 24.4):
    place("tape_safety_yellow_black_12m15cm_f06", 30.1, yy, yaw=90, parent="Props", anchor="origin")
    place("tape_safety_yellow_black_12m15cm_f06", 34.0, yy, yaw=90, parent="Props", anchor="origin")
for xx in (34.6, 46.6):
    place("tape_safety_yellow_10m10cm_b05", xx, 18.3, yaw=0, parent="Props", anchor="origin", sx=1.2 if xx < 40 else 0.3)

movers = []
walls = [[W / 2, -0.5, W / 2 + 1, 0.5, 0.0], [W / 2, D + 0.5, W / 2 + 1, 0.5, 0.0],
         [-0.5, D / 2, 0.5, D / 2 + 1, 0.0], [W + 0.5, D / 2, 0.5, D / 2 + 1, 0.0]]


def mover(key, name, path, speed, mode, offset=0.0, yaw_offset=-90.0):
    p = f"/Warehouse/Movers/{key}"
    xf = UsdGeom.Xform.Define(stage, p)
    xf.AddTranslateOp().Set(Gf.Vec3d(path[0][0], path[0][1], 0.0))
    xf.AddRotateZOp().Set(0.0)
    a = INFO[name]
    body = stage.DefinePrim(p + "/body")
    body.GetReferences().AddReference(a["usd"])
    c = [(a["min"][k] + a["max"][k]) / 2 for k in range(2)]
    UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(-c[0], -c[1], -a["min"][2]))
    if "PhysX" in a["variants"]:
        body.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Disabled")
    movers.append({"prim": "/World" + p, "path": path, "speed": speed, "mode": mode, "offset": offset,
                   "yaw_offset": yaw_offset, "half": [round((a["max"][0] - a["min"][0]) / 2, 3),
                                                      round((a["max"][1] - a["min"][1]) / 2, 3)]})


mover("forklift_main", "vehicle_forklift_blue_a01", [[33.3, 3.0], [33.3, 36.0]], 1.0, "pingpong", yaw_offset=90.0)
mover("forklift_north", "vehicle_forklift_red_b01", [[5.0, north_aisle + 1.6], [28.0, north_aisle + 1.6]], 0.8,
      "pingpong", offset=8.0, yaw_offset=90.0)
mover("palletjack", "palletJack_heavyDuty_a01", [[1.6, 3.5], [1.6, 30.0]], 0.7, "pingpong", yaw_offset=90.0)
mover("trolley", "equipment_robot_trolley_a01", [[36.5, 31.5], [47.5, 31.5]], 0.6, "pingpong", yaw_offset=0.0)

SHUTTLE_X, SHUTTLE_A, SHUTTLE_B = 47.6, 3.6, 15.0
sp = "/Warehouse/Movers/shuttle"
xf = UsdGeom.Xform.Define(stage, sp)
xf.AddTranslateOp().Set(Gf.Vec3d(SHUTTLE_X, 8.0, 0.0))
xf.AddRotateZOp().Set(0.0)
a = INFO["vehicle_forklift_orange_a01"]
body = stage.DefinePrim(sp + "/body")
body.GetReferences().AddReference(a["usd"])
body.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Disabled")
cx, cy = [(a["min"][k] + a["max"][k]) / 2 for k in range(2)]
UsdGeom.Xformable(body).AddTranslateOp().Set(Gf.Vec3d(-cx, -cy, -a["min"][2]))
fork = stage.OverridePrim(sp + "/body/Geometry/sm_fork_a01_obj_00")
UsdGeom.Xformable(fork).AddTranslateOp(opSuffix="lift").Set(Gf.Vec3d(0.0, 0.0, 0.0))
fork_ahead = cy - (-1.53 + -0.18) / 2
pp = "/Warehouse/Movers/shuttle_pallet"
xf = UsdGeom.Xform.Define(stage, pp)
xf.AddTranslateOp().Set(Gf.Vec3d(SHUTTLE_X, SHUTTLE_A, 0.0))
xf.AddRotateZOp().Set(90.0)
pa = INFO["pallet_assembly_c01"]
pb = stage.DefinePrim(pp + "/body")
pb.GetReferences().AddReference(pa["usd"])
pb.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Disabled")
pcx, pcy = [(pa["min"][k] + pa["max"][k]) / 2 for k in range(2)]
UsdGeom.Xformable(pb).AddTranslateOp().Set(Gf.Vec3d(-pcx, -pcy, -pa["min"][2] + 0.04))
place("pallet_assembly_d01", SHUTTLE_X, SHUTTLE_B + 1.9, yaw=90.0, parent="Staging")
shuttle = {"forklift": "/World" + sp, "fork": "/World" + sp + "/body/Geometry/sm_fork_a01_obj_00",
           "pallet": "/World" + pp, "x": SHUTTLE_X, "a": SHUTTLE_A, "b": SHUTTLE_B, "fork_ahead": round(fork_ahead, 3),
           "lift": 0.28, "speed": 0.8, "standoff": 2.4,
           "half": [round((a["max"][0] - a["min"][0]) / 2, 3), round((a["max"][1] - a["min"][1]) / 2, 3)],
           "pallet_half": [round((pa["max"][0] - pa["min"][0]) / 2, 3), round((pa["max"][1] - pa["min"][1]) / 2, 3)]}
fans = []
for k, (x, yy) in enumerate(((10.0, 10.0), (10.0, 30.0), (25.0, 20.0), (42.0, 30.0))):
    p = f"/Warehouse/Movers/fan_{k}"
    xf = UsdGeom.Xform.Define(stage, p)
    xf.AddTranslateOp().Set(Gf.Vec3d(x, yy, 8.6))
    xf.AddRotateZOp().Set(0.0)
    body = stage.DefinePrim(p + "/body")
    body.GetReferences().AddReference(INFO["ventilation_ceilingFan_hvls_a01"]["usd"])
    body.GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Disabled")
    movers.append({"prim": "/World" + p, "path": [[x, yy]], "speed": 50.0 + 7 * k, "mode": "spin", "z": 8.6})


PEOPLE = os.path.abspath(os.environ.get("PEOPLE_DIR", "assets/people"))
CHARS = {
    "M_Construction_01": "Characters/original_male_adult_construction_01/male_adult_construction_01.usd",
    "M_Medical_01": "Characters/M_Medical_01/M_Medical_01.usd",
    "F_Business_02": "Characters/F_Business_02/F_Business_02.usd",
    "F_Medical_01": "Characters/F_Medical_01/F_Medical_01.usd",
}
FACING = 90.0
people = []
skel_rel, clip_joints, keep = {}, {}, []


def person(key, char, mode, path, speed=1.0, offset=0.0, heading=0.0, phase=0.0):
    p = f"/Warehouse/People/{key}"
    xf = UsdGeom.Xform.Define(stage, p)
    xf.AddTranslateOp().Set(Gf.Vec3d(path[0][0], path[0][1], 0.0))
    xf.AddRotateZOp().Set(heading + FACING)
    body = stage.DefinePrim(p + "/body")
    body.GetReferences().AddReference(f"{PEOPLE}/{CHARS[char]}")
    if char not in skel_rel:
        cs = Usd.Stage.Open(f"{PEOPLE}/{CHARS[char]}")
        keep.append(cs)
        sk = [q for q in cs.Traverse() if q.IsA(UsdSkel.Skeleton)][0]
        skel_rel[char] = str(sk.GetPath().MakeRelativePath(cs.GetDefaultPrim().GetPath()))
    clip = f"{PEOPLE}/baked/{char}_{'idle' if mode == 'idle' else 'walk'}.usda"
    if clip not in clip_joints:
        cst = Usd.Stage.Open(clip)
        keep.append(cst)
        clip_joints[clip] = UsdSkel.Animation(cst.GetDefaultPrim()).GetJointsAttr().Get()
    anim = UsdSkel.Animation.Define(stage, p + "/anim")
    anim.CreateJointsAttr(clip_joints[clip])
    skel = stage.OverridePrim(f"{p}/body/{skel_rel[char]}")
    UsdSkel.BindingAPI.Apply(skel).CreateAnimationSourceRel().SetTargets([anim.GetPath()])
    people.append({"prim": "/World" + p, "anim": "/World" + p + "/anim", "clip": clip, "mode": mode,
                   "path": path if mode != "idle" else [], "speed": speed, "offset": offset, "phase": phase,
                   "yaw_offset": FACING})


person("aisle_main", "M_Construction_01", "pingpong", [[32.2, 4.0], [32.2, 35.0]], 1.0, offset=12.0)
person("north_walk", "M_Construction_01", "pingpong", [[5.0, 36.7], [26.5, 36.7]], 0.9, offset=4.0)
person("packing_walk", "F_Business_02", "pingpong", [[35.6, 25.4], [47.4, 25.4]], 0.8)
person("aisle_2", "M_Medical_01", "pingpong", [[10.9, 9.87], [19.1, 9.87]], 0.9, offset=6.0)
person("aisle_4", "F_Medical_01", "pingpong", [[5.0, 22.31], [28.0, 22.31]], 0.85, offset=15.0)
person("yard_walk", "M_Construction_01", "pingpong", [[36.0, 8.5], [45.4, 8.5]], 0.9, offset=3.0)
for k, x in enumerate((37.0, 40.0, 43.0)):
    person(f"packer_{k}", ["M_Construction_01", "F_Business_02", "M_Medical_01"][k], "pingpong",
           [[x - 1.1, 22.1], [x + 1.1, 22.1]], 0.45 + 0.05 * k, offset=0.9 * k)
person("supervisor", "F_Business_02", "pingpong", [[35.6, 18.6], [43.4, 18.6]], 0.7, offset=2.0)

zones = [[44.6, 25.6, 48.6, 29.0], [35.6, 25.6, 39.6, 29.0]]
per = NBOX // len(DYN_TYPES)
dyn = []
for t, name in enumerate(DYN_TYPES):
    for i in range(per):
        z = zones[(t + i) % len(zones)]
        k = t * per + i
        x = z[0] + 0.5 + (k % 7) * (z[2] - z[0] - 1.0) / 6
        yy = z[1] + 0.5 + ((k // 7) % 5) * (z[3] - z[1] - 1.0) / 4
        h = 0.4 + 0.6 * (k // 35)
        p = f"/Warehouse/Dyn/{name}_{i:03d}"
        xf = UsdGeom.Xform.Define(stage, p)
        xf.AddTranslateOp().Set(Gf.Vec3d(x, yy, h))
        xf.AddRotateZOp().Set(random.uniform(0, 360))
        xf.GetPrim().GetReferences().AddReference(INFO[name]["usd"])
        xf.GetPrim().GetVariantSets().GetVariantSet("PhysX").SetVariantSelection("Enabled")
    if per:
        dyn.append(name)

n = 0
for i in range(10):
    for j in range(8):
        x, yy = 2.5 + 5.0 * i, 2.5 + 5.0 * j
        if (i + j) % 2 == 0:
            place("lighting_overheadLed_rectangularFrame_a01", x, yy, z=8.3, parent="Lights")
        L = UsdLux.RectLight.Define(stage, f"/Warehouse/Lights/rect_{n:03d}")
        L.CreateWidthAttr(2.0)
        L.CreateHeightAttr(0.6)
        L.CreateIntensityAttr(16000.0)
        L.CreateColorTemperatureAttr(5000.0)
        L.CreateEnableColorTemperatureAttr(True)
        UsdGeom.Xformable(L).AddTranslateOp().Set(Gf.Vec3d(x, yy, 8.25))
        n += 1
UsdLux.DomeLight.Define(stage, "/Warehouse/Lights/dome").CreateIntensityAttr(450.0)

hulls = 0
for root in stage.GetPrimAtPath("/Warehouse/Dyn").GetChildren():
    for p in Usd.PrimRange(root):
        if p.IsInstance():
            p.SetInstanceable(False)
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.MeshCollisionAPI):
            UsdPhysics.MeshCollisionAPI(p).CreateApproximationAttr().Set("convexHull")
            hulls += 1
print("convex hull colliders", hulls)
kin = joints = scenes = 0
for prim in list(stage.Traverse()):
    if str(prim.GetPath()).startswith("/Warehouse/Dyn"):
        continue
    if prim.IsA(UsdPhysics.Scene):
        prim.SetActive(False)
        scenes += 1
    elif prim.IsA(UsdPhysics.Joint):
        prim.SetActive(False)
        joints += 1
for prim in stage.Traverse():
    if str(prim.GetPath()).startswith("/Warehouse/Dyn"):
        continue
    for api in (UsdPhysics.RigidBodyAPI, UsdPhysics.ArticulationRootAPI, UsdPhysics.MassAPI):
        if prim.HasAPI(api):
            prim.RemoveAPI(api)
            kin += api is UsdPhysics.RigidBodyAPI
nocol = 0
for root in stage.GetPrimAtPath("/Warehouse/Movers").GetChildren():
    for p in Usd.PrimRange(root):
        if p.IsInstance():
            p.SetInstanceable(False)
    for p in Usd.PrimRange(root):
        if p.HasAPI(UsdPhysics.CollisionAPI):
            UsdPhysics.CollisionAPI(p).CreateCollisionEnabledAttr().Set(False)
            nocol += 1
print("mover colliders disabled", nocol)
print("removed rigid bodies", kin, "joints", joints, "physics scenes", scenes)
stage.GetRootLayer().Save()

robots = [{"start": [31.4, 2.6, 90.0], "route": [[31.4, 36.0], [31.4, 3.0]], "role": "hero"}]
COLUMN_LINES = (10.0, 20.0, 30.0)


def lane(y):
    for yc in COLUMN_LINES:
        if abs(y - yc) < 0.9:
            return yc + math.copysign(0.9, y - yc)
    return y


for a in aisles:
    lo, hi = lane(a - 0.8), lane(a + 0.8)
    dh = hi - lo
    loop = [[4.6, lo], [28.4, lo], [28.4, hi], [4.6, hi]]
    L = 2 * (28.4 - 4.6) + 2 * dh
    for k in range(3):
        s = k * L / 3
        if s < 23.8:
            st, w = [4.6 + s, lo, 0.0], 1
        elif s < 23.8 + dh:
            st, w = [28.4, lo + (s - 23.8), 90.0], 2
        else:
            st, w = [28.4 - (s - 23.8 - dh), hi, 180.0], 3
        robots.append({"start": st, "route": loop[w:] + loop[:w], "role": "aisle"})
yard = [[35.8, 5.3], [46.1, 5.3], [46.1, 11.7], [35.8, 11.7]]
for k, st in enumerate(([36.5, 5.3, 0.0], [46.1, 8.0, 90.0], [42.0, 11.7, 180.0])):
    robots.append({"start": st, "route": yard[k + 1:] + yard[:k + 1], "role": "yard"})
nl = [[4.6, north_aisle - 0.7], [28.4, north_aisle - 0.7], [28.4, north_aisle + 0.3], [4.6, north_aisle + 0.3]]
robots.append({"start": [8.0, north_aisle - 0.7, 0.0], "route": nl[1:] + nl[:1], "role": "north"})
robots.append({"start": [24.0, north_aisle + 0.3, 180.0], "route": nl[3:] + nl[:3], "role": "north"})

hero = robots[0]["start"]
yaw0 = math.radians(hero[2])
goals_world = [[31.4, 20.0], [16.0, aisles[3]], [4.4, aisles[3]], [4.4, aisles[1]], [16.0, aisles[1]],
               [31.4, aisles[1]], [41.0, 8.5], [41.0, 27.0], [31.4, north_aisle], [16.0, north_aisle],
               [31.4, 30.0], [31.4, 3.0]]
goals_map = []
for gx, gy in goals_world:
    dx, dy = gx - hero[0], gy - hero[1]
    goals_map.append([round(math.cos(yaw0) * dx + math.sin(yaw0) * dy, 3),
                      round(-math.sin(yaw0) * dx + math.cos(yaw0) * dy, 3)])

layout = {"size": [W, D], "people": people, "shuttle": shuttle, "footprints": foot + walls, "aisles": aisles, "north_aisle": north_aisle, "robots": robots, "movers": movers,
          "boxes": dyn, "drop_zones": zones, "drop_batch": 24, "lidar_forward": 0.2, "lidar_height": 0.45,
          "hero_goals_map": goals_map, "hero_goals_world": goals_world}
json.dump(layout, open(OUT.replace(".usda", "_layout.json"), "w"), indent=1)
json.dump(goals_map, open(OUT.replace(".usda", "_goals.json"), "w"))
print("prims", sum(count.values()), "dyn", per * len(DYN_TYPES), "lights", n, "bodies removed", kin,
      "robot slots", len(robots), "movers", len(movers), "people", len(people))
