import argparse
import json
import math
import signal

import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

STATUS = {GoalStatus.STATUS_SUCCEEDED: "succeeded", GoalStatus.STATUS_ABORTED: "aborted", GoalStatus.STATUS_CANCELED: "canceled"}


class Robot:
    def __init__(self, node, i, route, log, timeout, stagger, start=None):
        self.node, self.i, self.route, self.log, self.timeout = node, i, route, log, timeout
        self.start = start or (0.0, 0.0, 0.0)
        self.pose = None
        self.mode = "route"
        self.aside = None
        self.hold_t = -100.0
        self.moved_t = 0.0
        node.create_subscription(Odometry, f"/go2_{i}/odom", self.on_odom, 10)
        self.client = ActionClient(node, NavigateToPose, f"/go2_{i}/navigate_to_pose")
        self.k = i % len(route) if stagger else 0
        self.busy = False
        self.sent = 0.0
        self.handle = None
        self.retry_at = 0.0

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def track_motion(self):
        if self.pose is None:
            return
        ref = getattr(self, "move_ref", None)
        if ref is None or math.dist(ref, self.pose[:2]) > 0.15:
            self.move_ref, self.moved_t = self.pose[:2], self.now()

    def on_odom(self, m):
        x0, y0, a0 = self.start
        q = m.pose.pose.orientation
        oyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        ox, oy = m.pose.pose.position.x, m.pose.pose.position.y
        c, s = math.cos(a0), math.sin(a0)
        self.pose = (x0 + c * ox - s * oy, y0 + s * ox + c * oy, a0 + oyaw)

    def cancel(self):
        self.gen = getattr(self, "gen", 0) + 1
        if self.handle is not None:
            self.handle.cancel_goal_async()
        self.handle = None
        self.busy = False

    def send(self, target=None):
        if target is not None:
            x, y, yaw = target
        else:
            x, y = self.route[self.k]
            nx, ny = self.route[(self.k + 1) % len(self.route)]
            yaw = math.atan2(ny - y, nx - x)
        g = NavigateToPose.Goal()
        g.pose = PoseStamped()
        g.pose.header.frame_id = "map"
        g.pose.header.stamp = self.node.get_clock().now().to_msg()
        g.pose.pose.position.x, g.pose.pose.position.y = float(x), float(y)
        g.pose.pose.orientation.z, g.pose.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
        self.busy, self.sent = True, self.now()
        self.target_is_route = target is None
        self.gen = getattr(self, "gen", 0) + 1
        gen = self.gen
        self.client.send_goal_async(g).add_done_callback(lambda f: self.on_accept(f, gen))

    def on_accept(self, fut, gen):
        h = fut.result()
        if gen != self.gen:
            if h.accepted:
                h.cancel_goal_async()
            return
        if not h.accepted:
            self.finish("rejected", gen)
            return
        self.handle = h
        h.get_result_async().add_done_callback(lambda f: self.finish(STATUS.get(f.result().status, str(f.result().status)), gen))

    def finish(self, status, gen=None):
        if gen is not None and gen != self.gen:
            return
        if not getattr(self, "target_is_route", True):
            self.busy, self.handle = False, None
            return
        self.log.append({"robot": self.i, "goal": self.k, "to": self.route[self.k], "status": status,
                         "sim_s": round(self.now() - self.sent, 2), "end": round(self.now(), 2)})
        print(f"[patrol] go2_{self.i} goal {self.k} {status} in {self.now() - self.sent:.1f} s", flush=True)
        if status == "succeeded" and self.pose is not None and math.dist(self.pose[:2], self.route[self.k]) > 0.8:
            self.log[-1]["status"] = status = "spurious"
            print(f"[patrol] go2_{self.i} goal {self.k} success ignored, {math.dist(self.pose[:2], self.route[self.k]):.1f} m away", flush=True)
        if status == "succeeded":
            if self.k in getattr(self, "turns", ()):
                self.turned_at = self.now()
            self.k = (self.k + 1) % len(self.route)
            self.retry_at = 0.0
        else:
            self.retry_at = self.now() + 3.0
        self.busy, self.handle = False, None

    def tick(self):
        if self.mode in ("hold", "wait", "yield", "narrow"):
            return
        if self.mode in ("aside", "pass"):
            if not self.busy and self.aside is not None and self.pose is not None and math.dist(self.pose[:2], self.aside[:2]) > 0.5:
                self.send(self.aside)
            return
        if not self.busy and self.now() >= self.retry_at:
            self.send()
        elif self.busy and self.handle is None and self.now() - self.sent > 8.0:
            print(f"[patrol] go2_{self.i} goal not acknowledged, resending", flush=True)
            self.cancel()
        elif self.handle is not None and self.now() - self.sent > self.timeout:
            self.handle.cancel_goal_async()
            self.handle = None


class FreeMap:
    def __init__(self, yaml_path):
        import os
        import numpy as np
        import yaml
        m = yaml.safe_load(open(yaml_path))
        img = os.path.join(os.path.dirname(yaml_path), m["image"])
        raw = open(img, "rb").read()
        parts, pos = [], 0
        while len(parts) < 4:
            while raw[pos:pos + 1].isspace():
                pos += 1
            if raw[pos:pos + 1] == b"#":
                pos = raw.index(b"\n", pos)
                continue
            end = pos
            while not raw[end:end + 1].isspace():
                end += 1
            parts.append(raw[pos:end])
            pos = end
        w, h = int(parts[1]), int(parts[2])
        pix = np.frombuffer(raw[pos + 1:pos + 1 + w * h], dtype=np.uint8).reshape(h, w)[::-1]
        free = pix > 250 if not m.get("negate", 0) else pix < 5
        self.free, self.res, self.ox, self.oy = free, float(m["resolution"]), float(m["origin"][0]), float(m["origin"][1])
        self.np = np

    def clearance_ok(self, x, y, r):
        np = self.np
        n = int(math.ceil(r / self.res))
        cx, cy = int((x - self.ox) / self.res), int((y - self.oy) / self.res)
        y0, y1, x0, x1 = cy - n, cy + n + 1, cx - n, cx + n + 1
        if y0 < 0 or x0 < 0 or y1 > self.free.shape[0] or x1 > self.free.shape[1]:
            return False
        win = self.free[y0:y1, x0:x1]
        yy, xx = np.mgrid[-n:n + 1, -n:n + 1]
        return bool(win[(xx * xx + yy * yy) <= n * n].all())

    def width(self, x, y, ux, uy, cap=3.0):
        key = (round(x * 4), round(y * 4), round(ux), round(uy))
        cache = self.__dict__.setdefault("_wcache", {})
        if key in cache:
            return cache[key]
        total = 0.0
        for sd in (1.0, -1.0):
            d = 0.0
            while d < cap:
                px, py = x - sd * uy * d, y + sd * ux * d
                i, j = int((py - self.oy) / self.res), int((px - self.ox) / self.res)
                if not (0 <= i < self.free.shape[0] and 0 <= j < self.free.shape[1]) or not self.free[i, j]:
                    break
                d += self.res
            total += d
        cache[key] = total
        return total

    def line_ok(self, a, b, r):
        d = math.dist(a, b)
        k = max(2, int(d / 0.2))
        return all(self.clearance_ok(a[0] + (b[0] - a[0]) * t / k, a[1] + (b[1] - a[1]) * t / k, r) for t in range(k + 1))


def seg_dist(a, b, p):
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = dx * dx + dy * dy
    t = 0.0 if L == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / L))
    return math.dist(p, (a[0] + t * dx, a[1] + t * dy))


def segment(r):
    a = r.route[(r.k - 1) % len(r.route)]
    b = r.route[r.k]
    d = math.dist(a, b) or 1.0
    return a, b, ((b[0] - a[0]) / d, (b[1] - a[1]) / d)


def along(u, p, q):
    return u[0] * (q[0] - p[0]) + u[1] * (q[1] - p[1])


def across(u, p, q):
    return u[0] * (q[1] - p[1]) - u[1] * (q[0] - p[0])


def plan_pass(fmap, lo, hi, lane_a, uh, ul):
    pl, ph = lo.pose[:2], hi.pose[:2]
    n_lo, n_hi = across(uh, lane_a, pl), across(uh, lane_a, ph)
    s_goal = along(uh, lane_a, hi.route[hi.k])

    def at(s, n):
        return (lane_a[0] + uh[0] * s - uh[1] * n, lane_a[1] + uh[1] * s + uh[0] * n)

    best = None
    offs = [o * sd for o in (1.45, 1.6, 1.75, 1.9) for sd in (1.0, -1.0)]
    for f in (0.0, 0.5, 1.0, 1.5, 2.0):
        s_lo = along(uh, lane_a, pl) + f * (ul[0] * uh[0] + ul[1] * uh[1])
        for n_l in [n_lo] + offs:
            if f == 0.0 and n_l != n_lo:
                continue
            spot = at(s_lo, n_l)
            if f > 0.0 and (not fmap.clearance_ok(spot[0], spot[1], CLEAR_PARK) or not fmap.line_ok(pl, spot, CLEAR_PATH)):
                continue
            for n_h in (n_l - 1.45, n_l + 1.45, n_l - 1.6, n_l + 1.6):
                for ds in (2.0, 1.5, 2.5, 3.0, 1.2):
                    sp = min(along(uh, lane_a, pl) + ds, s_goal)
                    via = at(sp, n_h)
                    if not fmap.clearance_ok(via[0], via[1], CLEAR_PARK) or not fmap.line_ok(ph, via, CLEAR_PATH):
                        continue
                    if seg_dist(ph, via, spot) < ROBOT_GAP or seg_dist(pl, spot, ph) < ROBOT_GAP:
                        continue
                    cost = abs(n_l - n_lo) + f * 0.2 + abs(n_h - n_hi) + 0.5 * abs(n_h) + 0.2 * abs(ds - 2.0)
                    if best is None or cost < best[0]:
                        best = (cost, spot, via)
                    break
    if best is None:
        return None, None
    yl, yh = math.atan2(ul[1], ul[0]), math.atan2(uh[1], uh[0])
    return (best[1][0], best[1][1], yl), (best[2][0], best[2][1], yh)


TURN_R = 0.5
TURN_CLEAR = 1.05


def turn_room(fmap, x, y, yaw):
    for side in (1.0, -1.0):
        cx, cy = x - side * TURN_R * math.sin(yaw), y + side * TURN_R * math.cos(yaw)
        if fmap.clearance_ok(cx, cy, TURN_CLEAR):
            return True
    return False


def turn_points(route, short=2.5):
    n, idx = len(route), set()

    def unit(p, q):
        d = math.dist(p, q)
        return ((q[0] - p[0]) / d, (q[1] - p[1]) / d) if d else None

    for k in range(n):
        a, b = route[k - 1], route[k]
        din = unit(a, b)
        if din is None:
            continue
        j = (k + 1) % n
        while True:
            dout = unit(route[j - 1], route[j])
            if dout is not None and din[0] * dout[0] + din[1] * dout[1] < -0.5:
                idx.add(k)
                break
            if math.dist(route[j - 1], route[j]) > short or j == k:
                break
            j = (j + 1) % n
    return idx


NARROW = 2.4
LOOK = 4.0


def narrow_span(fmap, r):
    a, b, u = segment(r)
    p = r.pose[:2]
    s_now = along(u, a, p)
    s_end = along(u, a, b)
    span, s = None, s_now - 1.0
    while s <= min(s_end, s_now + LOOK):
        x, y = a[0] + u[0] * s, a[1] + u[1] * s
        if fmap.width(x, y, u[0], u[1]) < NARROW:
            span = (span[0], s) if span else (s, s)
        elif span:
            break
        s += 0.25
    if span is None:
        return None
    return a, u, span


def yield_narrow(robots, fmap):
    for B in robots:
        if B.pose is None or B.mode not in ("route", "narrow"):
            continue
        ns = narrow_span(fmap, B) if B.mode == "route" else B.narrow
        if ns is None:
            continue
        a, u, (s0, s1) = ns
        sB = along(u, a, B.pose[:2])
        for A in robots:
            if A is B or A.pose is None:
                continue
            _, _, ua = segment(A)
            if ua[0] * u[0] + ua[1] * u[1] > -0.5 or abs(across(u, a, A.pose[:2])) > 1.3:
                continue
            sA = along(u, a, A.pose[:2])
            a_inside = s0 - 0.8 <= sA <= s1 + 0.8
            b_inside = s0 - 0.8 <= sB <= s1 + 0.8
            if B.mode == "route" and sA > max(sB, s0 - 0.8) and sA < s1 + 3.0 and not b_inside and (a_inside or (sA - s1) < (s0 - sB) or (abs((sA - s1) - (s0 - sB)) < 0.5 and A.i < B.i)):
                B.mode, B.narrow, B.yield_t = "narrow", ns, B.now()
                B.cancel()
                print(f"[patrol] go2_{B.i} waits before a narrow section for go2_{A.i}", flush=True)
            elif B.mode == "narrow" and (sA < s0 - 0.5 or B.now() - B.yield_t > 45.0):
                B.mode, B.narrow = "route", None
                B.retry_at = 0.0
                print(f"[patrol] go2_{B.i} enters the narrow section", flush=True)


def yield_turns(robots):
    for A in robots:
        if A.pose is None:
            continue
        ref = getattr(A, "yaw_ref", None)
        if ref is None or abs(math.atan2(math.sin(A.pose[2] - ref), math.cos(A.pose[2] - ref))) > 0.17:
            A.yaw_ref, A.yaw_moved = A.pose[2], A.now()
        k = A.k
        prev = (k - 1) % len(A.route)
        turning = k in A.turns and math.dist(A.pose[:2], A.route[k]) < 2.0
        turning = turning or (prev in A.turns and math.dist(A.route[prev], A.route[k]) <= 2.5)
        if not turning and getattr(A, "turned_at", -1e9) > A.now() - 40.0:
            _, _, ua = segment(A)
            turning = math.cos(A.pose[2]) * ua[0] + math.sin(A.pose[2]) * ua[1] < 0.7
        for B in robots:
            if B is A or B.pose is None or B.mode not in ("route", "yield"):
                continue
            _, _, ub = segment(B)
            ahead = along(ub, B.pose[:2], A.pose[:2])
            lateral = abs(across(ub, B.pose[:2], A.pose[:2]))
            if B.mode == "route" and turning and 0.0 < ahead < 3.0 and lateral < 1.3 and B.now() > getattr(B, "yield_block", 0.0):
                B.mode, B.yield_to, B.yield_t = "yield", A.i, A.now()
                B.cancel()
                print(f"[patrol] go2_{B.i} waits for go2_{A.i} to turn around", flush=True)
            elif B.mode == "yield" and B.yield_to == A.i and (not turning or B.now() - max(A.yaw_moved, A.moved_t) > 20.0 or B.now() - B.yield_t > 120.0):
                if turning:
                    B.yield_block = B.now() + 20.0
                B.mode = "route"
                B.retry_at = 0.0
                print(f"[patrol] go2_{B.i} resumes", flush=True)


def fit_turnarounds(fmap, route, tag):
    out = [list(p) for p in route]
    n = len(route)
    for k in range(n):
        a, b, c = route[k - 1], route[k], route[(k + 1) % n]
        din = (b[0] - a[0], b[1] - a[1])
        dout = (c[0] - b[0], c[1] - b[1])
        L1, L2 = math.hypot(*din), math.hypot(*dout)
        if L1 == 0 or L2 == 0 or (din[0] * dout[0] + din[1] * dout[1]) / (L1 * L2) > -0.5:
            continue
        u = (din[0] / L1, din[1] / L1)
        yaw = math.atan2(u[1], u[0])
        for back in [0.25 * j for j in range(int(min(4.0, L1 - 1.0) / 0.25) + 1)]:
            x, y = b[0] - u[0] * back, b[1] - u[1] * back
            if turn_room(fmap, x, y, yaw):
                if back > 0:
                    print(f"[patrol] {tag} turnaround {b} moved {back:.2f} m to ({x:.2f}, {y:.2f}) for turning room", flush=True)
                out[k] = [x, y]
                break
        else:
            print(f"[patrol] {tag} turnaround {b} has no turning room within 4 m", flush=True)
    return out


CLEAR_PARK = 0.75
CLEAR_PATH = 0.45
ROBOT_GAP = 1.25
STUCK = 12.0
MEET = 6.5
SAME_LANE = 1.2


def coordinate(robots, fmap, stuck_only=False):
    for a in range(len(robots)):
        for b in range(a + 1, len(robots)):
            hi, lo = robots[a], robots[b]
            if hi.pose is None or lo.pose is None:
                continue
            ph, pl = hi.pose[:2], lo.pose[:2]
            ha, hb, uh = segment(hi)
            _, _, ul = segment(lo)
            ahead = along(uh, ph, pl)
            same = abs(across(uh, ph, pl)) < SAME_LANE
            closing = ahead > 0 and ul[0] * uh[0] + ul[1] * uh[1] < -0.5
            now = hi.now()
            turns_short = along(uh, pl, hb) < 1.5
            facing_l = math.cos(lo.pose[2]) * ul[0] + math.sin(lo.pose[2]) * ul[1] > 0.0
            facing_h = math.cos(hi.pose[2]) * uh[0] + math.sin(hi.pose[2]) * uh[1] > 0.0
            meet = lo.mode == "route" and same and closing and ahead < MEET and facing_l and facing_h
            if stuck_only:
                meet = meet and ahead < 3.0 and now - lo.moved_t > STUCK and now - hi.moved_t > STUCK and hi.mode == "route"
            if meet and turns_short and not stuck_only:
                lo.mode, lo.wait_t = "wait", now
                lo.cancel()
                print(f"[patrol] go2_{lo.i} waits for go2_{hi.i} to turn", flush=True)
            elif lo.mode == "wait" and (ahead <= 0 or now - lo.wait_t > 40.0):
                lo.mode = "route"
                lo.retry_at = 0.0
                print(f"[patrol] go2_{lo.i} resumes (turned)", flush=True)
            elif meet:
                spot, via = plan_pass(fmap, lo, hi, ha, uh, ul)
                if spot is None:
                    lo.mode, lo.wait_t = "wait", now
                    lo.cancel()
                    print(f"[patrol] go2_{lo.i} waits for go2_{hi.i} (no free spot)", flush=True)
                else:
                    lo.aside, lo.via = spot, via
                    lo.mode, lo.aside_t = "aside", now
                    lo.cancel()
                    print(f"[patrol] go2_{lo.i} pulls aside to ({spot[0]:.1f}, {spot[1]:.2f}) for go2_{hi.i}", flush=True)
            passed = -ahead > 1.0
            if lo.mode == "aside" and (passed or now - lo.aside_t > 60.0):
                lo.mode, lo.aside = "route", None
                lo.cancel()
                lo.retry_at = 0.0
                print(f"[patrol] go2_{lo.i} resumes ({'passed' if passed else 'timeout'})", flush=True)
            parked = lo.mode == "aside" and math.dist(pl, lo.aside[:2]) < 0.35
            near = lo.mode == "aside" and abs(ahead) < 3.0 and not passed
            if hi.mode in ("pass", "hold") and (passed or lo.mode != "aside"):
                hi.mode, hi.aside = "route", None
                hi.cancel()
                hi.retry_at = 0.0
                print(f"[patrol] go2_{hi.i} continues", flush=True)
            elif near and not parked and hi.mode in ("route", "pass") and now - hi.hold_t > 40.0:
                hi.mode, hi.hold_t = "hold", now
                hi.cancel()
                print(f"[patrol] go2_{hi.i} holds for go2_{lo.i}", flush=True)
            elif lo.mode == "aside" and not passed and abs(ahead) < MEET and (
                    hi.mode == "route" or (hi.mode == "hold" and (parked or now - hi.hold_t > 20.0))):
                hi.aside = lo.via
                tx, ty = hi.aside[0], hi.aside[1]
                hi.mode = "pass"
                hi.cancel()
                print(f"[patrol] go2_{hi.i} passes via ({tx:.1f}, {ty:.2f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", required=True)
    ap.add_argument("--robots", type=int, default=1)
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--log", default="/tmp/fleet_patrol.json")
    ap.add_argument("--goal_timeout", type=float, default=120.0)
    ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--stagger", type=int, default=1)
    ap.add_argument("--coordinate", type=int, default=0)
    ap.add_argument("--map", default="")
    a = ap.parse_args()
    routes = [r["route"] for r in json.load(open(a.layout))["robots"]]
    stop = []
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    rclpy.init()
    node = rclpy.create_node("fleet_patrol", parameter_overrides=[Parameter("use_sim_time", value=True)])
    log = []
    starts = [r["start"] for r in json.load(open(a.layout))["robots"]]
    robots = [Robot(node, i, routes[i], log, a.goal_timeout, a.stagger, (starts[i][0], starts[i][1], math.radians(starts[i][2])))
              for i in range(a.first, a.first + a.robots)]
    for r in robots:
        while not stop and not r.client.wait_for_server(timeout_sec=2.0):
            print(f"[patrol] waiting for /go2_{r.i}/navigate_to_pose", flush=True)
    fmap = FreeMap(a.map) if a.map else None
    if fmap is not None:
        for r in robots:
            r.route = fit_turnarounds(fmap, r.route, f"go2_{r.i}")
    for r in robots:
        r.turns = turn_points(r.route)
    t0 = None
    while not stop:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = node.get_clock().now().nanoseconds * 1e-9
        if now <= 0:
            continue
        t0 = now if t0 is None else t0
        if a.duration and now - t0 > a.duration:
            break
        if fmap is not None and a.coordinate:
            coordinate(robots, fmap)
        else:
            for r in robots:
                r.track_motion()
            yield_turns(robots)
            yield_narrow(robots, fmap)
            coordinate(robots, fmap, stuck_only=True)
        for r in robots:
            r.tick()
        if len(log) and len(log) % 10 == 0:
            json.dump(log, open(a.log, "w"))
    json.dump(log, open(a.log, "w"))
    ok = sum(1 for e in log if e["status"] == "succeeded")
    print(f"[patrol] done: {ok} of {len(log)} goals succeeded", flush=True)


if __name__ == "__main__":
    main()
