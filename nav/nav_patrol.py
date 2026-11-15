import argparse
import json
import math
import signal
import time

import rclpy
import rclpy.parameter
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped
from lifecycle_msgs.msg import Transition
from lifecycle_msgs.srv import ChangeState, GetState
from nav2_msgs.srv import ManageLifecycleNodes
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.signals import SignalHandlerOptions
from tf2_msgs.msg import TFMessage

from nav2_start import ACTIVATE, CONFIGURE, MANAGER, NODES, next_transition


class Nav2Start:

    def __init__(self, nav):
        self.nav = nav
        self.get = {n: nav.create_client(GetState, f"{n}/get_state") for n in NODES}
        self.change = {n: nav.create_client(ChangeState, f"{n}/change_state") for n in NODES}
        self.manager = nav.create_client(ManageLifecycleNodes, f"{MANAGER}/manage_nodes")
        self.tf_seen = False
        self.tf_sub = nav.create_subscription(TFMessage, "/tf", self.on_tf, 50)

    def on_tf(self, msg):
        if any(t.header.frame_id == "odom" and t.child_frame_id == "base_link" for t in msg.transforms):
            self.tf_seen = True

    def call(self, client, request, timeout):
        if not client.wait_for_service(timeout_sec=1.0):
            return None
        fut = client.call_async(request)
        rclpy.spin_until_future_complete(self.nav, fut, timeout_sec=timeout)
        if not fut.done():
            client.remove_pending_request(fut)
            return None
        return fut.result()

    def states(self, timeout=2.0):
        out = {}
        for n in NODES:
            r = self.call(self.get[n], GetState.Request(), timeout)
            out[n] = r.current_state.label if r is not None else "unknown"
        return out

    def run(self, stopped, delay, manager_timeout, call_timeout, rounds):
        nav, info = self.nav, {"mode": "not active", "direct": 0, "rounds": 0, "wall": 0.0}
        while not stopped() and not (self.tf_seen and nav.get_clock().now().nanoseconds > 0):
            rclpy.spin_once(nav, timeout_sec=0.2)
        nav.destroy_subscription(self.tf_sub)
        seen = time.monotonic()
        while not stopped() and time.monotonic() - seen < delay:
            rclpy.spin_once(nav, timeout_sec=0.2)
        t0 = time.monotonic()
        states = self.states()
        if next_transition(states)[0] != "done" and not stopped():
            if self.manager.wait_for_service(timeout_sec=1.0):
                self.manager.call_async(ManageLifecycleNodes.Request(command=ManageLifecycleNodes.Request.STARTUP))
            else:
                print(f"NAV2 start: no {MANAGER}/manage_nodes service", flush=True)
            while not stopped() and time.monotonic() - t0 < manager_timeout:
                rclpy.spin_once(nav, timeout_sec=1.0)
                states = self.states()
                if next_transition(states)[0] == "done":
                    break
        if next_transition(states)[0] == "done":
            info["mode"] = "manager"
        else:
            print(f"NAV2 start: not active {manager_timeout:.0f} s after the manager's STARTUP request, states "
                  f"{json.dumps(states)}; starting the servers directly", flush=True)
            for r in range(1, rounds + 1):
                if stopped():
                    break
                info["rounds"] = r
                for _ in range(2 * len(NODES) + 2):
                    action, node = next_transition(states)
                    if action in ("done", "fail") or stopped():
                        break
                    if action == "wait":
                        rclpy.spin_once(nav, timeout_sec=call_timeout)
                        states = self.states()
                        if next_transition(states) == (action, node):
                            break
                        continue
                    transition = CONFIGURE if action == "configure" else ACTIVATE
                    res = self.call(self.change[node], ChangeState.Request(transition=Transition(id=transition)), call_timeout)
                    info["direct"] += 1
                    states = self.states()
                    print(f"NAV2 start: round {r} {action} {node}: response "
                          f"{'none' if res is None else res.success}, state {states[node]}", flush=True)
                    if next_transition(states) == (action, node):
                        break
                if next_transition(states)[0] in ("done", "fail"):
                    break
                rclpy.spin_once(nav, timeout_sec=2.0)
                states = self.states()
            if next_transition(states)[0] == "done":
                info["mode"] = "direct"
        info["wall"] = time.monotonic() - t0
        info["states"] = states
        print(f"NAV2 start: {info['mode']} after {info['wall']:.1f} s of wall time, {info['direct']} direct transitions "
              f"in {info['rounds']} rounds", flush=True)
        return info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--goals", required=True)
    p.add_argument("--log", default="")
    p.add_argument("--goal-timeout", type=float, default=180.0)
    p.add_argument("--laps", type=int, default=0)
    args = p.parse_args()
    goals = json.load(open(args.goals))
    stop = []
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.append(1))
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    nav = BasicNavigator()
    nav.set_parameters([rclpy.parameter.Parameter("use_sim_time", value=True)])
    start = Nav2Start(nav).run(lambda: bool(stop), 5.0, 30.0, 10.0, 3)
    if start["mode"] == "not active":
        raise SystemExit(2)
    log, lap, k = [], 0, 0
    while not stop and (args.laps == 0 or lap < args.laps):
        gx, gy = goals[k]
        g = PoseStamped()
        g.header.frame_id = "map"
        g.header.stamp = Time()
        g.pose.position.x, g.pose.position.y = gx, gy
        g.pose.orientation.w = 1.0
        nav.goToPose(g)
        t0 = nav.get_clock().now().nanoseconds * 1e-9
        n = 0
        while not nav.isTaskComplete() and not stop:
            rclpy.spin_once(nav, timeout_sec=0.2)
            now = nav.get_clock().now().nanoseconds * 1e-9
            n += 1
            fb = nav.getFeedback()
            if fb and n % 50 == 0:
                print(f"NAV goal {k} remaining {fb.distance_remaining:.1f} m recoveries {fb.number_of_recoveries}", flush=True)
            if now - t0 > args.goal_timeout:
                nav.cancelTask()
                break
        r = nav.getResult()
        dt = nav.get_clock().now().nanoseconds * 1e-9 - t0
        log.append({"lap": lap, "goal": k, "xy": [gx, gy], "result": r.name if r else "TIMEOUT", "sim_s": round(dt, 1)})
        print("NAV", json.dumps(log[-1]), flush=True)
        if args.log:
            json.dump(log, open(args.log, "w"))
        if log[-1]["result"] != "SUCCEEDED":
            time.sleep(5.0)
        k = (k + 1) % len(goals)
        lap += k == 0
    nav.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
