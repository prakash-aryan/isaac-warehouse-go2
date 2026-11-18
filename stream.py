import math
import pickle
import socket

import numpy as np


class Sender:
    def __init__(self, ports, names):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dst = [("127.0.0.1", int(p)) for p in str(ports).split(",") if p]
        self.names = list(names)
        self.n = 0

    def send(self, ts, robot, movers, people, shuttle, clock=None):
        b = robot.data.body_link_pose_w.torch.detach().float().cpu().numpy()
        m = [(it["pos"][0], it["pos"][1], math.degrees(it["head"]) + it["yoff"]) for it in movers.items]
        p = []
        for it in people.items:
            mm = it["m"]
            walking = mm["mode"] in ("pingpong", "loop") and it["total"] > 0 and it["pos"] is not None
            x, y = (it["pos"][0], it["pos"][1]) if walking else (None, None)
            p.append((x, y, math.degrees(it["head"]) + mm.get("yaw_offset", 0.0), it["phase"]))
        s = None
        if shuttle.ok:
            kind, arg, *_ = shuttle.plan[shuttle.k]
            vis = shuttle.carry or (kind == "lift" and arg > 0.0)
            fx = shuttle.pos[0] + shuttle.ahead * math.cos(shuttle.head)
            fy = shuttle.pos[1] + shuttle.ahead * math.sin(shuttle.head)
            s = (shuttle.pos[0], shuttle.pos[1], shuttle.head, shuttle.lift, vis, fx, fy)
        msg = {"t": ts, "clock": ts if clock is None else clock, "n": self.n, "shape": b.shape, "b": b.tobytes(), "m": m, "p": p, "s": s, "names": self.names}
        self.n += 1
        data = pickle.dumps(msg, protocol=5)
        for dst in self.dst:
            try:
                self.sock.sendto(data, dst)
            except OSError:
                pass


class Receiver:
    def __init__(self, port):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.count = 0

    def latest(self):
        data = None
        while True:
            try:
                data = self.sock.recv(1 << 20)
                self.count += 1
            except BlockingIOError:
                break
        if data is None:
            return None
        d = pickle.loads(data)
        d["b"] = np.frombuffer(d["b"], dtype=np.float32).reshape(d["shape"])
        return d
