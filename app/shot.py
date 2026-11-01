from collections import deque

import numpy as np


class ShotMachine:
    """Fase tembakan dari jarak dua pergelangan D (satuan torso) dan kecepatan tangan tarikan:
    IDLE -> DRAW -> HOLD(full draw) -> [lepas] -> FOLLOW -> selesai."""

    def __init__(self, cfg):
        c = cfg.get("shot", {})
        self.start_D, self.full_D = c.get("start_D", 1.2), c.get("full_D", 1.7)
        self.hold_min, self.min_aim = c.get("hold_min_sec", 0.15), c.get("min_aim_sec", 0.1)
        self.release_speed, self.follow_sec = c.get("release_speed", 2.0), c.get("follow_sec", 0.8)
        self.max_hold, self.letdown_sec = c.get("max_hold_sec", 15), 0.4
        self.reset()

    def reset(self):
        self.state, self.armed = "IDLE", True
        self.buf = deque()
        self.t_draw = self.t_full = self.t_hold = self.t_rel = self.t_low = None
        self.last_ok = 0.0

    def _speed(self, t):
        cur = self.buf[-1]
        for b in reversed(self.buf):
            if t - b[0] >= 0.1:
                dt = t - b[0]
                return float(np.hypot(*(cur[2] - b[2]))) / dt / cur[4]
        return 0.0

    def _cancel(self, t, why):
        self.state, self.armed, self.t_full, self.t_low = "IDLE", False, None, None
        return ("cancel", t, {"reason": why})

    @staticmethod
    def _median(samples, min_n=2):
        out = {}
        for view in {v for s in samples for v in s}:
            for name in {n for s in samples for n in s.get(view, {})}:
                vals = [s[view][name] for s in samples if name in s.get(view, {})]
                if len(vals) >= min_n:
                    out.setdefault(view, {})[name] = float(np.median(vals))
        return out

    def update(self, t, obs):
        ev = []
        D = obs.get("D")
        if D is None or obs.get("dwr") is None or obs.get("bwr") is None or not obs.get("scale"):
            if self.state in ("DRAW", "HOLD") and t - self.last_ok > 0.8:
                ev.append(self._cancel(t, "pemanah tidak terlihat"))
            return ev
        self.last_ok = t
        self.buf.append((t, D, np.asarray(obs["dwr"], float), np.asarray(obs["bwr"], float),
                         obs["scale"], obs.get("samples", {})))
        while self.buf and t - self.buf[0][0] > 3.0:
            self.buf.popleft()
        st = self.state
        if st == "IDLE":
            if D < self.start_D:
                self.armed = True
            elif self.armed:
                self.state, self.t_draw, self.t_full = "DRAW", t, None
                ev.append(("draw", t, {}))
        elif st == "DRAW":
            if D < self.start_D * 0.9:
                ev.append(self._cancel(t, "tarikan dibatalkan"))
            elif D >= self.full_D:
                if self.t_full is None:
                    self.t_full = t
                elif t - self.t_full >= self.hold_min:
                    self.state, self.t_hold, self.t_low = "HOLD", self.t_full, None
                    ev.append(("hold", t, {}))
            else:
                self.t_full = None
        elif st == "HOLD":
            if self._speed(t) >= self.release_speed and t - self.t_hold >= self.min_aim:
                ref = [b for b in self.buf if self.t_hold <= b[0] <= t - 0.1] or list(self.buf)[-3:-1]
                self.ref_dwr = np.median([b[2] for b in ref], axis=0)
                self.ref_bwr = np.median([b[3] for b in ref], axis=0)
                self.ref_S = float(np.median([b[4] for b in ref]))
                self.hold_samples = [b[5] for b in ref]
                self.t_rel, self.state, self.follow = t, "FOLLOW", []
                ev.append(("release", t, {}))
            elif D < self.full_D * 0.88:
                self.t_low = self.t_low or t
                if t - self.t_low > self.letdown_sec:
                    ev.append(self._cancel(t, "tarikan diturunkan"))
            else:
                self.t_low = None
            if self.state == "HOLD" and t - self.t_hold > self.max_hold:
                ev.append(self._cancel(t, "terlalu lama"))
        elif st == "FOLLOW":
            self.follow.append((obs["dwr"], obs["bwr"]))
            if t - self.t_rel >= self.follow_sec:
                ev.append(("shot_done", t, self._finish()))
        return ev

    def _finish(self):
        u = self.ref_bwr - self.ref_dwr
        u = u / (np.linalg.norm(u) + 1e-9)
        fol = {}
        if self.follow:
            fol["follow_back"] = float(max(np.dot(np.asarray(d) - self.ref_dwr, -u) for d, _ in self.follow) / self.ref_S)
            fol["bow_drop"] = float(max(np.asarray(b)[1] - self.ref_bwr[1] for _, b in self.follow) / self.ref_S)
        metrics = self._median(self.hold_samples)
        metrics["follow"] = fol
        rec = {"t_release": self.t_rel, "draw_time": self.t_hold - self.t_draw,
               "aim_time": self.t_rel - self.t_hold, "metrics": metrics}
        self.state, self.armed = "IDLE", False
        return rec
