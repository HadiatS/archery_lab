import json
import math
import os
import time

import cv2
import numpy as np

from .metrics import NOSE, LWR, RWR, Baseline, _d, arm_idx, compute, evaluate, resolve_target
from .report import compute_stats, write_reports
from .shot import ShotMachine
from .target import TargetScorer

VIEW = {"depan": "perp", "samping": "inline"}
SKEL = [(5, 6), (5, 11), (6, 12), (11, 12), (0, 1), (0, 2), (1, 3), (2, 4)]


def draw_pose(vis, kp, hand, thr=0.3):
    if kp is None:
        return
    (dsh, de, dwr), (bsh, be, bwr) = arm_idx(hand)

    def line(a, b, col):
        if kp[a, 2] >= thr and kp[b, 2] >= thr:
            cv2.line(vis, (int(kp[a, 0]), int(kp[a, 1])), (int(kp[b, 0]), int(kp[b, 1])), col, 3)
    for a, b in SKEL:
        line(a, b, (230, 230, 230))
    for a, b in ((dsh, de), (de, dwr)):
        line(a, b, (0, 165, 255))
    for a, b in ((bsh, be), (be, bwr)):
        line(a, b, (255, 200, 0))
    for i in range(17):
        if kp[i, 2] >= thr:
            cv2.circle(vis, (int(kp[i, 0]), int(kp[i, 1])), 4, (255, 255, 255), -1)


def _text(img, s, org, sc, col=(255, 255, 255)):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, sc, col, 2, cv2.LINE_AA)


def _resize_h(img, h):
    return cv2.resize(img, (int(img.shape[1] * h / img.shape[0]), h))


class Session:
    def __init__(self, cfg, out_dir, poses, target=None, emit=None, reference=None):
        self.cfg, self.out, self.poses, self.target = cfg, out_dir, poses, target
        self.emit = emit or (lambda e: None)
        os.makedirs(os.path.join(out_dir, "shots"), exist_ok=True)
        sc = cfg.get("shooter", {})
        self.hand_auto = sc.get("handedness", "right") == "auto"
        self.hand = "right" if self.hand_auto else sc.get("handedness", "right")
        self.hand_locked, self.anchor = False, sc.get("anchor", "cheek")
        self.ref_cfg, self.reference = cfg.get("reference", {}), reference or {}
        self.main = cfg.get("shot", {}).get("main_camera", "depan")
        if self.main not in poses:
            self.main = next(iter(poses), None)
        self.machine, self.baseline = ShotMachine(cfg), Baseline()
        tcfg = cfg.get("target", {})
        self.timeout, self.src_mode = tcfg.get("timeout_sec", 8), tcfg.get("source", "camera")
        self.scorer = target if target is not None else TargetScorer(tcfg)
        self.match = tcfg.get("match", "time")      # time | order (rekaman tak sinkron: cocokkan menurut urutan)
        self.queue = []                              # panah yang menunggu tembakan (mode urutan)
        self.shots, self.last_D, self.last_target_vis, self.t0, self.t_last = [], None, None, None, 0.0

    # ---------- pengaturan langsung ----------
    def update_settings(self, cfg):
        sc = cfg.get("shooter", {})
        self.anchor = sc.get("anchor", self.anchor)
        h = sc.get("handedness", "right")
        self.hand_auto = h == "auto"
        if not self.hand_auto:
            self.hand, self.hand_locked = h, False
        self.ref_cfg = cfg.get("reference", self.ref_cfg)

    # ---------- langkah per frame ----------
    def step(self, frames, t):
        if self.t0 is None:
            self.t0 = t
        self.t_last = t
        kps, mets, vis = {}, {}, {}
        for name, pose in self.poses.items():
            fr = frames.get(name)
            if fr is None:
                continue
            kps[name] = pose.update(fr)
            mets[name] = compute(kps[name], VIEW[name], self.hand, self.anchor) if kps[name] is not None else {}
            vis[name] = fr.copy()
        kpm, mm = kps.get(self.main), mets.get(self.main, {})
        obs = {}
        if kpm is not None and mm.get("D") is not None:
            (_, _, dwr), (_, _, bwr) = arm_idx(self.hand)
            obs = {"D": mm["D"], "scale": mm["_scale"], "dwr": kpm[dwr, :2].copy(), "bwr": kpm[bwr, :2].copy(),
                   "samples": {n: {k: v for k, v in m.items() if not k.startswith("_")} for n, m in mets.items() if m}}
            self.last_D = mm["D"]
        events = self.machine.update(t, obs) if self.main in frames else []

        for name in vis:
            draw_pose(vis[name], kps.get(name), self.hand)
            self._hud(vis[name], name, mets.get(name, {}))

        for typ, te, payload in events:
            if typ == "hold" and self.hand_auto and not self.hand_locked and kpm is not None and kpm[NOSE, 2] >= 0.3:
                dl, dr = _d(kpm[LWR, :2], kpm[NOSE, :2]), _d(kpm[RWR, :2], kpm[NOSE, :2])
                if abs(dl - dr) > 0.3 * mm.get("_scale", 1e9):
                    self.hand, self.hand_locked = ("left" if dl < dr else "right"), True
                    self.emit({"type": "info", "message": f"Tangan tarikan terdeteksi: {'kiri' if self.hand == 'left' else 'kanan'}"})
            elif typ == "release":
                self._new_shot(te, vis)
            elif typ == "shot_done":
                self._pose_done(payload)
            elif typ == "cancel":
                self.emit({"type": "cancel", "message": payload["reason"]})

        if self.target is not None and "target" in frames:
            pending = sum(1 for s in self.shots if s["arrow_status"] == "pending")
            accept = 99 if (self.match == "order" or not self.poses) else pending
            evs = self.target.process(frames["target"], t, accept) if self.src_mode == "camera" else []
            for e in evs:
                if e["type"] == "arrow":
                    a = {k: v for k, v in e.items() if k not in ("type", "_ref")}
                    s = next((x for x in self.shots if x["arrow_status"] == "pending"), None)
                    if s is None and not self.poses:                 # mode "skor saja": tiap panah = satu tembakan
                        s = self._new_shot(t, {}, pose=False)
                    if s is not None:
                        self._assign(s, a, e["_ref"])
                    else:
                        self.queue.append((a, e["_ref"]))           # tunggu tembakan berikutnya
                elif e["type"] == "removed":
                    self.emit({"type": "info", "message": "Anak panah diambil dari target"})
            tv = self.target.draw(frames["target"].copy())
            vis["target"] = self.last_target_vis = tv
            for s in self.shots:
                if s["arrow_status"] == "ok" and s["frames"].get("target") is None:
                    s["frames"]["target"] = tv.copy()
                elif s["arrow_status"] == "pending" and self.match != "order" and t - s["_t"] > self.timeout:
                    s["arrow_status"], s["frames"]["target"] = "tidak_terdeteksi", tv.copy()
                    self.emit({"type": "arrow", "no": s["no"], "score": None})
        for s in self.shots:
            if not s["done"] and s["rec"] is not None and s["arrow_status"] != "pending":
                self._complete(s)
        return vis

    def _tol_color(self, view, name, v):
        spec = (self.ref_cfg.get(view) or {}).get(name)
        if not spec:
            return (255, 255, 255)
        tg = resolve_target(spec, view, name, self.reference, self.baseline)
        if tg is None:
            return (255, 255, 255)
        return (0, 220, 0) if abs(v - tg) <= spec[1] else (60, 60, 255)

    def _hud(self, vis, name, m):
        sc = max(0.5, vis.shape[1] / 1280 * 0.7)
        head = f"{name.upper()}"
        if name == self.main:
            head += f" | {self.machine.state}" + (f" | D={self.last_D:.2f}" if self.last_D is not None else "")
        y = int(28 * sc)
        _text(vis, head, (10, y), sc, (0, 255, 255))
        for k in (self.ref_cfg.get(name) or {}):
            if k in m:
                y += int(26 * sc)
                _text(vis, f"{k}: {m[k]:.2f}", (10, y), sc * 0.8, self._tol_color(name, k, m[k]))

    # ---------- tembakan ----------
    def _assign(self, s, a, ref):
        a = dict(a)
        a["shot_no"] = ref["shot_no"] = s["no"]
        s["arrow"], s["arrow_status"] = a, "ok"
        self.emit({"type": "arrow", "no": s["no"], "score": a["score"]})

    def _new_shot(self, t, vis, pose=True):
        if self.src_mode == "camera":
            status = "pending" if (self.target is not None and self.target.ready) else "off"
        else:
            status = "menunggu"      # skor diisi manual / sensor eksternal lewat /api/score
        s = {"no": len(self.shots) + 1, "t_release": round(t - (self.t0 or t), 2), "_t": t, "frames": {k: v.copy() for k, v in vis.items()},
             "rec": None, "form": None, "details": [], "notes": [], "metrics": {}, "arrow": None,
             "arrow_status": status, "done": False, "image": None,
             "draw_time": None, "aim_time": None}
        if not pose:
            s["rec"] = {}                       # tidak ada analisa pose untuk tembakan ini
        self.shots.append(s)
        self.emit({"type": "release", "no": s["no"]})
        if pose and self.queue and s["arrow_status"] == "pending":
            a, ref = self.queue.pop(0)
            self._assign(s, a, ref)
        return s

    def _pose_done(self, rec):
        s = next((x for x in self.shots if x["rec"] is None), None)
        if s is None:
            return
        s["rec"] = rec
        s["draw_time"], s["aim_time"] = rec["draw_time"], rec["aim_time"]
        s["metrics"] = {v: d for v, d in rec["metrics"].items() if d}
        s["form"], s["details"], s["notes"] = evaluate(s["metrics"], self.ref_cfg, self.reference, self.baseline)
        self.baseline.add(s["metrics"])
        self.emit({"type": "form", "no": s["no"], "form": s["form"], "notes": s["notes"]})

    @staticmethod
    def _header(bar, s):
        a = s["arrow"]
        arrow = f"{a['score']}" if a else ("tidak terdeteksi" if s["arrow_status"] == "tidak_terdeteksi" else "-")
        form = "-" if s["form"] is None else f"{s['form']:.0f}/100"
        _text(bar, f"Tembakan #{s['no']}  |  Form {form}  |  Panah {arrow}", (12, 34), 0.8, (0, 255, 255))

    def _rewrite_header(self, s):
        if not s.get("image"):
            return
        path = os.path.join(self.out, "shots", s["image"])
        img = cv2.imread(path)
        if img is None:
            return
        img[:52] = 0
        self._header(img[:52], s)
        cv2.imwrite(path, img)

    def add_score(self, score=None, x=None, y=None, shot_no=None, source="eksternal"):
        """Isi skor anak panah dari sumber selain kamera. x,y = posisi relatif terhadap radius target
        (x kanan +, y bawah +). Return nomor tembakan."""
        if shot_no:
            s = next((q for q in self.shots if q["no"] == int(shot_no)), None)
        else:
            s = next((q for q in self.shots if q["arrow"] is None), None)
        if s is None:
            raise ValueError("Belum ada tembakan yang menunggu skor.")
        dx = dy = d = None
        sc = None
        if x is not None and y is not None:
            dx, dy = float(x), float(y)
            d = math.hypot(dx, dy)
            sc = self.scorer.score_of(d)
        if score is not None:
            sc = int(score)
        if sc is None:
            raise ValueError("Kirim 'score' atau posisi 'x' dan 'y'.")
        s["arrow"] = {"x": None, "y": None, "dx_rel": dx, "dy_rel": dy, "d_rel": d, "score": sc,
                      "shot_no": s["no"], "source": source}
        s["arrow_status"] = "ok"
        if s["done"]:
            self._rewrite_header(s)
        self.emit({"type": "arrow", "no": s["no"], "score": sc})
        return s["no"]

    def _compose(self, s):
        panels = [_resize_h(s["frames"][n], 420) for n in ("depan", "samping", "target") if s["frames"].get(n) is not None]
        if not panels:
            return None
        body = np.hstack(panels)
        bar = np.zeros((52, body.shape[1], 3), np.uint8)
        self._header(bar, s)
        out = np.vstack([bar, body])
        for i, n in enumerate(s["notes"][:2]):
            _text(out, n[:90], (12, out.shape[0] - 12 - 28 * (1 - i)), 0.6, (80, 200, 255))
        return out

    def _complete(self, s):
        img = self._compose(s)
        if img is not None:
            s["image"] = f"shot_{s['no']:03d}.jpg"
            cv2.imwrite(os.path.join(self.out, "shots", s["image"]), img)
        s["frames"], s["done"] = {}, True
        self.emit({"type": "shot", "no": s["no"]})

    # ---------- akses publik ----------
    def public_shots(self):
        pub = []
        for s in self.shots:
            pub.append({"no": s["no"], "t_release": s["t_release"], "form": s["form"], "notes": s["notes"],
                        "details": s["details"], "metrics": s["metrics"], "arrow": s["arrow"],
                        "arrow_status": s["arrow_status"], "draw_time": s["draw_time"], "aim_time": s["aim_time"],
                        "image": s["image"], "done": s["done"]})
        return pub

    def status(self):
        st = compute_stats(self.public_shots(), self.ref_cfg)
        return {"phase": self.machine.state, "D": self.last_D, "hand": self.hand, "hand_auto": self.hand_auto,
                "stats": st, "arrows": len(self.target.arrows) if self.target else 0}

    def set_reference_from_last(self):
        s = next((x for x in reversed(self.shots) if x["metrics"]), None)
        if s is None:
            return False
        ref = {v: {k: float(x) for k, x in d.items() if (self.ref_cfg.get(v) or {}).get(k)} for v, d in s["metrics"].items()}
        self.reference = ref
        return True

    def new_round(self):
        if self.target:
            self.target.arrows.clear()
            self.target.ref = None
            self.target.dirty = False
            self.target.stable = 0.0

    def finalize(self):
        for s in self.shots:
            if not s["done"]:
                if s["arrow_status"] == "pending":
                    s["arrow_status"] = "tidak_terdeteksi"
                if self.last_target_vis is not None and s["frames"].get("target") is None:
                    s["frames"]["target"] = self.last_target_vis.copy()
                self._complete(s)
        pub = self.public_shots()
        meta = {"durasi_detik": round(self.t_last - (self.t0 or self.t_last), 1), "tangan_tarik": self.hand}
        return write_reports(self.out, pub, self.ref_cfg, self.last_target_vis, meta)
