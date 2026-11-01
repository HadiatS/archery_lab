import json
import os
import queue
import threading
import time
import traceback
from datetime import datetime

import cv2

from .estimator import CamPose, Models
from .session import Session
from .source import Source, to_target
from .target import TargetScorer


class Engine:
    """Menjalankan sesi di thread latar (kamera langsung ATAU analisa rekaman video) -> Session; MJPEG & event."""

    def __init__(self):
        self.lock = threading.RLock()
        self.cond = threading.Condition()
        self.subs, self.jpeg, self.raw = [], {}, {}
        self.thread, self.stop_ev = None, threading.Event()
        self.state, self.message = "idle", ""
        self.session, self.out_dir, self.active, self.cfg = None, None, [], None
        self.offline, self.progress = False, None
        self._models = None

    # ---------- event ----------
    def emit(self, ev):
        msg = json.dumps(ev, ensure_ascii=False, default=float)
        with self.lock:
            for q in self.subs:
                q.put(msg)

    def subscribe(self):
        q = queue.Queue()
        with self.lock:
            self.subs.append(q)
        return q

    def unsubscribe(self, q):
        with self.lock:
            if q in self.subs:
                self.subs.remove(q)

    def _set(self, state, message=""):
        self.state, self.message = state, message
        self.emit({"type": "engine", "state": state, "message": message})

    # ---------- kontrol ----------
    @property
    def running(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self, cfg, reference=None):
        if self.running:
            return False
        self.cfg, self.stop_ev = cfg, threading.Event()
        self.offline, self.progress = bool(cfg.get("_offline")), None
        self.thread = threading.Thread(target=self._run, args=(cfg, reference), daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.stop_ev.set()
        if self.thread:
            self.thread.join(timeout=30)

    def _make_poses(self, cfg, names):
        if self._models is None:
            self._models = Models(cfg)
            if getattr(self._models.pose, "fallback", None):
                self.emit({"type": "info", "message": self._models.pose.fallback})
        det_every = cfg["detector"].get("det_every", 5)
        return {n: CamPose(self._models, (cfg.get("archer_roi") or {}).get(n), det_every,
                           cfg.get("pose", {}).get("smooth", 0.5)) for n in names}

    # ---------- loop utama ----------
    def _run(self, cfg, reference):
        srcs, session = {}, None
        try:
            self._set("loading", "Memuat model dan membuka " + ("rekaman…" if self.offline else "kamera…"))
            mode = cfg.get("analysis_mode", "both")            # both | pose | score
            specs = {n: to_target(c.get("mode"), c.get("value")) for n, c in cfg["cameras"].items()}
            if mode == "pose":
                specs["target"] = None
            if mode == "score":
                specs["depan"] = specs["samping"] = None
            active = {n: s for n, s in specs.items() if s}
            pose_cams = [n for n in ("depan", "samping") if n in active]
            if mode == "score":
                if "target" not in active:
                    raise RuntimeError("Mode 'Skor saja' butuh video/kamera target.")
            elif not pose_cams:
                raise RuntimeError("Aktifkan minimal satu kamera/rekaman pose (Depan atau Samping).")
            poses = self._make_poses(cfg, pose_cams) if pose_cams else {}
            target = TargetScorer(cfg["target"]) if "target" in active else None
            if target is not None and not target.ready:
                self.emit({"type": "info", "message": "Target belum dikalibrasi (tab Target) — skor panah belum dihitung."})
            for n, (conn, tg) in active.items():
                srcs[n] = Source(conn, tg, cfg.get("usb"))
            self.active = list(active)
            self.out_dir = os.path.join(cfg.get("output_dir", "data"), datetime.now().strftime("%Y%m%d_%H%M%S"))
            session = Session(cfg, self.out_dir, poses, target, emit=self.emit, reference=reference)
            self.session, self.jpeg, self.raw = session, {}, {}
            master = cfg.get("shot", {}).get("main_camera", "depan")
            master = master if master in srcs else (pose_cams[0] if pose_cams else "target")
            offsets = cfg.get("offsets") or {}
            total = srcs[master].duration if srcs[master].conn == "file" else 0.0
            self._set("running")
            skip = max(int(cfg.get("frame_skip", 1)), 1)
            i, t_first, wall0, last_prog = 0, None, time.time(), 0.0
            while not self.stop_ev.is_set():
                ok, f, t = srcs[master].read()
                if not ok:
                    if srcs[master].ended:
                        break
                    time.sleep(0.005)
                    continue
                fresh = {master: f}
                for n, s in srcs.items():
                    if n == master:
                        continue
                    if s.live:
                        ok2, f2, _ = s.read()
                        if ok2:
                            fresh[n] = f2
                    else:                       # rekaman: samakan waktu dengan master (+ offset)
                        tt = t - float(offsets.get(n, 0.0))
                        if tt < 0:
                            continue
                        for _ in range(12):
                            ok2, f2, t2 = s.read()
                            if not ok2:
                                break
                            fresh[n] = f2
                            if t2 >= tt - 0.02:
                                break
                i += 1
                if i % skip:
                    continue
                t_first = t if t_first is None else t_first
                if srcs[master].conn == "file":
                    now = time.time()
                    if total > 0 and now - last_prog >= 0.8:
                        last_prog = now
                        done = min(1.0, (t - t_first) / total)
                        el = now - wall0
                        self.progress = {"pct": round(100 * done, 1), "t": round(t - t_first, 1), "total": round(total, 1),
                                         "eta": round(el * (1 - done) / done) if done > 0.02 else None}
                    if cfg.get("realtime_files", False):
                        d = (t - t_first) - (now - wall0)
                        if d > 0:
                            time.sleep(min(d, 0.5))
                vis = session.step(fresh, t)
                for n, im in vis.items():
                    if n in fresh:
                        self.raw[n] = fresh[n]
                    h, w = im.shape[:2]
                    if w > 960:
                        im = cv2.resize(im, (960, int(h * 960 / w)))
                    ok3, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok3:
                        with self.cond:
                            self.jpeg[n] = (buf.tobytes(), self.jpeg.get(n, (None, 0))[1] + 1)
                            self.cond.notify_all()
            if self.progress is not None and not self.stop_ev.is_set():
                self.progress = {**self.progress, "pct": 100.0, "eta": 0}
        except Exception as e:   # noqa: BLE001
            traceback.print_exc()
            self.emit({"type": "error", "message": str(e)})
        finally:
            self._set("stopping", "Menyimpan laporan…")
            for s in srcs.values():
                s.close()
            if session is not None:
                try:
                    session.finalize()
                    self.emit({"type": "report", "path": os.path.basename(self.out_dir)})
                except Exception as e:   # noqa: BLE001
                    traceback.print_exc()
                    self.emit({"type": "error", "message": f"Gagal menyimpan laporan: {e}"})
            with self.cond:
                self.cond.notify_all()
            self._set("idle")

    # ---------- MJPEG ----------
    def mjpeg(self, cam):
        last = 0
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.jpeg.get(cam, (None, 0))[1] != last, timeout=3)
                fr = self.jpeg.get(cam)
            if fr is None or fr[1] == last:
                if not self.running:
                    return
                continue
            last = fr[1]
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + fr[0] + b"\r\n"
