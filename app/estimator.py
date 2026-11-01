import numpy as np

from .pose import build_pose


def _iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def _area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


class Models:
    """YOLO (deteksi orang) + MoveNet (pose) — dimuat sekali, dipakai semua kamera."""

    def __init__(self, cfg):
        from ultralytics import YOLO
        d = cfg["detector"]
        self.yolo = YOLO(d["weights"])
        self.conf, self.imgsz, self.device = d.get("conf", 0.4), d.get("imgsz", 960), d.get("device")
        self.pid = [i for i, n in self.yolo.names.items() if n == "person"]
        self.pose = build_pose(cfg.get("pose", {}))

    def persons(self, frame):
        r = self.yolo.predict(frame, conf=self.conf, classes=self.pid, imgsz=self.imgsz,
                              device=self.device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            return []
        return [tuple(float(v) for v in b) for b in r.boxes.xyxy.cpu().numpy()]


class CamPose:
    """Pose satu pemanah untuk satu kamera: YOLO tiap N frame, MoveNet tiap frame, dengan penghalusan."""

    def __init__(self, models, roi=None, det_every=5, smooth=0.5):
        self.m, self.roi, self.det_every = models, roi, max(int(det_every), 1)
        self.alpha = 1.0 - float(smooth)
        self.box, self.kp, self.n, self.miss = None, None, 0, 0

    def _in_roi(self, b, W, H):
        if not self.roi:
            return True
        cx, cy = (b[0] + b[2]) / 2 / W, (b[1] + b[3]) / 2 / H
        return self.roi[0] <= cx <= self.roi[2] and self.roi[1] <= cy <= self.roi[3]

    def update(self, frame):
        H, W = frame.shape[:2]
        self.n += 1
        if self.box is None or self.n % self.det_every == 1:
            c = [b for b in self.m.persons(frame) if self._in_roi(b, W, H)]
            if c:
                best = max(c, key=_area)
                if self.box is not None:
                    b2 = max(c, key=lambda b: _iou(b, self.box))
                    best = b2 if _iou(b2, self.box) > 0.2 else best
                self.box, self.miss = best, 0
            else:
                self.miss += 1
                if self.miss > 10:
                    self.box = self.kp = None
        if self.box is None:
            return None
        x1, y1, x2, y2 = self.box
        w, h = x2 - x1, y2 - y1
        cx1, cy1 = int(max(x1 - 0.2 * w, 0)), int(max(y1 - 0.2 * h, 0))
        cx2, cy2 = int(min(x2 + 0.2 * w, W)), int(min(y2 + 0.2 * h, H))
        if cx2 - cx1 < 32 or cy2 - cy1 < 32:
            return self.kp
        kp = self.m.pose.predict(frame[cy1:cy2, cx1:cx2], cx1, cy1)
        if kp is None:
            return self.kp
        if self.kp is not None and self.alpha < 1.0:
            ok = (kp[:, 2] >= 0.3) & (self.kp[:, 2] >= 0.3)
            kp[ok, :2] = self.alpha * kp[ok, :2] + (1 - self.alpha) * self.kp[ok, :2]
        good = kp[kp[:, 2] >= 0.3]
        if len(good) >= 5:      # ikuti gerak pemanah untuk crop berikutnya
            gx1, gy1, gx2, gy2 = good[:, 0].min(), good[:, 1].min(), good[:, 0].max(), good[:, 1].max()
            mx, my = 0.25 * (gx2 - gx1), 0.25 * (gy2 - gy1)
            self.box = (max(gx1 - mx, 0), max(gy1 - my, 0), min(gx2 + mx, W), min(gy2 + my, H))
        self.kp = kp
        return kp
