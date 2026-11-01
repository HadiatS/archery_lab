import math

import cv2
import numpy as np


class TargetScorer:
    """Deteksi anak panah baru di papan sasaran (selisih gambar saat adegan stabil) + skor."""

    def __init__(self, cfg):
        c = cfg
        self.thresh, self.motion_frac = c.get("diff_threshold", 28), c.get("motion_frac", 0.003)
        self.min_area, self.max_area = c.get("min_area_frac", 0.0004), c.get("max_area_frac", 0.25)
        self.stable_sec = c.get("stable_sec", 0.5)
        self.set_scoring(c.get("scoring", "rings"), c.get("rings", 10), c.get("zones"))
        self.set_calibration(c.get("center"), c.get("radius"))
        self.reset()

    def set_scoring(self, scoring, rings, zones=None):
        self.scoring, self.rings, self.zones = scoring, int(rings), zones

    def set_calibration(self, center, radius):
        self.center = (float(center[0]), float(center[1])) if center else None
        self.radius = float(radius) if radius else None
        self._mask = None

    @property
    def ready(self):
        return self.center is not None and bool(self.radius)

    def reset(self):
        self.ref = self.prev = self.last_t = None
        self.stable, self.dirty, self.arrows = 0.0, False, []

    def score_of(self, d_rel):
        if self.scoring == "hit_miss":
            return 1 if d_rel <= 1 else 0
        if self.zones:
            for rf, sc in sorted(self.zones):
                if d_rel <= rf:
                    return sc
            return 0
        return max(self.rings - int(d_rel * self.rings), 0) if d_rel <= 1 else 0

    def _get_mask(self, H, W):
        key = (H, W, self.center, self.radius)
        if self._mask is None or self._mask[0] != key:
            m = np.zeros((H, W), np.uint8)
            cv2.circle(m, (int(self.center[0]), int(self.center[1])), int(self.radius * 1.15), 255, -1)
            self._mask = (key, m > 0, int((m > 0).sum()))
        return self._mask[1], self._mask[2]

    def process(self, frame, t, pending):
        if not self.ready:
            return []
        H, W = frame.shape[:2]
        mask, cnt = self._get_mask(H, W)
        g = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)
        dt = 0.0 if self.last_t is None else min(max(t - self.last_t, 0.0), 1.0)
        self.last_t = t
        frac = 1.0 if self.prev is None else float(np.count_nonzero((cv2.absdiff(g, self.prev) > self.thresh) & mask)) / max(cnt, 1)
        self.prev = g
        if frac > self.motion_frac:
            self.stable, self.dirty = 0.0, True
            return []
        self.stable += dt
        if self.stable < self.stable_sec:
            return []
        if self.ref is None:
            self.ref, self.dirty = g.copy(), False
            return []
        if not self.dirty:
            return []
        self.dirty = False
        m = ((cv2.absdiff(g, self.ref) > self.thresh) & mask).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area_ref = math.pi * self.radius ** 2
        blobs = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a >= self.min_area * area_ref:
                mo = cv2.moments(c)
                blobs.append((a, mo["m10"] / mo["m00"], mo["m01"] / mo["m00"]))
        out = []
        if sum(b[0] for b in blobs) > self.max_area * area_ref:   # pencahayaan berubah / target diganti
            self.ref = g.copy()
            return [{"type": "reset"}]
        near = max(8.0, 0.04 * self.radius)
        removed = [a for a in self.arrows if any(math.hypot(b[1] - a["x"], b[2] - a["y"]) < near for b in blobs)]
        for a in removed:
            self.arrows.remove(a)
        if removed:
            out.append({"type": "removed", "n": len(removed)})
        new = [b for b in blobs if not any(math.hypot(b[1] - a["x"], b[2] - a["y"]) < near for a in removed)]
        for a_, x, y in sorted(new, key=lambda b: -b[0])[:max(pending, 0)]:
            dx, dy = (x - self.center[0]) / self.radius, (y - self.center[1]) / self.radius
            d = math.hypot(dx, dy)
            arrow = {"x": x, "y": y, "dx_rel": dx, "dy_rel": dy, "d_rel": d, "score": self.score_of(d), "shot_no": None}
            self.arrows.append(arrow)
            out.append({"type": "arrow", **arrow, "_ref": arrow})
        self.ref = g.copy()
        return out

    def draw(self, vis):
        if not self.ready:
            return vis
        c = (int(self.center[0]), int(self.center[1]))
        n = self.rings if self.scoring == "rings" else 1
        for k in range(1, n + 1):
            cv2.circle(vis, c, int(self.radius * k / n), (0, 200, 255) if k == n else (0, 140, 200), 1)
        cv2.drawMarker(vis, c, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
        for a in self.arrows:
            p = (int(a["x"]), int(a["y"]))
            cv2.circle(vis, p, 9, (0, 0, 0), 4)
            cv2.circle(vis, p, 9, (0, 60, 255), 2)
            cv2.putText(vis, f"{a['shot_no'] or ''}:{a['score']}", (p[0] + 11, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(vis, f"{a['shot_no'] or ''}:{a['score']}", (p[0] + 11, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return vis
