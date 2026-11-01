import math

import numpy as np

(NOSE, LEYE, REYE, LEAR, REAR, LSH, RSH, LEL, REL, LWR, RWR, LHIP, RHIP) = range(13)

LABEL = {
    "D": "Panjang tarikan (rasio)", "draw_elbow_angle": "Sudut siku tarikan (°)",
    "bow_elbow_angle": "Sudut siku lengan busur (°)", "elbow_line_offset": "Tinggi siku vs garis anak panah",
    "shoulder_arrow_angle": "Selisih garis bahu–anak panah (°)", "torso_lean": "Condong badan ke target (°)",
    "anchor_dist": "Jarak tangan ke anchor", "draw_elbow_height": "Tinggi siku tarikan vs bahu",
    "bow_wrist_height": "Tinggi lengan busur vs bahu", "head_tilt": "Kemiringan kepala (°)",
    "shoulder_tilt": "Kemiringan bahu (°)", "torso_side_lean": "Condong badan ke samping (°)",
    "follow_back": "Lanjutan tangan tarikan ke belakang", "bow_drop": "Turunnya lengan busur setelah lepas",
}
MSG = {
    ("elbow_line_offset", 1): "Siku tarikan terlalu tinggi dari garis anak panah",
    ("elbow_line_offset", -1): "Siku tarikan terlalu rendah dari garis anak panah",
    ("bow_elbow_angle", -1): "Lengan busur menekuk (kurang lurus)",
    ("torso_lean", 1): "Badan condong ke arah target", ("torso_lean", -1): "Badan condong ke belakang",
    ("shoulder_arrow_angle", 1): "Garis bahu tidak sejajar dengan anak panah",
    ("anchor_dist", 1): "Tangan tarikan menjauh dari titik anchor",
    ("draw_elbow_height", 1): "Siku tarikan terlalu tinggi", ("draw_elbow_height", -1): "Siku tarikan terlalu rendah",
    ("bow_wrist_height", 1): "Lengan busur terlalu tinggi", ("bow_wrist_height", -1): "Lengan busur terlalu rendah",
    ("head_tilt", 1): "Kepala miring", ("shoulder_tilt", 1): "Bahu miring",
    ("torso_side_lean", 1): "Badan condong ke samping",
    ("D", 1): "Tarikan lebih panjang dari biasanya", ("D", -1): "Tarikan lebih pendek dari biasanya",
    ("follow_back", -1): "Tangan tarikan tidak lanjut ke belakang setelah lepas",
    ("bow_drop", 1): "Lengan busur jatuh terlalu cepat setelah lepas",
}


def _P(kp, i):
    return kp[i, :2].astype(float)


def _ok(kp, idx, thr):
    return all(kp[i, 2] >= thr for i in idx)


def _d(a, b):
    return float(math.hypot(a[0] - b[0], a[1] - b[1]))


def angle3(a, b, c):
    v1, v2 = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    n = np.linalg.norm(v1) * np.linalg.norm(v2)
    return None if n < 1e-6 else float(math.degrees(math.acos(np.clip(np.dot(v1, v2) / n, -1, 1))))


def line_diff_deg(v1, v2):
    a = abs(math.degrees(math.atan2(v1[1], v1[0]) - math.atan2(v2[1], v2[0]))) % 180
    return min(a, 180 - a)


def tilt_deg(v):
    a = abs(math.degrees(math.atan2(v[1], v[0])))
    return min(a, 180 - a)


def arm_idx(hand):
    """((bahu, siku, pergelangan) tarikan, (…) lengan busur)."""
    return ((RSH, REL, RWR), (LSH, LEL, LWR)) if hand == "right" else ((LSH, LEL, LWR), (RSH, REL, RWR))


def body_scale(kp, thr):
    if not _ok(kp, (LSH, RSH), thr):
        return None
    sh = (_P(kp, LSH) + _P(kp, RSH)) / 2
    if _ok(kp, (LHIP, RHIP), thr):
        return _d(sh, (_P(kp, LHIP) + _P(kp, RHIP)) / 2)
    hips = [i for i in (LHIP, RHIP) if kp[i, 2] >= thr]
    return _d(sh, _P(kp, hips[0])) if hips else None


def anchor_point(kp, hand, anchor, thr):
    ear, dsh = (REAR, RSH) if hand == "right" else (LEAR, LSH)
    if anchor == "ear":
        return _P(kp, ear) if kp[ear, 2] >= thr else None
    if anchor == "cheek":
        return (_P(kp, NOSE) + _P(kp, ear)) / 2 if _ok(kp, (NOSE, ear), thr) else None
    if anchor in ("chin", "nose"):
        return _P(kp, NOSE) if kp[NOSE, 2] >= thr else None
    if anchor == "shoulder":
        return _P(kp, dsh) if kp[dsh, 2] >= thr else None
    if anchor == "chest" and _ok(kp, (LSH, RSH, LHIP, RHIP), thr):
        sh, hp = (_P(kp, LSH) + _P(kp, RSH)) / 2, (_P(kp, LHIP) + _P(kp, RHIP)) / 2
        return sh + 0.3 * (hp - sh)
    return None


def compute(kp, view, hand, anchor="cheek", thr=0.3):
    """Metrik bentuk dari 17 titik. view: 'perp' (kamera depan) | 'inline' (kamera samping)."""
    S = body_scale(kp, thr)
    if S is None or S < 10:
        return {}
    (dsh, de, dwr), (bsh, be, bwr) = arm_idx(hand)
    P = lambda i: _P(kp, i)   # noqa: E731
    m = {"_scale": S}
    if _ok(kp, (dwr, bwr), thr):
        m["D"] = _d(P(dwr), P(bwr)) / S
    if view == "perp":
        if _ok(kp, (dsh, de, dwr), thr):
            m["draw_elbow_angle"] = angle3(P(dsh), P(de), P(dwr))
        if _ok(kp, (bsh, be, bwr), thr):
            m["bow_elbow_angle"] = angle3(P(bsh), P(be), P(bwr))
        if _ok(kp, (dwr, bwr, de), thr):
            dx = P(bwr)[0] - P(dwr)[0]
            if abs(dx) > 0.3 * S:
                yl = P(dwr)[1] + (P(bwr)[1] - P(dwr)[1]) * (P(de)[0] - P(dwr)[0]) / dx
                m["elbow_line_offset"] = (yl - P(de)[1]) / S          # + = siku di atas garis anak panah
        if _ok(kp, (dsh, bsh, dwr, bwr), thr):
            m["shoulder_arrow_angle"] = line_diff_deg(P(bsh) - P(dsh), P(bwr) - P(dwr))
        if _ok(kp, (LSH, RSH, LHIP, RHIP, dwr, bwr), thr):
            sh, hp = (P(LSH) + P(RSH)) / 2, (P(LHIP) + P(RHIP)) / 2
            s = 1 if P(bwr)[0] > P(dwr)[0] else -1                    # arah ke target
            m["torso_lean"] = math.degrees(math.atan2((sh[0] - hp[0]) * s, hp[1] - sh[1]))
        ap = anchor_point(kp, hand, anchor, thr)
        if ap is not None and kp[dwr, 2] >= thr:
            m["anchor_dist"] = _d(P(dwr), ap) / S
    else:
        if _ok(kp, (dsh, de), thr):
            m["draw_elbow_height"] = (P(dsh)[1] - P(de)[1]) / S      # + = siku di atas bahu
        if _ok(kp, (bsh, bwr), thr):
            m["bow_wrist_height"] = (P(bsh)[1] - P(bwr)[1]) / S
        if _ok(kp, (LEAR, REAR), thr):
            m["head_tilt"] = tilt_deg(P(REAR) - P(LEAR))
        if _ok(kp, (LSH, RSH), thr) and _d(P(LSH), P(RSH)) > 0.2 * S:
            m["shoulder_tilt"] = tilt_deg(P(RSH) - P(LSH))
        if _ok(kp, (LSH, RSH, LHIP, RHIP), thr):
            v = (P(LSH) + P(RSH)) / 2 - (P(LHIP) + P(RHIP)) / 2
            m["torso_side_lean"] = abs(math.degrees(math.atan2(v[0], -v[1])))
    return {k: (float(v) if v is not None else None) for k, v in m.items() if v is not None}


class Baseline:
    """Rata-rata tembakan pribadi sebagai acuan jika target tidak ditentukan."""

    def __init__(self):
        self.v = {}

    def add(self, shot_metrics):
        for view, d in shot_metrics.items():
            for k, x in d.items():
                self.v.setdefault((view, k), []).append(x)

    def target(self, view, name):
        vals = self.v.get((view, name), [])
        return float(np.median(vals[-10:])) if len(vals) >= 3 else None


def resolve_target(spec, view, name, reference, baseline):
    r = (reference.get(view) or {}).get(name)
    if r is not None:
        return float(r)
    if spec[0] is not None:
        return float(spec[0])
    return baseline.target(view, name)


def evaluate(shot_metrics, ref_cfg, reference, baseline):
    """Return (skor_form 0-100 | None, detail[], catatan[])."""
    details, num, den = [], 0.0, 0.0
    for view, mets in shot_metrics.items():
        for name, v in mets.items():
            spec = (ref_cfg.get(view) or {}).get(name)
            if not spec or v is None:
                continue
            tol, w = float(spec[1]), (float(spec[2]) if len(spec) > 2 else 1.0)
            target = resolve_target(spec, view, name, reference, baseline)
            if target is None or tol <= 0:
                continue
            dev = v - target
            s = 100.0 if abs(dev) <= tol else max(0.0, 100.0 * (1 - (abs(dev) - tol) / (2 * tol)))
            details.append({"view": view, "name": name, "label": LABEL.get(name, name), "value": float(v),
                            "target": float(target), "tol": tol, "weight": w, "score": s, "dev": float(dev)})
            num += s * w
            den += w
    bad = sorted([d for d in details if abs(d["dev"]) > d["tol"]],
                 key=lambda d: -(abs(d["dev"]) - d["tol"]) / d["tol"] * d["weight"])
    notes = []
    for d in bad[:3]:
        txt = MSG.get((d["name"], 1 if d["dev"] > 0 else -1), d["label"])
        notes.append(f"{txt} ({d['value']:.2f} vs {d['target']:.2f})")
    return (num / den if den else None), details, notes
