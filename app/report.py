import base64
import csv
import html
import json
import math
import os

import cv2
import numpy as np

from .metrics import LABEL


def _mean(x):
    return float(sum(x) / len(x)) if x else None


def compute_stats(shots, ref_cfg):
    """shots: daftar shot publik (dict). Return statistik ringkas."""
    forms = [s["form"] for s in shots if s.get("form") is not None]
    arrows = [s["arrow"] for s in shots if s.get("arrow")]
    st = {"n": len(shots), "form_avg": _mean(forms), "form_best": max(forms) if forms else None,
          "arrow_n": len(arrows), "arrow_total": sum(a["score"] for a in arrows) if arrows else 0,
          "arrow_avg": _mean([a["score"] for a in arrows]), "group": None, "bias": None,
          "consistency": [], "correlation": []}
    geo = [a for a in arrows if a.get("dx_rel") is not None]
    if len(geo) >= 2:
        cx, cy = _mean([a["dx_rel"] for a in geo]), _mean([a["dy_rel"] for a in geo])
        r = _mean([math.hypot(a["dx_rel"] - cx, a["dy_rel"] - cy) for a in geo])
        st["group"] = {"cx": cx, "cy": cy, "r": r}
        parts = []
        if abs(cx) > 0.05:
            parts.append("kanan" if cx > 0 else "kiri")
        if abs(cy) > 0.05:
            parts.append("bawah" if cy > 0 else "atas")
        st["bias"] = " & ".join(parts) or "sekitar pusat"
    # konsistensi: simpangan baku tiap metrik relatif terhadap toleransi
    vals = {}
    for s in shots:
        for view, d in (s.get("metrics") or {}).items():
            for k, v in d.items():
                vals.setdefault((view, k), []).append(v)
    for (view, k), v in vals.items():
        spec = (ref_cfg.get(view) or {}).get(k)
        if spec and len(v) >= 3 and spec[1]:
            sd = float(np.std(v))
            st["consistency"].append({"view": view, "name": k, "label": LABEL.get(k, k), "std": sd,
                                      "ratio": sd / float(spec[1])})
    st["consistency"] = sorted(st["consistency"], key=lambda x: -x["ratio"])[:4]
    # korelasi bentuk vs jarak dari pusat (butuh >= 8 tembakan)
    pairs = [(s["metrics"], s["arrow"]["d_rel"]) for s in shots if s.get("arrow") and s["arrow"].get("d_rel") is not None and s.get("metrics")]
    if len(pairs) >= 8:
        for (view, k) in vals:
            xs = [(m.get(view) or {}).get(k) for m, _ in pairs]
            ys = [d for _, d in pairs]
            ok = [(x, y) for x, y in zip(xs, ys) if x is not None]
            if len(ok) >= 8 and np.std([o[0] for o in ok]) > 1e-9 and np.std([o[1] for o in ok]) > 1e-9:
                r = float(np.corrcoef([o[0] for o in ok], [o[1] for o in ok])[0, 1])
                if abs(r) >= 0.5:
                    st["correlation"].append({"label": LABEL.get(k, k), "r": r})
        st["correlation"] = sorted(st["correlation"], key=lambda x: -abs(x["r"]))[:2]
    return st


def _b64(path, w=1000):
    img = cv2.imread(path)
    if img is None:
        return ""
    if img.shape[1] > w:
        img = cv2.resize(img, (w, int(img.shape[0] * w / img.shape[1])))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def write_reports(out, shots, ref_cfg, target_img=None, meta=None):
    meta = meta or {}
    st = compute_stats(shots, ref_cfg)
    # CSV
    metric_cols = sorted({(v, k) for s in shots for v, d in (s.get("metrics") or {}).items() for k in d})
    with open(os.path.join(out, "shots.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["no", "waktu_lepas_s", "durasi_tarik_s", "durasi_tahan_s", "form_score", "skor_panah",
                    "jarak_pusat_rel", "dx_rel", "dy_rel", "status_panah", "catatan"] + [f"{v}.{k}" for v, k in metric_cols])
        for s in shots:
            a = s.get("arrow") or {}
            r = lambda x, n=2: "" if x is None else round(x, n)   # noqa: E731
            w.writerow([s["no"], r(s.get("t_release")), r(s.get("draw_time")), r(s.get("aim_time")), r(s.get("form"), 1),
                        a.get("score", ""), r(a.get("d_rel"), 3), r(a.get("dx_rel"), 3), r(a.get("dy_rel"), 3),
                        s.get("arrow_status", ""), " | ".join(s.get("notes") or [])] +
                       [r(((s.get("metrics") or {}).get(v) or {}).get(k), 3) for v, k in metric_cols])
    data = {"meta": meta, "stats": st, "shots": [{k: v for k, v in s.items() if k not in ("details",)} for s in shots]}
    with open(os.path.join(out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=float)
    # TXT
    fmt = lambda x, n=1: "-" if x is None else f"{x:.{n}f}"   # noqa: E731
    txt = [f"RINGKASAN SESI ({out})", f"Jumlah tembakan: {st['n']}",
           f"Skor bentuk rata-rata: {fmt(st['form_avg'])}/100 (terbaik {fmt(st['form_best'])})"]
    if st["arrow_n"]:
        txt.append(f"Skor panah: total {st['arrow_total']} | rata-rata {fmt(st['arrow_avg'])} ({st['arrow_n']} panah terdeteksi)")
    if st["group"]:
        txt.append(f"Ukuran grup: {st['group']['r']:.2f} x radius target | kecenderungan: {st['bias']}")
    for c in [c for c in st["consistency"] if c["ratio"] > 0.5]:
        txt.append(f"Kurang konsisten: {c['label']} (simpangan {c['ratio']:.1f}x toleransi)")
    for c in st["correlation"]:
        txt.append(f"Berkaitan dengan jarak dari pusat: {c['label']} (r={c['r']:+.2f}; korelasi bukan bukti sebab-akibat)")
    open(os.path.join(out, "summary.txt"), "w", encoding="utf-8").write("\n".join(txt))
    if target_img is not None:
        cv2.imwrite(os.path.join(out, "target_final.jpg"), target_img)
    # HTML
    rows = []
    for s in shots:
        a = s.get("arrow")
        arrow_txt = f"{a['score']}" if a else {"off": "-", "tidak_terdeteksi": "tidak terdeteksi"}.get(s.get("arrow_status"), "-")
        img = _b64(os.path.join(out, "shots", s["image"])) if s.get("image") else ""
        rows.append(f"<div class='s'><div class='h'><b>#{s['no']}</b> Form <b>{fmt(s.get('form'), 0)}</b>/100 · Panah <b>{html.escape(str(arrow_txt))}</b></div>"
                    f"<img src='{img}'><ul>{''.join('<li>' + html.escape(n) + '</li>' for n in (s.get('notes') or []))}</ul></div>")
    page = f"""<!doctype html><meta charset=utf-8><title>Laporan Panahan</title>
<style>body{{font-family:Segoe UI,system-ui,sans-serif;background:#0b1020;color:#e8eefc;max-width:1100px;margin:auto;padding:24px}}
h1{{letter-spacing:2px}}.k{{display:flex;gap:12px;flex-wrap:wrap}}.k div{{background:#141b34;border-radius:14px;padding:14px 18px;min-width:150px}}
.k b{{font-size:26px;display:block;color:#5eead4}}.s{{background:#141b34;border-radius:14px;padding:14px;margin:14px 0}}
.s img{{width:100%;border-radius:10px;margin-top:8px}}li{{color:#fbbf24}}</style>
<h1>🏹 LAPORAN SESI PANAHAN</h1><div class=k>
<div>Tembakan<b>{st['n']}</b></div><div>Form rata-rata<b>{fmt(st['form_avg'], 0)}</b></div>
<div>Skor panah<b>{st['arrow_total']}</b></div><div>Ukuran grup<b>{fmt(st['group']['r'], 2) if st['group'] else '-'}</b></div>
<div>Kecenderungan<b style='font-size:18px'>{html.escape(st['bias'] or '-')}</b></div></div>
<p>{'<br>'.join(html.escape(t) for t in txt[3:])}</p>{''.join(rows)}"""
    open(os.path.join(out, "report.html"), "w", encoding="utf-8").write(page)
    return st
