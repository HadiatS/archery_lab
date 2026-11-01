"""Archery Form Lab — GUI web.   python web.py   ->  http://localhost:8080  (atau http://IP-PC:8080 dari HP/PC lain)"""
import argparse
import base64
import copy
import json
import logging
import os
import queue
import time

import cv2
import yaml
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from app.engine import Engine
from app.report import compute_stats
from app.source import Source, grab_one, list_usb, to_target

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
engine = Engine()
CFG_PATH = os.path.join(BASE, "config.yaml")
ENGINE_CLS = Engine


def deep_merge(a, b):
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            deep_merge(a[k], v)
        else:
            a[k] = v
    return a


def out_dir():
    return os.path.join(BASE, load_base().get("output_dir", "data"))


def settings_path():
    return os.path.join(out_dir(), "settings.json")


def load_base():
    return yaml.safe_load(open(CFG_PATH, encoding="utf-8"))


def load_settings():
    try:
        return json.load(open(settings_path(), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_cfg():
    cfg = deep_merge(load_base(), load_settings())
    cfg["output_dir"] = out_dir()
    return cfg


def save_settings(patch):
    s = deep_merge(load_settings(), patch)
    os.makedirs(out_dir(), exist_ok=True)
    json.dump(s, open(settings_path(), "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    return s


def reference_file():
    return os.path.join(out_dir(), "reference.json")


def load_reference():
    try:
        return json.load(open(reference_file(), encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.route("/")
def index():
    return send_from_directory(os.path.join(BASE, "static"), "index.html")


@app.route("/api/state")
def state():
    cfg = load_cfg()
    ses = engine.session
    t = cfg["target"]
    return jsonify({
        "engine": {"state": engine.state, "message": engine.message, "active": engine.active if engine.running else []},
        "cameras": cfg["cameras"], "shooter": cfg["shooter"], "reference": cfg["reference"],
        "has_reference": bool(load_reference()),
        "target": {"center": t.get("center"), "radius": t.get("radius"), "rings": t.get("rings", 10),
                   "scoring": t.get("scoring", "rings"), "source": t.get("source", "camera"), "calibrated": bool(t.get("center") and t.get("radius"))},
        "session": ses.status() if ses else None, "offline": engine.offline, "progress": engine.progress,
        "shots": ses.public_shots() if ses else [],
        "session_id": os.path.basename(engine.out_dir) if engine.out_dir else None,
    })


@app.route("/api/settings", methods=["POST"])
def settings():
    patch = request.get_json(force=True) or {}
    allowed = {k: v for k, v in patch.items() if k in ("cameras", "shooter", "target", "reference")}
    s = save_settings(allowed)
    if engine.running and engine.session:
        engine.session.update_settings(load_cfg())
    return jsonify(ok=True, settings=s)


@app.route("/api/usb")
def usb():
    return jsonify(cameras=list_usb())


@app.route("/api/probe", methods=["POST"])
def probe():
    d = request.get_json(force=True) or {}
    cam, mode, value = d.get("cam"), d.get("mode"), d.get("value")
    frame = engine.raw.get(cam) if engine.running and cam in engine.active else None
    if frame is None:
        try:
            spec = to_target(mode, value)
            if spec is None:
                return jsonify(ok=False, error="Kamera ini diset 'Tidak dipakai'.")
            frame = grab_one(spec[0], spec[1], load_cfg().get("usb"))
        except Exception as e:   # noqa: BLE001
            return jsonify(ok=False, error=str(e))
    if frame is None:
        return jsonify(ok=False, error="Tidak ada gambar dari kamera. Periksa alamat/koneksi.")
    h, w = frame.shape[:2]
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jsonify(ok=True, w=w, h=h, image="data:image/jpeg;base64," + base64.b64encode(buf).decode())


@app.route("/api/calibrate", methods=["POST"])
def calibrate():
    d = request.get_json(force=True) or {}
    patch = {"target": {"center": d.get("center"), "radius": d.get("radius"),
                        "scoring": d.get("scoring", "rings"), "rings": int(d.get("rings", 10))}}
    save_settings(patch)
    if engine.running and engine.session and engine.session.target is not None:
        tg = engine.session.target
        tg.set_calibration(d.get("center"), d.get("radius"))
        tg.set_scoring(patch["target"]["scoring"], patch["target"]["rings"])
        tg.reset()
    return jsonify(ok=True)


@app.route("/api/score", methods=["POST"])
def score():
    """Skor anak panah dari luar kamera: tombol/klik di GUI, atau sensor / e-target / skrip lewat HTTP.
    Body JSON: {"score": 9}  atau  {"x": 0.12, "y": -0.3}  (+ opsional "shot": nomor, "source": "nama")."""
    d = request.get_json(force=True, silent=True) or {}
    ses = engine.session
    if not (engine.running and ses):
        return jsonify(ok=False, error="Sesi belum berjalan."), 409
    try:
        no = ses.add_score(d.get("score"), d.get("x"), d.get("y"), d.get("shot"), d.get("source", "eksternal"))
    except (ValueError, TypeError) as e:
        return jsonify(ok=False, error=str(e)), 409
    return jsonify(ok=True, shot=no)


@app.route("/api/upload/<cam>", methods=["POST"])
def upload(cam):
    """Unggah rekaman video dari browser (disimpan di data/uploads)."""
    from werkzeug.utils import secure_filename
    if cam not in ("depan", "samping", "target"):
        abort(404)
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify(ok=False, error="Tidak ada file."), 400
    d = os.path.join(out_dir(), "uploads")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"{cam}_{int(time.time())}_{secure_filename(f.filename) or 'video.mp4'}")
    f.save(path)
    return jsonify(ok=True, path=path, name=f.filename, size=os.path.getsize(path))


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """Analisa rekaman video: pose, skor panah, atau keduanya. Berjalan secepat mungkin dengan progres."""
    if engine.running:
        return jsonify(ok=False, error="Sesi lain sedang berjalan. Hentikan dulu."), 409
    d = request.get_json(force=True) or {}
    cfg = load_cfg()
    cams = d.get("cams") or {}
    mode = d.get("mode", "both")
    for n, pth in cams.items():
        if pth and not os.path.isfile(pth):
            return jsonify(ok=False, error=f"File {n} tidak ditemukan: {pth}"), 400
    cfg["cameras"] = {n: ({"mode": "file", "value": cams[n]} if cams.get(n) else {"mode": "off", "value": ""})
                      for n in ("depan", "samping", "target")}
    cfg["analysis_mode"] = mode if mode in ("both", "pose", "score") else "both"
    cfg["offsets"] = {k: float(v or 0) for k, v in (d.get("offsets") or {}).items()}
    cfg["realtime_files"] = bool(d.get("realtime", False))
    cfg["frame_skip"] = max(1, min(int(d.get("skip", 2) or 2), 6))
    cfg.setdefault("target", {})["match"] = d.get("match", "order")
    cfg["target"]["source"] = "camera"
    cfg["_offline"] = True
    engine.start(cfg, load_reference())
    return jsonify(ok=True)


@app.route("/api/session/start", methods=["POST"])
def start():
    if engine.running:
        return jsonify(ok=False, error="Sesi sudah berjalan.")
    engine.start(load_cfg(), load_reference())
    return jsonify(ok=True)


@app.route("/api/session/stop", methods=["POST"])
def stop():
    if engine.running:
        engine.stop()
    return jsonify(ok=True)


@app.route("/api/action/<name>", methods=["POST"])
def action(name):
    ses = engine.session
    if not (engine.running and ses):
        return jsonify(ok=False, error="Sesi belum berjalan.")
    if name == "reference":
        if not ses.set_reference_from_last():
            return jsonify(ok=False, error="Belum ada tembakan yang bisa dijadikan acuan.")
        os.makedirs(out_dir(), exist_ok=True)
        json.dump(ses.reference, open(reference_file(), "w", encoding="utf-8"), indent=2)
        engine.emit({"type": "info", "message": "Tembakan terakhir dijadikan acuan bentuk."})
    elif name == "newround":
        ses.new_round()
        engine.emit({"type": "info", "message": "Ronde baru: target di-reset."})
    elif name == "capture":
        for n, fr in list(engine.raw.items()):
            cv2.imwrite(os.path.join(ses.out, f"capture_{n}_{time.strftime('%H%M%S')}.jpg"), fr)
        engine.emit({"type": "info", "message": "Capture tersimpan."})
    else:
        abort(404)
    return jsonify(ok=True)


@app.route("/api/action/clear_reference", methods=["POST"])
def clear_reference():
    try:
        os.remove(reference_file())
    except OSError:
        pass
    if engine.session:
        engine.session.reference = {}
    return jsonify(ok=True)


@app.route("/video/<cam>")
def video(cam):
    if cam not in ("depan", "samping", "target"):
        abort(404)
    return Response(engine.mjpeg(cam), mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-store"})


@app.route("/shots/<path:fn>")
def shot_img(fn):
    if not engine.out_dir:
        abort(404)
    return send_from_directory(os.path.join(engine.out_dir, "shots"), fn)


@app.route("/api/sessions")
def sessions():
    root, items = out_dir(), []
    if os.path.isdir(root):
        for d in sorted(os.listdir(root), reverse=True):
            p = os.path.join(root, d, "summary.json")
            if os.path.isfile(p):
                try:
                    st = json.load(open(p, encoding="utf-8")).get("stats", {})
                except ValueError:
                    continue
                items.append({"id": d, "n": st.get("n"), "form_avg": st.get("form_avg"),
                              "arrow_total": st.get("arrow_total"), "arrow_n": st.get("arrow_n")})
    return jsonify(items)


@app.route("/sessions/<sid>/<path:fn>")
def session_files(sid, fn):
    return send_from_directory(os.path.join(out_dir(), sid), fn)


@app.route("/api/stream")
def stream():
    q = engine.subscribe()

    def gen():
        try:
            while True:
                try:
                    yield f"data: {q.get(timeout=15)}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            engine.unsubscribe(q)
    return Response(gen(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args()
    logging.getLogger("werkzeug").setLevel(logging.WARNING)
    print(f"\n  Archery Form Lab: http://localhost:{a.port}\n")
    app.run(host=a.host, port=a.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
