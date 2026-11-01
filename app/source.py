import os
import threading
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
import cv2  # noqa: E402


def _usb_cap(index):
    return cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)


def list_usb(max_index=6):
    found = []
    for i in range(max_index):
        cap = _usb_cap(i)
        if cap.isOpened():
            found.append(i)
        cap.release()
    return found


def normalize(mode, url):
    url = (url or "").strip()
    if mode == "rtsp" and not url.lower().startswith("rtsp://"):
        url = "rtsp://" + url
    if mode == "wifi":
        if not url.lower().startswith(("http://", "https://", "rtsp://")):
            url = "http://" + url
        if "/" not in url.split("://", 1)[1]:      # hanya IP:port -> path MJPEG umum
            url += "/video"
    return url


def to_target(mode, value):
    """(mode, nilai) -> (conn, target) atau None bila 'Tidak dipakai'."""
    mode, value = (mode or "off").lower(), str(value or "").strip()
    if mode in ("off", "none", ""):
        return None
    if mode == "usb":
        return "usb", int(value or 0)
    if mode in ("wifi", "rtsp"):
        return mode, normalize(mode, value)
    if mode == "file":
        return "file", value
    raise ValueError(f"mode kamera tidak dikenal: {mode}")


class Source:
    """Live: thread terpisah (selalu frame terbaru, auto-reconnect). File: berurutan.
    read() -> (ok, frame, t). t = jam dinding (live) atau waktu video (file)."""

    def __init__(self, conn, target, usb_cfg=None):
        self.conn, self.target, self.usb_cfg = conn, target, usb_cfg or {}
        self.live = conn != "file"
        self.frame, self.stopped, self.ended, self.n = None, False, False, 0
        self.lock = threading.Lock()
        self.cap = self._open()
        if not self.live and not self.cap.isOpened():
            raise RuntimeError(f"Tidak bisa membuka file: {target}")
        self.duration = 0.0
        if not self.live:
            fps, n = self.cap.get(cv2.CAP_PROP_FPS) or 25.0, self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
            self.duration = float(n / fps) if n and n > 0 else 0.0
        if self.live:
            threading.Thread(target=self._loop, daemon=True).start()

    def _open(self):
        if self.conn == "usb":
            cap = _usb_cap(self.target)
            if self.usb_cfg.get("width"):
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.usb_cfg["width"])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.usb_cfg.get("height", 720))
            return cap
        return cv2.VideoCapture(self.target)

    def _loop(self):
        while not self.stopped:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(2)
                self.cap.release()
                self.cap = self._open()
                continue
            with self.lock:
                self.frame = f

    def read(self):
        if self.live:
            with self.lock:
                f, self.frame = self.frame, None
            return f is not None, f, time.time()
        ok, f = self.cap.read()
        if not ok:
            self.ended = True
            return False, None, 0.0
        self.n += 1
        t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        if t <= 0:
            t = self.n / (self.cap.get(cv2.CAP_PROP_FPS) or 25.0)
        return True, f, t

    def close(self):
        self.stopped = True
        time.sleep(0.1)
        self.cap.release()


def grab_one(conn, target, usb_cfg=None, timeout=8.0):
    """Ambil satu frame untuk uji koneksi / kalibrasi."""
    s = Source(conn, target, usb_cfg)
    try:
        end = time.time() + timeout
        while time.time() < end:
            ok, f, _ = s.read()
            if ok:
                return f
            if s.ended:
                break
            time.sleep(0.05)
    finally:
        s.close()
    return None
