import cv2
import numpy as np


def _letterbox(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    nh, nw = max(int(round(h * s)), 1), max(int(round(w * s)), 1)
    canvas = np.zeros((size, size, 3), np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    return canvas, s, px, py


def _load_tflite(path):
    for mod in ("tflite_runtime.interpreter", "ai_edge_litert.interpreter"):
        try:
            return __import__(mod, fromlist=["Interpreter"]).Interpreter(model_path=path)
        except ImportError:
            pass
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=path)


class MoveNet:
    """MoveNet SinglePose (17 titik COCO). Dijalankan pada crop tiap orang (top-down) dari YOLO."""

    def __init__(self, cfg):
        variant = cfg.get("variant", "lightning")
        self.size = 192 if variant == "lightning" else 256
        if cfg.get("model_path"):
            self.mode = "tflite"
            self.it = _load_tflite(cfg["model_path"])
            self.it.allocate_tensors()
            self.inp = self.it.get_input_details()[0]
            self.out = self.it.get_output_details()[0]
            self.size = int(self.inp["shape"][1])
        else:
            self.mode = "hub"
            import tensorflow as tf
            import tensorflow_hub as hub
            self.tf = tf
            self.fn = hub.load(f"https://tfhub.dev/google/movenet/singlepose/{variant}/4").signatures["serving_default"]

    def _run(self, img):
        if self.mode == "tflite":
            self.it.set_tensor(self.inp["index"], img[None].astype(self.inp["dtype"]))
            self.it.invoke()
            return self.it.get_tensor(self.out["index"])
        out = self.fn(self.tf.constant(img[None], dtype=self.tf.int32))
        return out["output_0"].numpy()

    def predict(self, crop_bgr, ox, oy):
        """Return array (17,3): x,y (koordinat frame), skor."""
        canvas, s, px, py = _letterbox(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB), self.size)
        k = np.asarray(self._run(canvas)).reshape(17, 3)
        out = np.zeros((17, 3), np.float32)
        out[:, 0] = (k[:, 1] * self.size - px) / s + ox
        out[:, 1] = (k[:, 0] * self.size - py) / s + oy
        out[:, 2] = k[:, 2]
        return out


class YoloPose:
    """Cadangan: YOLO-pose (urutan 17 titik sama dengan MoveNet/COCO)."""

    def __init__(self, cfg):
        from ultralytics import YOLO
        self.m = YOLO(cfg.get("weights", "yolo11n-pose.pt"))

    def predict(self, crop_bgr, ox, oy):
        r = self.m.predict(crop_bgr, verbose=False, conf=0.25)[0]
        if r.keypoints is None or len(r.boxes) == 0 or r.keypoints.conf is None:
            return None
        i = int(r.boxes.conf.cpu().numpy().argmax())
        xy, cf = r.keypoints.xy[i].cpu().numpy(), r.keypoints.conf[i].cpu().numpy()
        out = np.zeros((17, 3), np.float32)
        out[:, 0], out[:, 1], out[:, 2] = xy[:, 0] + ox, xy[:, 1] + oy, cf
        return out


def build_pose(cfg):
    """MoveNet (default). Jika TensorFlow tidak bisa dimuat, otomatis beralih ke YOLO-pose (17 titik yang sama)."""
    if cfg.get("backend", "movenet") == "yolo_pose":
        return YoloPose(cfg)
    try:
        return MoveNet(cfg)
    except Exception as e:   # noqa: BLE001  (ImportError protobuf/TensorFlow, dsb.)
        msg = f"MoveNet/TensorFlow gagal dimuat ({type(e).__name__}: {str(e)[:120]}). Memakai YOLO-pose sebagai gantinya."
        print("[pose] " + msg)
        p = YoloPose(cfg)
        p.fallback = msg
        return p
