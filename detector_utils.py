"""AI part 3 - Part detector: YOLOv8n (fine-tuned on open PCB photos), run
with ONNX Runtime on CPU. Same pre/post-processing as the Android app."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).parent
MODEL = ROOT / "models" / "parts.onnx"
META = ROOT / "models" / "parts.json"

# Fixed categorical colours, assigned by class name (alphabetical), never by rank
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


class PartDetector:
    def __init__(self, model=MODEL, meta=META):
        import onnxruntime as ort
        m = json.loads(Path(meta).read_text())
        self.names = m["names"]
        self.size = int(m.get("imgsz", 640))
        self.meta = m
        self.session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    @staticmethod
    def available():
        return MODEL.exists() and META.exists()

    def letterbox(self, img):
        w, h = img.size
        s = min(self.size / w, self.size / h)
        nw, nh = round(w * s), round(h * s)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = Image.new("RGB", (self.size, self.size), (114, 114, 114))
        canvas.paste(img.convert("RGB").resize((nw, nh), Image.BILINEAR), (px, py))
        x = np.asarray(canvas, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        return x, s, px, py

    def detect(self, img, score=0.30, iou=0.45):
        x, s, px, py = self.letterbox(img)
        out = self.session.run(None, {self.input: x})[0][0]      # (4 + nc, N)
        dets = decode(out, s, px, py, img.size[0], img.size[1], score, iou)
        for det in dets:
            det["name"] = self.names[det["class_id"]]
        return dets

    def draw(self, img, dets):
        out = img.convert("RGB").copy()
        d = ImageDraw.Draw(out)
        names = sorted({x["name"] for x in dets})
        lw = max(2, round(max(out.size) / 400))
        for x in dets:
            d.rectangle(x["box"], outline=COLORS[names.index(x["name"]) % len(COLORS)], width=lw)
        return out, {n: COLORS[i % len(COLORS)] for i, n in enumerate(names)}


def decode(out, s, px, py, w, h, score=0.30, iou=0.45):
    """YOLOv8 raw output (4 + nc, N) -> boxes in original pixels, per-class NMS.
    Mirrors YoloDecoder.java (checked by a shared parity fixture)."""
    boxes, scores = out[:4].T, out[4:].T
    cls = scores.argmax(1)
    conf = scores.max(1)
    keep = conf > score
    boxes, cls, conf = boxes[keep], cls[keep], conf[keep]
    cx, cy, bw, bh = boxes.T if len(conf) else (np.zeros(0),) * 4
    x1 = np.clip((cx - bw / 2 - px) / s, 0, w); y1 = np.clip((cy - bh / 2 - py) / s, 0, h)
    x2 = np.clip((cx + bw / 2 - px) / s, 0, w); y2 = np.clip((cy + bh / 2 - py) / s, 0, h)
    ok = ((x2 - x1) >= 1) & ((y2 - y1) >= 1)
    x1, y1, x2, y2, cls, conf = x1[ok], y1[ok], x2[ok], y2[ok], cls[ok], conf[ok]
    area = (x2 - x1) * (y2 - y1)
    order = np.argsort(-conf, kind="stable")
    suppressed = np.zeros(len(order), bool)
    dets = []
    for a, i in enumerate(order):
        if suppressed[a]:
            continue
        dets.append({"class_id": int(cls[i]), "score": float(conf[i]),
                     "box": (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]))})
        for b in range(a + 1, len(order)):
            j = order[b]
            if suppressed[b] or cls[j] != cls[i]:
                continue
            inter = (max(0, min(x2[i], x2[j]) - max(x1[i], x1[j])) *
                     max(0, min(y2[i], y2[j]) - max(y1[i], y1[j])))
            union = area[i] + area[j] - inter
            if union > 0 and inter / union > iou:
                suppressed[b] = True
    return dets
