"""Object weight estimation (PROTOTYPE): which object is the person handling, and roughly
how heavy is that kind of object?

1. ObjectDetector runs a small COCO detector (YOLOX-s, same family and library as the
   person detector) in a background thread, a few times a second, never in the video loop.
2. carried_object() says an object is being handled when a visible wrist is inside (or
   right next to) its box.
3. The object type is looked up in object_weights.json (typical kg), and REBA's load rule
   turns kg into the load score: under 5 kg +0, 5-10 kg +1, over 10 kg +2.

The camera cannot weigh anything, so this is always shown as an estimate. If no known
object is being handled, the manual LOAD_SCORE setting is used instead."""
import json
import os
import threading
import time

import numpy as np

from pipeline import config
from pipeline.reba import L_WR, R_WR

COCO_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.onnx"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_weights(path=None):
    """object type -> typical kg, from object_weights.json (keys starting with _ are notes)."""
    path = path or config.text("OBJECT_WEIGHTS_FILE", os.path.join(ROOT, "object_weights.json"))
    try:
        with open(path) as f:
            return {k: float(v) for k, v in json.load(f).items() if not k.startswith("_")}
    except (OSError, ValueError) as e:
        print(f"[ErgoWatch] could not read {path}: {e}; object weights disabled")
        return {}


def load_score_for(kg):
    """REBA load/force score for a weight in kg."""
    return 0 if kg < 5 else 1 if kg <= 10 else 2


def carried_object(kps, conf, detections, weights, margin=0.25, thr=0.3):
    """detections: list of (type name, [x1, y1, x2, y2]). Returns (name, kg) for the heaviest
    known object that a visible wrist is touching, or None."""
    best = None
    for name, box in detections or []:
        if name not in weights:
            continue
        x1, y1, x2, y2 = box
        mx, my = (x2 - x1) * margin, (y2 - y1) * margin
        for w in (L_WR, R_WR):
            if conf[w] >= thr and x1 - mx <= kps[w][0] <= x2 + mx and y1 - my <= kps[w][1] <= y2 + my:
                if best is None or weights[name] > best[1]:
                    best = (name, weights[name])
                break
    return best


class LoadEstimator:
    """Per person: remembers what was in their hands over the last moment and reports the
    type seen most often, so the load does not flicker between detector updates, when a
    hand hides the object, or when the detector misnames it for a frame."""

    def __init__(self, weights=None, manual_score=0, hold_seconds=None):
        self.weights = load_weights() if weights is None else weights
        self.manual = manual_score
        self.hold = config.num("OBJECT_HOLD_SECONDS", 1.5) if hold_seconds is None else hold_seconds
        self.margin = config.num("OBJECT_REACH_MARGIN", 0.25)
        self._hits = {}     # track id -> recent [(time, name)]

    def update(self, now, track, kps, conf, detections):
        """Returns {"score", "kg", "name", "source"}; source is "estimate" or "manual"."""
        hits = [h for h in self._hits.get(track, []) if now - h[0] <= self.hold]
        hit = carried_object(kps, conf, detections, self.weights, self.margin)
        if hit:
            hits.append((now, hit[0]))
        if not hits:
            self._hits.pop(track, None)
            return {"score": self.manual, "kg": None, "name": None, "source": "manual"}
        self._hits[track] = hits
        names = [n for _, n in hits]
        name = max(set(names), key=lambda n: (names.count(n), self.weights[n]))
        kg = self.weights[name]
        return {"score": max(load_score_for(kg), self.manual), "kg": kg, "name": name, "source": "estimate"}


class ObjectDetector:
    """COCO object detector in a background thread. Give it frames with submit(); read the
    most recent result with latest(). It never blocks the caller."""

    def __init__(self, weights=None):
        from rtmlib import YOLOX
        model = config.text("OBJECT_MODEL", COCO_URL)          # another YOLOX-format ONNX can be plugged in
        size = int(config.num("OBJECT_MODEL_SIZE", 640))
        self.det = YOLOX(model, model_input_size=(size, size), det_mode="multiclass",
                         score_thr=config.num("OBJECT_SCORE_THRESHOLD", 0.3),
                         backend="onnxruntime", device=config.text("DEVICE", "cpu"))
        classes_file = config.text("OBJECT_CLASSES_FILE")       # one type name per line, for a custom model
        self.classes = [l.strip() for l in open(classes_file)] if classes_file else self.det.COCO_CLASSES
        self.wanted = set(load_weights() if weights is None else weights)
        self.interval = 1.0 / max(config.num("OBJECT_DETECT_HZ", 2), 0.1)
        self.lock = threading.Lock()
        self.frame = None
        self.result = []
        self.result_time = 0.0
        self.rate = 0.0
        threading.Thread(target=self._loop, daemon=True).start()

    def detect(self, frame):
        """One synchronous detection: list of (type name, [x1, y1, x2, y2]) for known types."""
        boxes, cls = self.det(frame)
        out = []
        for box, c in zip(boxes, cls):
            name = self.classes[int(c)] if int(c) < len(self.classes) else str(int(c))
            if name in self.wanted:
                out.append((name, [float(v) for v in box]))
        return out

    def _loop(self):
        while True:
            with self.lock:
                frame, self.frame = self.frame, None
            if frame is None:
                time.sleep(0.02)
                continue
            t0 = time.time()
            try:
                result = self.detect(frame)
            except Exception as e:      # a detector hiccup must never stop the demo
                print(f"[ErgoWatch] object detector error: {e}")
                result = []
            with self.lock:
                self.result, self.result_time = result, time.time()
            took = time.time() - t0
            self.rate = 1.0 / max(took, 1e-6)
            time.sleep(max(self.interval - took, 0))

    def submit(self, frame):
        with self.lock:
            self.frame = frame

    def latest(self, max_age=1.0):
        """Most recent detections, or [] if they are older than max_age seconds."""
        with self.lock:
            if time.time() - self.result_time > max_age:
                return []
            return list(self.result)
