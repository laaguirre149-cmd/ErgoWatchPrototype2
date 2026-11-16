import os
import threading
import time
import numpy as np
from rtmlib import YOLOX, RTMPose

BASE = "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/"
DETECTORS = {
    "tiny": (BASE + "yolox_tiny_8xb8-300e_humanart-6f3252f9.zip", (416, 416)),
    "s":    (BASE + "yolox_s_8xb8-300e_humanart-3ef259a7.zip", (640, 640)),
    "m":    (BASE + "yolox_m_8xb8-300e_humanart-c2c7a14a.zip", (640, 640)),
}
POSE = BASE + "rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip"


class PoseEstimator:
    """Detector finds people in a background thread. RTMPose runs on every frame
    using the most recent person boxes, so the video never freezes on detection."""

    def __init__(self):
        name = os.getenv("DETECTOR", "m")
        device = os.getenv("DEVICE", "cpu")
        det_url, det_size = DETECTORS[name]
        self.det = YOLOX(det_url, model_input_size=det_size, backend="onnxruntime", device=device)
        self.pose = RTMPose(POSE, model_input_size=(192, 256), backend="onnxruntime", device=device)
        print(f"[ErgoWatch] detector={name} (background) device={device}")

        self.lock = threading.Lock()
        self.latest_frame = None
        self.bboxes = None
        self.det_rate = 0.0
        threading.Thread(target=self._detect_loop, daemon=True).start()

    def _detect_loop(self):
        while True:
            with self.lock:
                frame = self.latest_frame
            if frame is None:
                time.sleep(0.01)
                continue
            t0 = time.time()
            bboxes = self.det(frame)
            with self.lock:
                self.bboxes = bboxes
            self.det_rate = 1.0 / max(time.time() - t0, 1e-6)

    def __call__(self, frame):
        with self.lock:
            self.latest_frame = frame.copy()
            bboxes = self.bboxes
        if bboxes is None:  # very first frame: wait for one detection
            bboxes = self.det(frame)
            with self.lock:
                self.bboxes = bboxes
        if len(bboxes) == 0:
            return np.empty((0, 17, 2)), np.empty((0, 17))
        return self.pose(frame, bboxes=bboxes)