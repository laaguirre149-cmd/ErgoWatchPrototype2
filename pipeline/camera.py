import os
import threading
import time
import cv2


class FrameSource:
    """Webcam index (0) or stream URL (rtsp://...). Keeps only the newest frame."""

    finished = False  # a live source never ends; VideoFileSource sets this at end of file

    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.frame_time = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            with self.lock:
                self.frame, self.frame_time = frame, time.time()

    def read(self):
        with self.lock:
            if self.frame is None:
                return None, 0.0
            return self.frame.copy(), self.frame_time

    def release(self):
        self.running = False
        self.cap.release()


class VideoFileSource(FrameSource):
    """A recorded video file (.mov, .mp4, ...) played back at its real speed, so it behaves
    like a live camera: only the newest frame is kept, and frames are skipped if the
    pipeline is slower than the video. OpenCV applies the phone's rotation metadata.
    Sets `finished` when the file ends (unless loop=True)."""

    def __init__(self, path, loop=False):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Video file not found: {path}")
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_interval = 1.0 / fps if fps and fps > 0 else 1.0 / 30
        self.loop = loop
        self.finished = False
        self.frame, self.frame_time = None, 0.0
        self.lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        start, n = time.time(), 0
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                if self.loop and n:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start, n = time.time(), 0
                    continue
                self.finished = True
                return
            delay = start + n * self.frame_interval - time.time()  # wait for this frame's playback time
            if delay > 0:
                time.sleep(delay)
            n += 1
            with self.lock:
                self.frame, self.frame_time = frame, time.time()
