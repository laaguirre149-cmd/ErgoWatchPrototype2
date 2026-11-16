"""Cumulative exposure: how long each worker spends in high-risk postures, by body region.

Pure logic only (no database, no clock): feed it one REBA result per frame with a
timestamp and it returns finished "bucket" rows (a few seconds each) ready to be stored.
No video or keypoints are kept, only seconds and scores.

A frame counts as high-risk when the alert rule says so (REBA >= ALERT_THRESHOLD, or a
max-band flag). During high-risk time, a body region is counted when its own REBA
component score is at or above that region's threshold (REGION_THRESHOLDS).
Wrists cannot be measured by the camera (assumed neutral), so wrist time is always 0."""

REGIONS = ["trunk", "neck", "upper_arm", "lower_arm", "wrist", "legs"]
REGION_LABELS = {"trunk": "Trunk / back", "neck": "Neck", "upper_arm": "Upper arms",
                 "lower_arm": "Lower arms", "wrist": "Wrists", "legs": "Legs / knees"}
# REBA component score at which a region counts toward high-risk time
REGION_THRESHOLDS = {"trunk": 3, "neck": 2, "upper_arm": 3, "lower_arm": 2, "wrist": 2, "legs": 2}


def parse_thresholds(spec):
    """'trunk=3,neck=2' -> dict, on top of the defaults."""
    out = dict(REGION_THRESHOLDS)
    for part in (spec or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            if k.strip() in out and v.strip().isdigit():
                out[k.strip()] = int(v.strip())
    return out


class _Bucket:
    def __init__(self, start):
        self.start = start
        self.duration = 0.0
        self.risky_s = 0.0
        self.region_s = {r: 0.0 for r in REGIONS}
        self.region_sum = {r: 0.0 for r in REGIONS}
        self.reba_sum = 0.0
        self.reba_max = 0


class ExposureAccumulator:
    def __init__(self, bucket_seconds=2.0, max_gap=1.0, thresholds=None):
        self.bucket_seconds = bucket_seconds
        self.max_gap = max_gap          # longer gaps between frames are not counted as exposure
        self.thresholds = thresholds or dict(REGION_THRESHOLDS)
        self._buckets = {}              # worker_id -> _Bucket
        self._last_t = {}               # worker_id -> time of that worker's previous frame

    def add(self, now, worker_id, result, risky):
        """One scored frame for one worker. Returns a list of finished rows (usually empty)."""
        rows = []
        last = self._last_t.get(worker_id)
        self._last_t[worker_id] = now
        dt = 0.0 if last is None else now - last
        if dt < 0 or dt > self.max_gap:
            dt = 0.0
        b = self._buckets.get(worker_id)
        if b is not None and now - b.start >= self.bucket_seconds:
            rows.append(self._row(worker_id, b))
            b = None
        if b is None:
            b = self._buckets[worker_id] = _Bucket(now)
        seg = dict(result["segments"])
        seg.setdefault("wrist", 1)
        b.duration += dt
        b.reba_sum += result["score"] * dt
        b.reba_max = max(b.reba_max, result["score"])
        for r in REGIONS:
            b.region_sum[r] += seg[r] * dt
        if risky:
            b.risky_s += dt
            for r in REGIONS:
                if seg[r] >= self.thresholds[r]:
                    b.region_s[r] += dt
        return [r for r in rows if r]

    def flush(self):
        """Finish every open bucket (call at shutdown or when the worker changes)."""
        rows = [self._row(w, b) for w, b in self._buckets.items()]
        self._buckets.clear()
        self._last_t.clear()
        return [r for r in rows if r]

    def _row(self, worker_id, b):
        if b.duration <= 0:
            return None
        row = {"ts": b.start, "worker_id": worker_id, "duration": round(b.duration, 3),
               "risky_s": round(b.risky_s, 3), "reba_avg": round(b.reba_sum / b.duration, 2),
               "reba_max": b.reba_max}
        for r in REGIONS:
            row[f"{r}_s"] = round(b.region_s[r], 3)
            row[f"{r}_avg"] = round(b.region_sum[r] / b.duration, 2)
        return row
