"""Gait fallback (PROTOTYPE): recognise a worker from a simple body "signature" built from
pose keypoints, for when nobody has checked in or two people are in frame.

Signature = limb-length ratios (upper arm, forearm, thigh, shin, each divided by torso
length) + walking cadence (steps per minute) when the person has been seen walking.
It is enrolled once per worker after check-in and matched later against whoever is seen.

Limits, stated plainly: this uses one 2D camera, so the ratios shift with camera angle and
body orientation, and people's proportions differ by only a few percent. The match comes
with a confidence value, and below GAIT_MIN_CONFIDENCE the answer is "unknown worker"
rather than a guess. It stores a handful of numbers per worker, never images."""
import collections
import math

import numpy as np

from pipeline import config
from pipeline.reba import (L_ANK, L_EL, L_HIP, L_KNEE, L_SH, L_WR, R_ANK, R_EL, R_HIP, R_KNEE, R_SH,
                           R_WR)

LIMBS = {
    "upper_arm": [(L_SH, L_EL), (R_SH, R_EL)],
    "forearm": [(L_EL, L_WR), (R_EL, R_WR)],
    "thigh": [(L_HIP, L_KNEE), (R_HIP, R_KNEE)],
    "shin": [(L_KNEE, L_ANK), (R_KNEE, R_ANK)],
}


def settings():
    return {
        "min_confidence": config.num("GAIT_MIN_CONFIDENCE", 0.6),
        "min_seconds": config.num("GAIT_MIN_SECONDS", 3.0),         # upright time needed before matching
        "enroll_seconds": config.num("GAIT_ENROLL_SECONDS", 6.0),   # upright time needed to enroll
        "ratio_tolerance": config.num("GAIT_RATIO_TOLERANCE", 0.05),      # a ratio difference this big = 1 "unit" apart
        "cadence_tolerance": config.num("GAIT_CADENCE_TOLERANCE", 12.0),  # steps/min
        "upright_deg": config.num("GAIT_UPRIGHT_DEG", 15.0),        # only measure while standing/walking upright
    }


class GaitAccumulator:
    """Collects one person's measurements over time and turns them into a signature."""

    def __init__(self, cfg=None, max_samples=600):
        self.cfg = cfg or settings()
        self.ratios = {k: collections.deque(maxlen=max_samples) for k in LIMBS}
        self.upright_seconds = 0.0
        self.last_t = None
        # walking cadence: recent ankle-to-ankle offsets (sideways and vertical, in torso lengths)
        self._ankles = collections.deque(maxlen=240)     # (time, dx, dy)
        self._cadences = collections.deque(maxlen=40)    # cadence estimates from short windows
        self._last_cadence_t = 0.0

    def add(self, now, kps, conf, angles, thr=0.5):
        """angles: the REBA angles for this frame. Only neutral upright frames are measured
        (standing or walking, arms down): bending, squatting or raised arms distort the ratios."""
        dt = 0.0 if self.last_t is None else min(max(now - self.last_t, 0.0), 0.5)
        self.last_t = now
        if (angles.get("trunk", 0) > self.cfg["upright_deg"] or angles.get("knee", 0) > 35
                or angles.get("arm_elevation", 0) > 60
                or min(conf[[L_SH, R_SH, L_HIP, R_HIP]]) < thr):
            return
        torso = float(np.linalg.norm((kps[L_SH] + kps[R_SH]) / 2 - (kps[L_HIP] + kps[R_HIP]) / 2))
        if torso < 1:
            return
        self.upright_seconds += dt
        for name, pairs in LIMBS.items():
            lengths = [float(np.linalg.norm(kps[a] - kps[b])) for a, b in pairs if min(conf[[a, b]]) >= thr]
            if lengths:
                self.ratios[name].append(max(lengths) / torso)   # the less foreshortened side
        if min(conf[[L_ANK, R_ANK]]) >= thr:
            d = (kps[L_ANK] - kps[R_ANK]) / torso
            self._ankles.append((now, float(d[0]), float(d[1])))
            if now - self._last_cadence_t >= 1.0:
                self._last_cadence_t = now
                c = self._window_cadence(now)
                if c:
                    self._cadences.append(c)

    def _window_cadence(self, now, window=6.0):
        """Steps per minute from the last few seconds, or None if the person is not walking.
        Each step swaps which foot is ahead: seen side-on that flips the sideways ankle offset,
        walking toward or away from the camera it flips the vertical one. Whichever of the two
        shows the more regular swapping is used."""
        pts = [p for p in self._ankles if now - p[0] <= window]
        if len(pts) < 20 or pts[-1][0] - pts[0][0] < window * 0.6:
            return None
        t = np.array([p[0] for p in pts])
        best = None
        for col in (1, 2):
            x = np.array([p[col] for p in pts])
            # remove slow drift (turning, perspective) with a 1.2 s moving average
            trend = np.array([x[(t >= ti - 0.6) & (t <= ti + 0.6)].mean() for ti in t])
            x = x - trend
            if x.std() < 0.08:           # feet barely moving: standing, not walking
                continue
            band = 0.3 * x.std()
            side, crossings = 0, []
            for ti, xi in zip(t, x):
                now_side = 1 if xi > band else -1 if xi < -band else 0
                if now_side and now_side != side:
                    if side:
                        crossings.append(ti)
                    side = now_side
            gaps = [b - a for a, b in zip(crossings, crossings[1:])]
            good = [g for g in gaps if 0.3 <= g <= 0.9]          # 67 to 200 steps a minute
            if len(good) >= 5 and len(good) >= 0.7 * len(gaps) and (best is None or len(good) > best[0]):
                best = (len(good), 60.0 / float(np.median(good)))
        return best[1] if best else None

    def cadence(self):
        """Typical steps per minute seen so far, or None if not enough walking was seen."""
        if len(self._cadences) < 5:
            return None
        return float(np.median(self._cadences))

    def ready(self, seconds=None):
        need = self.cfg["min_seconds"] if seconds is None else seconds
        return self.upright_seconds >= need and all(len(v) >= 10 for v in self.ratios.values())

    def signature(self):
        """{"ratios": {...}, "cadence": steps/min or None}. A limb is longest on screen when it
        lies flat to the camera, so the 85th percentile is used rather than the average."""
        sig = {"ratios": {k: round(float(np.percentile(v, 85)), 4) for k, v in self.ratios.items() if len(v)},
               "cadence": None, "seconds": round(self.upright_seconds, 1)}
        c = self.cadence()
        if c:
            sig["cadence"] = round(c, 1)
        return sig


def distance(a, b, cfg=None):
    """How far apart two signatures are, in "tolerance units" (0 = identical)."""
    cfg = cfg or settings()
    keys = [k for k in a["ratios"] if k in b["ratios"]]
    if not keys:
        return float("inf")
    terms = [((a["ratios"][k] - b["ratios"][k]) / cfg["ratio_tolerance"]) ** 2 for k in keys]
    if a.get("cadence") and b.get("cadence"):
        terms.append(((a["cadence"] - b["cadence"]) / cfg["cadence_tolerance"]) ** 2)
    return math.sqrt(sum(terms) / len(terms))


def similarity(a, b, cfg=None):
    return math.exp(-0.5 * distance(a, b, cfg) ** 2)


def identify(sig, enrolled, cfg=None):
    """enrolled: {worker_id: signature}. Returns (worker_id or None, confidence 0-1).
    Confidence = similarity to the best match, minus half the similarity to the runner-up
    (two enrolled people who both fit make the answer less certain)."""
    cfg = cfg or settings()
    if not enrolled:
        return None, 0.0
    scores = sorted(((similarity(sig, s, cfg), w) for w, s in enrolled.items()), reverse=True)
    best, worker = scores[0]
    conf = best - (0.5 * scores[1][0] if len(scores) > 1 else 0.0)
    conf = max(0.0, min(1.0, conf))
    return (worker if conf >= cfg["min_confidence"] else None), conf
