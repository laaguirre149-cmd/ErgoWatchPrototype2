"""Replays cached keypoints (from scripts/extract_keypoints.py) through the Engine with
simulated time. Used by scripts/replay.py and by the tests."""
import json
import os

import numpy as np


def load(npz_path):
    """Cached keypoints, plus cached object detections if <clip>.objects.json exists."""
    d = np.load(npz_path)
    clip = {k: d[k] for k in d.files}
    side = npz_path[:-4] + ".objects.json"
    if os.path.exists(side):
        with open(side) as f:
            clip["objects"] = [(t, [(n, b) for n, b in dets]) for t, dets in json.load(f)]
    return clip


def frames(clip, fps=15.0):
    """Yield (t, keypoints, scores, objects) at roughly `fps`, the way the live loop drops
    frames. objects = the most recent cached detections (None if the clip has none)."""
    next_t = 0.0
    objs, oi, current = clip.get("objects"), 0, None
    for t, kps, conf, n in zip(clip["t"], clip["kps"], clip["conf"], clip["n"]):
        if t + 1e-9 < next_t:
            continue
        next_t = t + 1.0 / fps
        while objs and oi < len(objs) and objs[oi][0] <= t + 1e-6:
            current, oi = objs[oi][1], oi + 1
        yield float(t), kps[:n].astype(float), conf[:n].astype(float), current


def replay(clip, engine, base_time=0.0, fps=15.0, on_frame=None):
    """Run a whole clip through the engine. Returns the list of per-frame states."""
    states = []
    for t, kps, conf, objects in frames(clip, fps):
        state = engine.process(kps, conf, base_time + t, objects=objects)
        state.t = t
        states.append(state)
        if on_frame:
            on_frame(state)
    return states
