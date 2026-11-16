"""Keeps a stable number for each person from frame to frame (nearest body centre), so rep
counts, static-hold timers and identity belong to a person, not to a list position."""
import numpy as np

from pipeline.reba import L_HIP, L_SH, R_HIP, R_SH


class PersonTracker:
    def __init__(self, max_age=2.0):
        self.max_age = max_age
        self.tracks = {}      # id -> (centre, size, last seen)
        self.next_id = 1

    def assign(self, people, now):
        """people: list of keypoint arrays (17, 2). Returns a track id for each."""
        self.tracks = {i: t for i, t in self.tracks.items() if now - t[2] <= self.max_age}
        feats = []
        for kps in people:
            core = kps[[L_SH, R_SH, L_HIP, R_HIP]]
            centre = core.mean(axis=0)
            size = float(np.linalg.norm((core[0] + core[1]) / 2 - (core[2] + core[3]) / 2)) + 1e-6
            feats.append((centre, size))
        pairs = sorted((float(np.linalg.norm(c - t[0])) / max(s, t[1]), pi, tid)
                       for pi, (c, s) in enumerate(feats) for tid, t in self.tracks.items())
        ids, used = [None] * len(people), set()
        for dist, pi, tid in pairs:
            if dist > 1.5 or ids[pi] is not None or tid in used:   # moved over 1.5 torso lengths: new person
                continue
            ids[pi] = tid
            used.add(tid)
        for pi, (c, s) in enumerate(feats):
            if ids[pi] is None:
                ids[pi] = self.next_id
                self.next_id += 1
            self.tracks[ids[pi]] = (c, s, now)
        return ids
