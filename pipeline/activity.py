"""Repetition counting and static-posture time for one person (pure logic, no clock, no I/O).

REBA's activity score adds +1 when a body part is held static for over a minute and +1 for
repeated actions more than 4 times a minute; ActivityTracker.activity_score() gives that.

Rep counting works on an "excursion" value: how far the person has moved away from
upright standing, as the largest of
    trunk bend / REP_TRUNK_DEG,  knee bend / REP_KNEE_DEG,  arm elevation / REP_ARM_DEG.
Going above 1.0 and coming back below REP_RELEASE (0.5) is one rep. The component that
was largest names the movement: bend, squat or reach."""
import collections

from pipeline import config

MOVES = {"bend": "back_bend", "squat": "kneel_squat", "reach": "overhead_reach"}  # -> posture type


def settings():
    """Thresholds from .env (all optional)."""
    return {
        "trunk_deg": config.num("REP_TRUNK_DEG", 30),        # trunk bend that counts as "down"
        "knee_deg": config.num("REP_KNEE_DEG", 45),          # knee bend that counts as "down"
        "arm_deg": config.num("REP_ARM_DEG", 120),           # arm elevation (0 = hanging) that counts as "up"
        "release": config.num("REP_RELEASE", 0.5),           # fraction of the above to count as "back"
        "min_seconds": config.num("REP_MIN_SECONDS", 0.25),  # ignore blips shorter than this
        "set_gap": config.num("REP_SET_GAP_SECONDS", 20),    # no rep for this long starts a new set
        "rate_per_min": config.num("REP_RATE_PER_MIN", 4),   # more than this many per minute = repetitive
        "static_seconds": config.num("STATIC_SECONDS", 60),  # held this long = static (REBA +1)
        "static_tolerance": config.num("STATIC_ANGLE_TOLERANCE", 15),  # degrees of allowed movement
        "static_break": config.num("STATIC_BREAK_SECONDS", 1.0),   # movement must last this long to end a hold
        "trunk_min": config.num("STATIC_MIN_TRUNK_DEG", 20),       # below these a body part is "neutral"
        "knee_min": config.num("STATIC_MIN_KNEE_DEG", 30),         #   and cannot be in a static hold
        "arm_min": config.num("STATIC_MIN_ARM_DEG", 45),
        "smooth": config.num("ACTIVITY_SMOOTH_SECONDS", 0.3),
        "max_gap": config.num("ACTIVITY_MAX_GAP_SECONDS", 2.0),  # person missing this long resets everything
    }


class _Smooth:
    """Exponential smoothing with a time constant, robust to an uneven frame rate."""

    def __init__(self, tau):
        self.tau, self.value, self.t = tau, None, None

    def update(self, x, now):
        if self.value is None or self.tau <= 0:
            self.value = x
        else:
            a = min((now - self.t) / self.tau, 1.0)
            self.value += a * (x - self.value)
        self.t = now
        return self.value


class RepCounter:
    def __init__(self, cfg=None):
        self.cfg = cfg or settings()
        self.state = "up"
        self.count = 0            # reps in the current set
        self.total = 0
        self.move = None          # movement type of the current set
        self.times = collections.deque()   # (time, risky) of recent reps, for the per-minute rate
        self._pending_since = None
        self._down = None         # info about the rep in progress
        self._last_rep_t = None
        self._smooth = {k: _Smooth(self.cfg["smooth"]) for k in ("bend", "squat", "reach")}

    def excursion(self, angles, now):
        c = self.cfg
        raw = {"bend": angles.get("trunk", 0.0) / c["trunk_deg"],
               "squat": angles.get("knee", 0.0) / c["knee_deg"],
               "reach": angles.get("arm_elevation", 0.0) / c["arm_deg"]}
        vals = {k: self._smooth[k].update(v, now) for k, v in raw.items()}
        return max(vals.values()), vals

    def update(self, now, angles, risky=False, score=0):
        """Call every frame. Returns a dict describing a rep when one has just finished, else None."""
        e, vals = self.excursion(angles, now)
        if self._last_rep_t is not None and self.state == "up" and now - self._last_rep_t > self.cfg["set_gap"]:
            self.count, self.move, self._last_rep_t = 0, None, None
        if self.state == "up":
            if e >= 1.0:
                if self._pending_since is None:
                    self._pending_since = now
                if now - self._pending_since >= self.cfg["min_seconds"]:
                    self.state = "down"
                    self._down = {"start": self._pending_since, "peaks": dict(vals),
                                  "risky": risky, "score": score}
            else:
                self._pending_since = None
            return None
        d = self._down
        for k, v in vals.items():
            d["peaks"][k] = max(d["peaks"][k], v)
        d["risky"] = d["risky"] or risky
        d["score"] = max(d["score"], score)
        if e > self.cfg["release"]:
            return None
        # back up: one rep finished. Name it: arms overhead = reach, knees bent = squat, else bend.
        self.state, self._pending_since, self._down = "up", None, None
        p = d["peaks"]
        d["move"] = "reach" if p["reach"] >= 1 else "squat" if p["squat"] >= 1 else "bend"
        if self.move != d["move"]:
            self.count, self.move = 0, d["move"]     # a different movement starts a new set
        self.count += 1
        self.total += 1
        self._last_rep_t = now
        self.times.append((now, d["risky"]))
        return {"t": now, "move": d["move"], "category": MOVES[d["move"]], "risky": d["risky"],
                "score": d["score"], "duration": now - d["start"], "count": self.count}

    def rate(self, now, risky_only=False):
        """Reps finished in the last 60 seconds."""
        while self.times and now - self.times[0][0] > 60:
            self.times.popleft()
        return sum(1 for _, r in self.times if r or not risky_only)

    def repetitive(self, now):
        return self.rate(now) > self.cfg["rate_per_min"]

    def reset(self):
        self.state, self._pending_since, self._down = "up", None, None


class StaticTracker:
    """How long an awkward posture has been held without change.

    REBA counts a posture as static when "one or more body parts" are held for over a
    minute, so each body part has its own timer: trunk bend, knee bend, arm elevation.
    A part's timer runs while it is out of its neutral range (STATIC_MIN_*_DEG) and stays
    within `static_tolerance` degrees of where the hold began. Movement has to last
    `static_break` seconds to end a hold, so one bad pose frame does not reset it.
    Standing upright or walking is neutral and never counts as a static hold."""

    PARTS = {"trunk": "trunk_min", "knee": "knee_min", "arm_elevation": "arm_min"}

    def __init__(self, cfg=None):
        self.cfg = cfg or settings()
        self.start = {}            # part -> time its current hold began
        self.ref = {}              # part -> angle at the start of the hold
        self._moved_since = {}     # part -> time it first left the tolerance band
        self.risky_since = None    # start of the current risky stretch
        self._not_risky_since = None
        self.longest = 0.0
        self.longest_risky = 0.0
        self._smooth = {k: _Smooth(self.cfg["smooth"] * 2) for k in self.PARTS}

    def update(self, now, angles, risky=False):
        """Call every frame. Returns seconds the current awkward posture has been held."""
        for part, min_key in self.PARTS.items():
            if part not in angles:
                continue
            v = self._smooth[part].update(angles[part], now)
            out = part not in self.ref or abs(v - self.ref[part]) > self.cfg["static_tolerance"] \
                or v < self.cfg[min_key]
            if not out:
                self._moved_since.pop(part, None)
                continue
            first = self._moved_since.setdefault(part, now)
            if part not in self.ref or now - first >= self.cfg["static_break"]:
                self._moved_since.pop(part, None)
                if v >= self.cfg[min_key]:
                    self.start[part], self.ref[part] = now, v     # a new hold begins here
                else:
                    self.start.pop(part, None)
                    self.ref.pop(part, None)
        if risky:
            self._not_risky_since = None
            if self.risky_since is None:
                self.risky_since = now
        else:
            if self._not_risky_since is None:
                self._not_risky_since = now
            if now - self._not_risky_since > 1.0:   # brief flickers below the risk line are forgiven
                self.risky_since = None
        self.longest = max(self.longest, self.seconds(now))
        self.longest_risky = max(self.longest_risky, self.risky_seconds(now))
        return self.seconds(now)

    def seconds(self, now):
        """Longest current hold over all body parts."""
        return now - min(self.start.values()) if self.start else 0.0

    def held_part(self):
        return min(self.start, key=self.start.get) if self.start else None

    def risky_seconds(self, now):
        """Seconds the posture has been both held and risky."""
        if not self.start or self.risky_since is None:
            return 0.0
        return now - max(min(self.start.values()), self.risky_since)

    def is_static(self, now):
        return self.seconds(now) >= self.cfg["static_seconds"]

    def reset(self):
        self.start, self.ref, self._moved_since, self.risky_since = {}, {}, {}, None
        for sm in self._smooth.values():
            sm.value = None


class ActivityTracker:
    """Reps + static time for one person, and the REBA activity score they add up to."""

    def __init__(self, cfg=None):
        self.cfg = cfg or settings()
        self.reps = RepCounter(self.cfg)
        self.static = StaticTracker(self.cfg)
        self.last_t = None

    def update(self, now, result, risky):
        """result: REBA result before the activity score. Returns a finished rep dict or None."""
        if self.last_t is not None and now - self.last_t > self.cfg["max_gap"]:
            self.reps.reset()
            self.static.reset()
        self.last_t = now
        self.static.update(now, result["angles"], risky)
        return self.reps.update(now, result["angles"], risky, result["score"])

    def activity_score(self, now):
        return int(self.static.is_static(now)) + int(self.reps.repetitive(now))
