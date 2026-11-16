"""Everything that happens to one frame's keypoints after pose estimation, with no video or
window code: REBA scoring, exposure tracking, alerts. run.py calls Engine.process() on
every live frame; scripts/replay.py and the tests call the same code on cached keypoints
with simulated time, so what is tested is what runs in the demo."""
import threading
import time
import types

import collections

from pipeline import activity, classify, config, gait
from pipeline.exposure import ExposureAccumulator, parse_thresholds
from pipeline.objects import LoadEstimator
from pipeline.reba import reba, with_activity
from pipeline.store import UNASSIGNED, AsyncWriter
from pipeline.tracking import PersonTracker

MOVE_WORDS = {"bend": "bends", "squat": "squats", "reach": "overhead reaches"}


class Engine:
    def __init__(self, alerts, store=None, features=None, load_score=0, bay=None, live=True):
        """store: a pipeline.store.Store, or None to keep nothing.
        live=True: database writes and check-in polling run on background threads.
        live=False (replay/tests): everything is done inline so results are deterministic."""
        self.alerts = alerts
        self.store = store
        self.features = features if features is not None else config.features()
        self.load_score = load_score
        self.bay = bay or config.text("BAY", "1")
        self.live = live
        self.worker_id = config.text("DEFAULT_WORKER", UNASSIGNED)   # who exposure is attributed to
        self.worker_name = self.worker_id
        self.checked_in = False
        self.today_risky = 0.0   # high-risk seconds today for the active worker (for the display)

        self.exposure = None
        if store is not None and self.features.get("exposure"):
            self.exposure = ExposureAccumulator(
                bucket_seconds=config.num("EXPOSURE_BUCKET_SECONDS", 2),
                thresholds=parse_thresholds(config.text("EXPOSURE_REGION_THRESHOLDS")))
        self.tracker = PersonTracker()
        self.activity = {}       # track id -> ActivityTracker (reps + static time per person)
        self.act_cfg = activity.settings()
        self.activity_in_score = config.flag("ACTIVITY_IN_SCORE")       # add REBA's activity score
        self.static_alert_seconds = config.num("STATIC_ALERT_SECONDS", 60)
        self.static_alert_repeat = config.num("STATIC_ALERT_REPEAT_SECONDS", 60)
        self.rep_alert_count = int(config.num("REP_ALERT_COUNT", 5))    # risky reps per minute
        self.rep_alert_cooldown = config.num("REP_ALERT_COOLDOWN_SECONDS", 60)
        self.last_rep_alert = -1e9
        self.loads = LoadEstimator(manual_score=load_score) if self.features.get("object_weight") else None
        self.gait_on = bool(self.features.get("gait"))
        self.gait_cfg = gait.settings()
        self.gaits = {}          # track id -> GaitAccumulator (body signature being measured)
        self.identity = {}       # track id -> (worker id or None, confidence, time of last match)
        self.checked_tracks = {} # track id -> worker id, for a person who was alone when checked in
        self.signatures = {}     # enrolled signatures: worker id -> signature
        self.names = {}          # worker id -> display name
        self.specific = bool(self.features.get("posture_alerts"))
        self.classify_thr = classify.thresholds()
        self.focus = []          # the active worker's chosen focus moves (empty = alert on everything)
        self.episode = None      # the risky episode in progress, stored as one event when it ends
        self.writer = AsyncWriter(store) if (store is not None and live) else None
        if store is not None and live:
            self._refresh_worker()
            threading.Thread(target=self._watch_worker, daemon=True).start()

    # --- active worker (set by check-in through the web app) ---
    def _refresh_worker(self):
        c = self.store.active_checkin(self.bay)
        wid = c["worker_id"] if c else config.text("DEFAULT_WORKER", UNASSIGNED)
        w = self.store.worker(wid)
        self.worker_name = w["name"] if w else wid
        self.checked_in = c is not None
        self.worker_id = wid
        self.focus = [c for c in (w["focus"].split(",") if w else []) if c]
        if self.gait_on:
            self.signatures = self.store.signatures()
            self.names = {x["id"]: x["name"] for x in self.store.workers()}
        if self.exposure:
            from pipeline.store import day_start
            self.today_risky = self.store.exposure(wid, day_start(), time.time() + 1)["risky"]

    def _watch_worker(self):
        while True:
            time.sleep(1.0)
            try:
                self._refresh_worker()
            except Exception as e:
                print(f"[ErgoWatch] could not read check-in: {e}")

    def set_worker(self, worker_id, name=None):
        """Replay/tests: set the active worker directly."""
        self.worker_id, self.worker_name, self.checked_in = worker_id, name or worker_id, True

    def _wid(self, r):
        """Worker id a person's data is stored under ("unassigned" if not known)."""
        return r.get("worker") or config.text("DEFAULT_WORKER", UNASSIGNED)

    def _db(self, fn, *args, **kwargs):
        if self.store is None:
            return
        if self.writer:
            self.writer.call(fn, *args, **kwargs)
        else:
            getattr(self.store, fn)(*args, **kwargs)

    # --- per frame ---
    def process(self, keypoints, scores, now, objects=None):
        """keypoints (P,17,2), scores (P,17) from the pose model; now = frame time in seconds.
        objects: latest object detections [(type, box), ...] or None.
        Returns a state object for the display."""
        people, confs = [], []
        for kps, conf in zip(keypoints, scores):
            r = reba(kps, conf, load_score=self.load_score)
            if r:
                people.append((kps, r))
                confs.append(conf)

        ids = self.tracker.assign([kps for kps, _ in people], now)
        for i, tid in enumerate(ids):
            kps, r = people[i]
            if self.loads:      # estimated load from the object in the person's hands
                load = self.loads.update(now, tid, kps, confs[i], objects)
                if load["score"] != self.load_score:
                    r = reba(kps, confs[i], load_score=load["score"])
                    people[i] = (kps, r)
                r["load_info"] = load
            r["track"] = tid
        self._who(now, people, confs)
        if self.features.get("reps"):
            for _, r in people:
                self._activity(now, r["track"], r)
        self.activity = {t: a for t, a in self.activity.items() if t in self.tracker.tracks}

        # Highest-risk person: alert-level first, then highest score
        top = max(people, key=lambda pr: (self.alerts.is_risky(pr[1]), pr[1]["score"]), default=None)
        category = classify.classify(top[1], top[1].get("load", 0), self.classify_thr) if top else None
        act = self.activity.get(top[1]["track"]) if top else None
        sent_before = len(self.alerts.log)
        # A static-hold alert that is about to fire replaces the repeat posture alert
        mute = act is not None and act.static.risky_seconds(now) > 0 and \
            self._static_due(act) - act.static.risky_seconds(now) <= self.alerts.cooldown
        hold_progress = self.alerts.update(top[1] if top else None, now, category=category,
                                           focus=self.focus, specific=self.specific, mute=mute)
        self._episode(now, top[1] if top else None, category, len(self.alerts.log) > sent_before)
        if act:
            self._static_alert(now, act, top[1], category)

        if self.exposure and top:
            # With gait on, every person's exposure goes to whoever they were recognised as
            # (unknown people share the "unassigned" record); otherwise the top person only.
            scored = sorted(people, key=lambda pr: -pr[1]["score"]) if self.gait_on else [top]
            done = set()
            for _, r in scored:
                wid = self._wid(r)
                if wid in done:
                    continue
                done.add(wid)
                rows = self.exposure.add(now, wid, r, self.alerts.is_risky(r))
                if rows:
                    self._db("add_samples", rows)

        if top and self.gait_on:
            who = top[1]
            self_name = self.names.get(who["worker"], who["worker"]) if who["worker"] else "unknown worker"
            worker_label = self_name if who["worker_source"] == "check-in" else \
                f"{self_name} (body match {who['worker_conf'] * 100:.0f}%)" if who["worker"] else \
                f"unknown worker ({who['worker_conf'] * 100:.0f}% match)" if self.signatures else "nobody checked in"
        else:
            worker_label = self.worker_name + ("" if self.checked_in else " (no check-in)")
        return types.SimpleNamespace(
            worker_label=worker_label, names=self.names,
            people=people, top=top, hold_progress=hold_progress,
            load_score=top[1]["load"] if top else self.load_score,
            load_info=top[1].get("load_info") if top else None, objects=objects or [],
            worker_id=self.worker_id, worker_name=self.worker_name, checked_in=self.checked_in,
            today_risky=self.today_risky, tracking=self.exposure is not None,
            category=category if (top and self.alerts.is_risky(top[1])) else None, focus=self.focus,
            reps=act.reps.count if act else None, rep_move=act.reps.move if act else None,
            rep_rate=act.reps.rate(now) if act else 0,
            static_seconds=act.static.seconds(now) if act else 0.0,
            static_risky_seconds=act.static.risky_seconds(now) if act else 0.0,
            longest_static=act.static.longest_risky if act else 0.0,
            static_alert_seconds=self.static_alert_seconds)

    def _who(self, now, people, confs):
        """Decide which worker each person is. Sets r["worker"] (id or None),
        r["worker_conf"] and r["worker_source"] ("check-in", "gait" or None)."""
        for _, r in people:
            r["worker"], r["worker_conf"], r["worker_source"] = None, 0.0, None
        if not self.gait_on:
            for _, r in people:      # original rule: whoever is in view is the checked-in worker
                r["worker"], r["worker_source"] = self.worker_id, "check-in" if self.checked_in else None
            return
        live = set()
        for (kps, r), conf in zip(people, confs):
            tid = r["track"]
            live.add(tid)
            acc = self.gaits.get(tid)
            if acc is None:
                acc = self.gaits[tid] = gait.GaitAccumulator(self.gait_cfg)
            acc.add(now, kps, conf, r["angles"])
            last = self.identity.get(tid)
            if acc.ready() and (last is None or now - last[2] >= 0.5):
                wid, c = gait.identify(acc.signature(), self.signatures, self.gait_cfg)
                self.identity[tid] = (wid, c, now)
        self.gaits = {t: a for t, a in self.gaits.items() if t in self.tracker.tracks}
        self.identity = {t: v for t, v in self.identity.items() if t in self.tracker.tracks}
        self.checked_tracks = {t: w for t, w in self.checked_tracks.items()
                               if t in self.tracker.tracks and w == self.worker_id and self.checked_in}

        if len(people) == 1 and self.checked_in:
            # One person and a check-in: trust the check-in, and enroll their signature if missing.
            r = people[0][1]
            r["worker"], r["worker_conf"], r["worker_source"] = self.worker_id, 1.0, "check-in"
            self.checked_tracks[r["track"]] = self.worker_id
            acc = self.gaits[r["track"]]
            if self.worker_id not in self.signatures and acc.ready(self.gait_cfg["enroll_seconds"]):
                sig = acc.signature()
                self.signatures[self.worker_id] = sig
                self._db("set_signature", self.worker_id, sig)
                print(f"[ErgoWatch] body signature enrolled for {self.worker_name}: {sig}")
            return
        # Nobody checked in, or several people: match each person against enrolled signatures.
        taken = {}
        for _, r in people:
            if r["track"] in self.checked_tracks:
                # This person was alone in view when checked in and has been followed since:
                # they keep the check-in identity when someone else walks into the frame.
                r["worker"], r["worker_conf"], r["worker_source"] = self.worker_id, 1.0, "check-in"
                taken[self.worker_id] = r
        for _, r in people:
            if r["worker_source"] == "check-in":
                continue
            wid, c, _ = self.identity.get(r["track"], (None, 0.0, 0))
            r["worker_conf"] = c
            if wid and (wid not in taken or c > taken[wid]["worker_conf"]):
                if wid in taken:     # two people matched the same worker: the weaker one is unknown
                    taken[wid]["worker"], taken[wid]["worker_source"] = None, None
                r["worker"], r["worker_source"] = wid, "gait"
                taken[wid] = r

    def _activity(self, now, tid, r):
        """Rep counting and static time for one person; adds REBA's activity score to r."""
        act = self.activity.get(tid)
        if act is None:
            act = self.activity[tid] = activity.ActivityTracker(self.act_cfg)
            act.static_alerted = 0.0
        risky = self.alerts.is_risky(r)          # judged before the activity bonus
        rep = act.update(now, r, risky)
        if self.activity_in_score:
            with_activity(r, act.activity_score(now))
        if rep:
            self._db("add_event", now, self._wid(r), "rep", rep["category"], rep["score"],
                     rep["duration"], 1, 0)
            n_risky = act.reps.rate(now, risky_only=True)
            if (rep["risky"] and n_risky >= self.rep_alert_count
                    and now - self.last_rep_alert >= self.rep_alert_cooldown
                    and not (self.specific and self.focus and rep["category"] not in self.focus)):
                self.last_rep_alert = now
                words = MOVE_WORDS[rep["move"]]
                reason = f"{n_risky} risky {words} in a minute"
                self.alerts.notify(now, "reps", "ErgoWatch: repetition",
                                   f"{reason}. Slow down, reset your form, or switch tasks for a bit.",
                                   reason, r["score"], sound=config.text("SOUND_REPS", "mechanical"),
                                   url=f"{config.web_url()}/learn/reps", category=rep["category"])
                self._db("add_event", now, self._wid(r), "rep_alert", rep["category"], r["score"],
                         0, n_risky, 1)

    def _static_alert(self, now, act, r, category=None):
        """Alert when a risky posture has been held without change for STATIC_ALERT_SECONDS."""
        held = act.static.risky_seconds(now)
        if held < act.static_alerted:            # the hold ended; arm the alert again
            act.static_alerted = 0.0
        if held >= self._static_due(act):
            act.static_alerted = held
            if self.specific and self.focus and category not in self.focus:
                return                           # not one of the worker's focus moves
            self.alerts.last_trigger = now       # counts toward the normal cooldowns too
            self.alerts.last_by_category[category] = now
            what = classify.label(category) if self.specific and category else "Risky posture"
            reason = f"{what} held {held:.0f}s"
            self.alerts.notify(now, "static", "ErgoWatch: static hold",
                               f"{reason} without moving. Stand up, stretch, change position.",
                               reason, r["score"], sound=config.text("SOUND_STATIC", "tugboat"),
                               url=f"{config.web_url()}/learn/static", category=category, held=held)
            self._db("add_event", now, self._wid(r), "static", category or "", r["score"], held, 0, 1)

    def _static_due(self, act):
        """Risky-hold seconds at which this person's next static-hold alert fires."""
        return self.static_alert_seconds if not act.static_alerted \
            else act.static_alerted + self.static_alert_repeat

    def _episode(self, now, r, category, alerted):
        """Group consecutive risky frames into one "episode" and store it when it ends, with
        the posture type seen for most of it. These rows rank the worker's most frequent moves."""
        risky = r is not None and self.alerts.is_risky(r)
        ep = self.episode
        if risky:
            if ep is None:
                ep = self.episode = {"start": now, "last": now, "cats": collections.Counter(),
                                     "reba": 0, "alerted": False, "worker": self._wid(r)}
            if r.get("worker"):
                ep["worker"] = r["worker"]       # recognised part-way through: credit the right person
            ep["cats"][category or ""] += 1
            ep["last"] = now
            ep["reba"] = max(ep["reba"], r["score"])
            ep["alerted"] = ep["alerted"] or alerted
        elif ep is not None and now - ep["last"] > 1.0:     # short flickers stay in the same episode
            self._end_episode()

    def _end_episode(self):
        ep, self.episode = self.episode, None
        if ep and ep["last"] - ep["start"] >= self.alerts.hold:   # shorter blips never count as an episode
            cat = ep["cats"].most_common(1)[0][0]
            self._db("add_event", ep["start"], ep["worker"], "posture", cat, ep["reba"],
                     ep["last"] - ep["start"], 0, int(ep["alerted"]))

    def close(self):
        self._end_episode()
        if self.exposure:
            rows = self.exposure.flush()
            if rows:
                self._db("add_samples", rows)
        if self.writer:
            self.writer.close()
