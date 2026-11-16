import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


class AlertManager:
    """Sends a phone alert when a posture stays risky for ALERT_HOLD_SECONDS, then waits
    ALERT_COOLDOWN_SECONDS before alerting again. Risky = REBA score >= ALERT_THRESHOLD,
    or (if ALERT_ON_SEGMENTS=1) any body part in REBA's maximum band.
    Uses Pushover if PUSHOVER_TOKEN and PUSHOVER_USER are set, otherwise ntfy."""

    def __init__(self, dry_run=False):
        """dry_run=True (replays/tests): decide alerts as usual but never contact a phone."""
        self.dry_run = dry_run
        self.log = []   # every alert decision: dicts with time, kind, reason, score
        self.po_token = os.getenv("PUSHOVER_TOKEN", "").strip()
        self.po_user = os.getenv("PUSHOVER_USER", "").strip()
        self.ntfy_topic = os.getenv("NTFY_TOPIC", "").strip()
        self.service = "pushover" if self.po_token and self.po_user else "ntfy" if self.ntfy_topic else None
        if dry_run:
            self.service = None

        self.threshold = int(os.getenv("ALERT_THRESHOLD", "8"))
        self.segment_alerts = os.getenv("ALERT_ON_SEGMENTS", "1") == "1"
        self.hold = float(os.getenv("ALERT_HOLD_SECONDS", "0.5"))
        self.cooldown = float(os.getenv("ALERT_COOLDOWN_SECONDS", "10"))
        # Posture-specific alerts: the same posture type stays quiet for this long
        self.category_cooldown = float(os.getenv("ALERT_CATEGORY_COOLDOWN_SECONDS", "30").split("#")[0])
        self.last_by_category = {}
        self.risky_since = None
        self.last_trigger = float("-inf")
        self.sent_at = 0.0
        self.status = f"Alerts via {self.service}" if self.service else "No alert service set in .env"

    def is_risky(self, result):
        if result is None:
            return False
        return result["score"] >= self.threshold or (self.segment_alerts and bool(result["flags"]))

    def reason(self, result):
        if result["score"] >= self.threshold or not result["flags"]:
            return f"REBA score {result['score']}"
        return result["flags"][0]

    def update(self, result, now, category=None, focus=None, specific=False, mute=False):
        """Call every frame with the highest-risk person's REBA result (or None).
        Returns hold progress, 0 to 1.
        specific=True (FEATURE_POSTURE_ALERTS): `category` is the posture type from
        pipeline.classify (or None); each type has its own sound, message and cooldown, and
        if the worker chose focus moves (`focus`), only those types alert.
        mute=True keeps timing the hold but sends nothing this frame."""
        if not self.is_risky(result):
            self.risky_since = None
            return 0.0
        if self.risky_since is None:
            self.risky_since = now
        held = now - self.risky_since
        if held >= self.hold and now - self.last_trigger >= self.cooldown and not mute:
            reason = self.reason(result)
            if not specific:
                self.last_trigger = now
                self.notify(now, "posture", "ErgoWatch posture alert",
                            f"{reason} (REBA {result['score']}). Bend at the knees and keep the load close.",
                            reason, result["score"], held=held)
            elif self.allowed(category, focus, now):
                self.last_trigger = now
                self.last_by_category[category] = now
                self._notify_category(now, result, category, reason, held)
        return 1.0 if self.hold <= 0 else min(held / self.hold, 1.0)

    def allowed(self, category, focus, now):
        """Posture-specific rule: respect the worker's focus list and the per-type cooldown."""
        if focus and category not in focus:
            return False
        return now - self.last_by_category.get(category, float("-inf")) >= self.category_cooldown

    def _notify_category(self, now, result, category, reason, held):
        from pipeline import classify, config
        if category in classify.CATEGORIES:
            info = classify.CATEGORIES[category]
            self.notify(now, "posture", f"ErgoWatch: {info['label']}",
                        f"{info['message']} (REBA {result['score']})", info["label"], result["score"],
                        sound=classify.sound(category), url=f"{config.web_url()}/learn/{category}",
                        url_title="How to do it safely", category=category, held=held)
        else:   # risky, but no single posture type stands out
            self.notify(now, "posture", "ErgoWatch posture alert",
                        f"{reason} (REBA {result['score']}). Change position and reset your posture.",
                        reason, result["score"], category=None, held=held)

    def notify(self, now, kind, title, message, reason, score=0, sound=None, url=None,
               url_title=None, category=None, held=0.0):
        """Log an alert and send it to the phone on a background thread.
        kind: 'posture', 'static' or 'reps'. sound/url are Pushover extras."""
        self.log.append({"t": now, "kind": kind, "reason": reason, "score": score,
                         "held": held, "category": category, "message": message, "sound": sound})
        self._send(title, message, reason, held, sound, url, url_title)

    def _request(self, title, message, sound=None, url=None, url_title=None):
        if self.service == "pushover":
            fields = {"token": self.po_token, "user": self.po_user,
                      "title": title, "message": message, "priority": 1}
            if sound:
                fields["sound"] = sound
            if url:
                fields["url"] = url
                fields["url_title"] = url_title or "How to do it safely"
            data = urllib.parse.urlencode(fields).encode()
            return urllib.request.Request("https://api.pushover.net/1/messages.json", data=data)
        headers = {"Title": title, "Priority": "urgent", "Tags": "warning"}
        if url:
            headers["Click"] = url
        return urllib.request.Request(f"https://ntfy.sh/{self.ntfy_topic}", data=message.encode(),
                                      headers=headers)

    def _send(self, title, message, reason, held, sound=None, url=None, url_title=None):
        if not self.service:
            return

        def worker():
            t0 = time.time()
            try:
                urllib.request.urlopen(self._request(title, message, sound, url, url_title), timeout=5).read()
                self.sent_at = time.time()
                self.status = f"{reason} | held {held:.1f}s | sent in {(self.sent_at - t0) * 1000:.0f} ms"
            except urllib.error.HTTPError as e:
                try:
                    body = json.loads(e.read().decode())
                    detail = "; ".join(body.get("errors") or [body.get("error", "")]) or f"HTTP {e.code}"
                except Exception:
                    detail = f"HTTP {e.code}"
                self.status = f"Send failed: {detail}"
                print(f"[ErgoWatch] alert failed: {detail}")
            except Exception as e:
                self.status = f"Send failed: {e.__class__.__name__}"
                print(f"[ErgoWatch] alert failed: {e}")

        threading.Thread(target=worker, daemon=True).start()
