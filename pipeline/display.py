import time
import cv2
import numpy as np

PANEL_W = 560
BG = (28, 28, 28)
WHITE = (240, 240, 240)
GRAY = (150, 150, 150)
RED = (50, 50, 230)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def score_color(s):
    if s <= 3:
        return (80, 200, 80)    # green
    if s <= 7:
        return (0, 210, 255)    # yellow
    if s <= 10:
        return (0, 140, 255)    # orange
    return RED


def text(img, s, org, scale=0.7, color=WHITE, thick=2):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def fmt_time(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def compose(frame, people, top, alerts, hold_progress, fps, latency_ms, load_score, source_label="live",
            state=None):
    """people: list of (keypoints, reba_result). top: the highest-risk person, or None.
    state: the Engine's per-frame state (extra lines for the newer features), or None.
    Returns the video with a side panel."""
    h, w = frame.shape[:2]

    # The object being handled, with its weight estimate (prototype)
    held = getattr(state, "load_info", None)
    if held and held["source"] == "estimate":
        for name, box in (getattr(state, "objects", None) or []):
            if name == held["name"]:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 200, 0), 3)
                text(frame, f"{name} ~{held['kg']:g} kg (est.)", (max(x1 - 40, 5), max(y1 - 14, 30)), 1.0,
                     (255, 200, 0), 3)
                break

    # Score tag above each person's head (red if alert-level)
    for kps, r in people:
        x = int(min(max(kps[0][0] - 40, 0), w - 110))
        y = int(max(kps[0][1] - 70, 60))
        color = RED if alerts.is_risky(r) else score_color(r["score"])
        cv2.rectangle(frame, (x, y - 50), (x + 100, y + 12), color, -1)
        text(frame, str(r["score"]), (x + 18, y), 1.6, (255, 255, 255), 4)
        if r.get("worker_source") == "gait" or (r.get("worker_source") is None and "worker_conf" in r
                                                and len(people) > 1):
            names = getattr(state, "names", {}) or {}
            who = names.get(r["worker"], r["worker"]) if r.get("worker") else "unknown"
            text(frame, f"{who} {r['worker_conf'] * 100:.0f}%", (x, y + 44), 0.9, (255, 255, 255), 2)

    p = np.full((1080, PANEL_W, 3), BG, np.uint8)
    x0, x1 = 40, PANEL_W - 40

    text(p, "ERGOWATCH", (x0, 70), 1.3, WHITE, 3)
    text(p, f"{source_label} | {fps:.0f} FPS | {len(people)} scored | {latency_ms:.0f} ms", (x0, 110), 0.6, GRAY, 1)

    if state is not None and state.tracking:
        text(p, f"Worker: {state.worker_label} | high-risk today {fmt_time(state.today_risky)}",
             (x0, 145), 0.6, WHITE, 1)

    primary = top[1] if top else None
    if primary:
        s = primary["score"]
        risky = alerts.is_risky(primary)
        c = RED if risky else score_color(s)
        text(p, "REBA SCORE", (x0, 190), 0.9, GRAY, 2)
        text(p, str(s), (x0, 380), 7.0, c, 16)
        (tw, _), _ = cv2.getTextSize(str(s), FONT, 7.0, 16)
        text(p, "/ 15", (x0 + tw + 20, 380), 1.6, GRAY, 3)
        text(p, primary["risk"].upper() + " RISK", (x0, 450), 1.3, score_color(s), 3)
        category = getattr(state, "category", None)
        if category:
            from pipeline.classify import label
            text(p, "! " + label(category), (x0, 492), 0.8, RED, 2)
        elif primary["flags"]:
            text(p, "! " + ", ".join(primary["flags"]), (x0, 492), 0.7, RED, 2)
    else:
        text(p, "Step into view:", (x0, 300), 1.0, GRAY, 2)
        text(p, "full body, side-on", (x0, 345), 1.0, GRAY, 2)

    # 1-15 gauge with alert threshold marker
    gy = 545
    seg = (x1 - x0) / 15
    for i in range(1, 16):
        cv2.rectangle(p, (int(x0 + (i - 1) * seg) + 2, gy), (int(x0 + i * seg) - 2, gy + 28),
                      score_color(i), -1)
    tx = int(x0 + (alerts.threshold - 1) * seg)
    cv2.line(p, (tx, gy - 14), (tx, gy + 42), WHITE, 3)
    text(p, f"alert at {alerts.threshold}+", (tx - 50, gy + 75), 0.7, WHITE, 2)
    text(p, "1", (x0, gy + 75), 0.6, GRAY, 1)
    text(p, "15", (x1 - 25, gy + 75), 0.6, GRAY, 1)
    if primary:
        px = int(x0 + (primary["score"] - 0.5) * seg)
        cv2.drawMarker(p, (px, gy - 16), WHITE, cv2.MARKER_TRIANGLE_DOWN, 24, 4)

    # Body-segment breakdown (red = segment at its maximum band)
    y = 680
    if primary:
        rows = [("trunk", "Trunk", "trunk", 4), ("neck", "Neck", "neck", 2),
                ("upper_arm", "Upper arm", "upper_arm", 4), ("lower_arm", "Lower arm", "lower_arm", 2),
                ("legs", "Legs / knees", "knee", 3)]
        for key, label, akey, max_band in rows:
            ang = primary["angles"].get(akey)
            seg_score = primary["segments"][key]
            color = RED if key in ("trunk", "upper_arm") and seg_score == max_band else WHITE
            text(p, label, (x0, y), 0.75, color, 2)
            text(p, f"{ang:.0f} deg" if ang is not None else "--", (x0 + 220, y), 0.75, GRAY, 2)
            text(p, str(seg_score), (x1 - 25, y), 0.8, color, 2)
            y += 42
    if state is not None and state.reps is not None:
        move = {"bend": "bends", "squat": "squats", "reach": "reaches"}.get(state.rep_move, "reps")
        bonus = primary.get("activity", 0) if primary else 0
        text(p, f"Reps: {state.reps} {move} ({state.rep_rate}/min)   Held still: {state.static_seconds:.0f}s"
             + (f"   activity +{bonus}" if bonus else ""), (x0, 872), 0.6, WHITE, 1)
        if state.static_risky_seconds >= 5:   # progress toward the static-hold alert
            frac = min(state.static_risky_seconds / max(state.static_alert_seconds, 1), 1.0)
            cv2.rectangle(p, (x0, 880), (x1, 886), (70, 70, 70), -1)
            cv2.rectangle(p, (x0, 880), (x0 + int((x1 - x0) * frac), 886), (0, 140, 255), -1)
    info = getattr(state, "load_info", None)
    if info and info["source"] == "estimate":
        text(p, f"Load: about {info['kg']:g} kg ({info['name']}), +{info['score']}  ESTIMATE",
             (x0, 905), 0.6, (255, 200, 0), 1)
    elif load_score:
        text(p, f"Load: +{load_score} (set for this task)", (x0, 905), 0.6, GRAY, 1)

    # Hold progress toward an alert
    text(p, f"Risky posture held ({alerts.hold:.1f}s triggers alert)", (x0, 925), 0.6, GRAY, 1)
    cv2.rectangle(p, (x0, 938), (x1, 962), (70, 70, 70), -1)
    cv2.rectangle(p, (x0, 938), (x0 + int((x1 - x0) * hold_progress), 962), RED, -1)

    # Alert status
    flashing = alerts.sent_at and time.time() - alerts.sent_at < 3
    cv2.rectangle(p, (x0, 980), (x1, 1060), RED if flashing else (60, 60, 60), -1)
    text(p, "ALERT SENT TO PHONE" if flashing else "Monitoring - no alert",
         (x0 + 20, 1018), 0.9, WHITE if flashing else GRAY, 2)
    text(p, alerts.status, (x0 + 20, 1048), 0.55, WHITE, 1)

    p = cv2.resize(p, (int(PANEL_W * h / 1080), h), interpolation=cv2.INTER_AREA)
    return np.hstack([frame, p])