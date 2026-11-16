"""Names the main risk in a posture so the alert can say something specific.

classify() looks at the REBA angles and returns one posture type:
    overhead_reach, risky_lift, kneel_squat, back_bend, neck   (or None)
Each type has its own Pushover sound and a short message. iOS does not allow custom
vibration patterns for push notifications, so the sound is what tells them apart."""
from pipeline import config

# Order = priority when more than one applies.
CATEGORIES = {
    "overhead_reach": {
        "label": "Overhead reach", "sound": "climb",
        "message": "Arms above shoulders. Step up or lower the work to chest height.",
    },
    "risky_lift": {
        "label": "Risky lift", "sound": "siren",
        "message": "Lifting with a bent back. Squat, keep the load close, lift with your legs.",
    },
    "kneel_squat": {
        "label": "Kneeling / deep squat", "sound": "bike",
        "message": "Deep knee bend. Use a kneeling pad or stool, and switch sides.",
    },
    "back_bend": {
        "label": "Back bend", "sound": "falling",
        "message": "Back bent forward. Bend your knees and hinge at the hips instead.",
    },
    "neck": {
        "label": "Neck strain", "sound": "magic",
        "message": "Neck bent. Bring the work to eye level or change your angle.",
    },
}
ORDER = list(CATEGORIES)


def thresholds():
    return {
        "overhead_deg": config.num("CLASSIFY_OVERHEAD_DEG", 85),    # arm elevation (0 = hanging, 90 = shoulder height)
        "kneel_deg": config.num("CLASSIFY_KNEEL_DEG", 60),          # knee bend
        "bend_deg": config.num("CLASSIFY_BEND_DEG", 20),            # trunk bend
        "neck_deg": config.num("CLASSIFY_NECK_DEG", 20),
    }


def classify(result, load_score=0, thr=None):
    """result: a REBA result dict. load_score > 0 means the person is handling a load.
    Returns the posture type that best explains the risk, or None if nothing stands out."""
    thr = thr or thresholds()
    a = result["angles"]
    trunk, knee = a.get("trunk", 0.0), a.get("knee", 0.0)
    if a.get("arm_elevation", 0.0) >= thr["overhead_deg"]:
        return "overhead_reach"
    if load_score > 0 and trunk > thr["bend_deg"]:
        return "risky_lift"
    if knee > thr["kneel_deg"]:
        return "kneel_squat"
    if trunk > thr["bend_deg"]:
        return "back_bend"
    if a.get("neck", 0.0) > thr["neck_deg"]:
        return "neck"
    return None


def sound(category):
    """Pushover sound for a posture type; override with e.g. SOUND_BACK_BEND=bugle in .env."""
    return config.text("SOUND_" + category.upper(), CATEGORIES[category]["sound"])


def label(category):
    return CATEGORIES[category]["label"] if category in CATEGORIES else "Risky posture"
