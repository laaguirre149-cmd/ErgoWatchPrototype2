import numpy as np

NOSE, L_EAR, R_EAR = 0, 3, 4
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 5, 6, 7, 8, 9, 10
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 13, 14, 15, 16
UP = np.array([0.0, -1.0])  # image y-axis points down

# Published REBA tables (Hignett & McAtamney, Applied Ergonomics, 2000)
TABLE_A = {  # TABLE_A[neck][trunk - 1][legs - 1]
    1: [[1, 2, 3, 4], [2, 3, 4, 5], [2, 4, 5, 6], [3, 5, 6, 7], [4, 6, 7, 8]],
    2: [[1, 2, 3, 4], [3, 4, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9]],
    3: [[3, 3, 5, 6], [4, 5, 6, 7], [5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 9]],
}
TABLE_B = {  # TABLE_B[lower_arm][upper_arm - 1][wrist - 1]
    1: [[1, 2, 2], [1, 2, 3], [3, 4, 5], [4, 5, 5], [6, 7, 8], [7, 8, 8]],
    2: [[1, 2, 3], [2, 3, 4], [4, 5, 5], [5, 6, 7], [7, 8, 8], [8, 9, 9]],
}
TABLE_C = [  # TABLE_C[score_a - 1][score_b - 1]
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 9],
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 10],
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],
    [8, 8, 8, 9, 10, 10, 10, 10, 10, 11, 11, 11],
    [9, 9, 9, 10, 10, 10, 11, 11, 11, 12, 12, 12],
    [10, 10, 10, 11, 11, 11, 11, 12, 12, 12, 12, 12],
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
]


def angle(v1, v2):
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def risk_level(score):
    if score <= 1:
        return "Negligible"
    if score <= 3:
        return "Low"
    if score <= 7:
        return "Medium"
    if score <= 10:
        return "High"
    return "Very high"


def reba(kps, conf, load_score=0, coupling_score=0, activity_score=0, thr=0.5, up=UP):
    """Full REBA score (1-15) for one person, or None if the core body isn't visible.
    Wrist, twisting and load can't be seen by the camera: wrist is assumed neutral,
    and load is set manually (LOAD_SCORE) per task or zone.
    kps is (17, 2) image points; (17, 3) points also work if `up` is the 3D up direction
    (used to score the Kinect skeletons when validating against UW-IOM)."""
    if min(conf[[L_SH, R_SH, L_HIP, R_HIP]]) < thr:
        return None
    sh = (kps[L_SH] + kps[R_SH]) / 2
    hip = (kps[L_HIP] + kps[R_HIP]) / 2
    spine = sh - hip
    angles = {}

    trunk = angle(spine, up)
    angles["trunk"] = trunk
    trunk_s = 1 if trunk <= 5 else 2 if trunk <= 20 else 3 if trunk <= 60 else 4

    neck_s = 1
    if min(conf[[L_EAR, R_EAR]]) >= thr:
        neck = angle((kps[L_EAR] + kps[R_EAR]) / 2 - sh, spine)
        angles["neck"] = neck
        neck_s = 1 if neck <= 20 else 2

    legs_s = 1
    knees = [180 - angle(kps[h] - kps[k], kps[a] - kps[k])
             for h, k, a in [(L_HIP, L_KNEE, L_ANK), (R_HIP, R_KNEE, R_ANK)]
             if min(conf[[h, k, a]]) >= thr]
    if knees:
        knee = max(knees)
        angles["knee"] = knee
        legs_s += 1 if 30 <= knee <= 60 else 2 if knee > 60 else 0

    ua_s, la_s = 1, 1
    arms = []
    for s, el, wr, h in [(L_SH, L_EL, L_WR, L_HIP), (R_SH, R_EL, R_WR, R_HIP)]:
        if min(conf[[s, el, h]]) < thr:
            continue
        ua = angle(kps[el] - kps[s], kps[h] - kps[s])
        ua_score = 1 if ua <= 20 else 2 if ua <= 45 else 3 if ua <= 90 else 4
        la, la_score = None, 1
        if conf[wr] >= thr:
            la = angle(kps[el] - kps[s], kps[wr] - kps[el])  # elbow flexion, 0 = straight
            la_score = 1 if 60 <= la <= 100 else 2
        arms.append((ua_score, la_score, ua, la, angle(kps[el] - kps[s], -up)))
    if arms:
        ua_s, la_s, ua, la, _ = max(arms, key=lambda a: (a[0], a[1]))
        angles["upper_arm"] = ua
        # Arm direction against gravity: 0 = hanging straight down, 90 = level, 180 = straight up
        angles["arm_elevation"] = max(a[4] for a in arms)
        if la is not None:
            angles["lower_arm"] = la
    wrist_s = 1

    score_a = TABLE_A[neck_s][trunk_s - 1][legs_s - 1] + load_score
    score_b = TABLE_B[la_s][ua_s - 1][wrist_s - 1] + coupling_score
    score_c = TABLE_C[min(score_a, 12) - 1][min(score_b, 12) - 1]
    final = min(score_c + activity_score, 15)

    # Body parts at REBA's maximum risk band
    flags = []
    if trunk_s == 4:
        flags.append("Trunk bent over 60 deg")
    if ua_s == 4:
        flags.append("Arm raised over 90 deg")

    return {
        "score": final,
        "risk": risk_level(final),
        "angles": angles,
        "flags": flags,
        "segments": {"trunk": trunk_s, "neck": neck_s, "upper_arm": ua_s,
                     "lower_arm": la_s, "legs": legs_s},
        "score_c": score_c, "activity": activity_score, "load": load_score,
    }


def with_activity(result, activity_score):
    """Add REBA's activity score (static hold / repetition, 0-2) to an already computed result."""
    result["activity"] = activity_score
    result["score"] = min(result["score_c"] + activity_score, 15)
    result["risk"] = risk_level(result["score"])
    return result