"""Reads settings from the environment (.env). Every new feature has an on/off flag here,
so anything that misbehaves during a demo can be switched off by adding e.g.
FEATURE_REPS=0 to .env (or in front of the command: FEATURE_REPS=0 python run.py)."""
import os


def flag(name, default=True):
    v = os.getenv(name, "").split("#")[0].strip().lower()
    if v == "":
        return default
    return v in ("1", "true", "yes", "on")


def num(name, default):
    v = os.getenv(name, "").split("#")[0].strip()
    try:
        return float(v) if v else float(default)
    except ValueError:
        return float(default)


def text(name, default=""):
    return os.getenv(name, default).split("#")[0].strip() or default


# Feature flags (all default on; set to 0 to switch one off)
def features():
    return {
        "exposure": flag("FEATURE_EXPOSURE"),          # 1. cumulative exposure in SQLite
        "reps": flag("FEATURE_REPS"),                  # 2. rep counting + static holds
        "posture_alerts": flag("FEATURE_POSTURE_ALERTS"),  # 3. distinct alert per posture type
        "object_weight": flag("FEATURE_OBJECT_WEIGHT"),    # 6. carried-object weight estimate
        "gait": flag("FEATURE_GAIT"),                  # 7. identify worker from body signature
    }


def db_path():
    return text("ERGOWATCH_DB", os.path.join("data", "ergowatch.db"))


def lan_ip():
    """This Mac's address on the local network (what a phone on the same WiFi should open)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))   # no data is sent; this only picks the outgoing interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def web_port():
    return int(num("WEB_PORT", 8000))


def web_url():
    """Base address of the web app, used for links inside phone alerts. Set WEB_URL to override."""
    return text("WEB_URL", f"http://{lan_ip()}:{web_port()}").rstrip("/")
