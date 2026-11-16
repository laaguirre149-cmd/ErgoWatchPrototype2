"""SQLite storage shared by the camera app (run.py) and the web app (web_app.py).
Stores timestamps, worker IDs, scores and seconds only. Never video or images.

Tables
  workers   id, name, focus (comma-separated posture types), signature (JSON, for gait matching)
  checkins  ts, worker_id, bay, kind ('in' / 'out'): who is at which camera/bay
  samples   one row per worker per few seconds: seconds observed, seconds high-risk, per-region
            high-risk seconds and average REBA component scores
  events    one row per risky episode / static hold / rep set / alert, with its posture type
"""
import datetime
import json
import os
import queue
import sqlite3
import threading
import time

from pipeline.exposure import REGIONS

UNASSIGNED = "unassigned"

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS workers (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, focus TEXT NOT NULL DEFAULT '',
    signature TEXT, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS checkins (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, worker_id TEXT NOT NULL,
    bay TEXT NOT NULL DEFAULT '1', kind TEXT NOT NULL DEFAULT 'in');
CREATE TABLE IF NOT EXISTS samples (
    ts REAL NOT NULL, worker_id TEXT NOT NULL, duration REAL NOT NULL, risky_s REAL NOT NULL,
    reba_avg REAL NOT NULL, reba_max INTEGER NOT NULL,
    {", ".join(f"{r}_s REAL NOT NULL DEFAULT 0, {r}_avg REAL NOT NULL DEFAULT 1" for r in REGIONS)});
CREATE INDEX IF NOT EXISTS samples_worker_ts ON samples (worker_id, ts);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, worker_id TEXT NOT NULL,
    kind TEXT NOT NULL, category TEXT NOT NULL DEFAULT '', reba INTEGER NOT NULL DEFAULT 0,
    duration REAL NOT NULL DEFAULT 0, reps INTEGER NOT NULL DEFAULT 0, alerted INTEGER NOT NULL DEFAULT 0);
CREATE INDEX IF NOT EXISTS events_worker_ts ON events (worker_id, ts);
"""

SAMPLE_COLS = ["ts", "worker_id", "duration", "risky_s", "reba_avg", "reba_max"] + \
              [f"{r}_{s}" for r in REGIONS for s in ("s", "avg")]
EVENT_COLS = ["ts", "worker_id", "kind", "category", "reba", "duration", "reps", "alerted"]


def day_start(ts=None):
    """Timestamp of local midnight for the day containing ts (a shift = one calendar day)."""
    d = datetime.datetime.fromtimestamp(ts if ts is not None else time.time())
    return d.replace(hour=0, minute=0, second=0, microsecond=0).timestamp()


def week_start(ts=None):
    """Timestamp of local midnight on the Monday of the week containing ts."""
    d = datetime.datetime.fromtimestamp(day_start(ts))
    return (d - datetime.timedelta(days=d.weekday())).timestamp()


class Store:
    """Each call opens its own short connection, so it is safe from any thread or process."""

    def __init__(self, path):
        self.path = path
        folder = os.path.dirname(path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.executescript(SCHEMA)

    def _conn(self):
        c = sqlite3.connect(self.path, timeout=5)
        c.row_factory = sqlite3.Row
        return c

    def _run(self, sql, args=()):
        c = self._conn()
        try:
            with c:
                return c.execute(sql, args).fetchall()
        finally:
            c.close()

    # --- writing (camera app) ---
    def add_samples(self, rows):
        c = self._conn()
        try:
            with c:
                c.executemany(f"INSERT INTO samples ({', '.join(SAMPLE_COLS)}) VALUES "
                              f"({', '.join('?' * len(SAMPLE_COLS))})",
                              [[r[k] for k in SAMPLE_COLS] for r in rows])
        finally:
            c.close()

    def add_event(self, ts, worker_id, kind, category="", reba=0, duration=0.0, reps=0, alerted=0):
        self._run(f"INSERT INTO events ({', '.join(EVENT_COLS)}) VALUES (?,?,?,?,?,?,?,?)",
                  (ts, worker_id, kind, category, int(reba), float(duration), int(reps), int(alerted)))

    # --- workers and check-in ---
    def upsert_worker(self, worker_id, name=None):
        self._run("INSERT INTO workers (id, name, created) VALUES (?,?,?) ON CONFLICT(id) DO UPDATE "
                  "SET name = COALESCE(?, name)", (worker_id, name or worker_id, time.time(), name))

    def workers(self):
        return [dict(r) for r in self._run("SELECT * FROM workers ORDER BY name")]

    def worker(self, worker_id):
        rows = self._run("SELECT * FROM workers WHERE id = ?", (worker_id,))
        return dict(rows[0]) if rows else None

    def set_focus(self, worker_id, categories):
        self._run("UPDATE workers SET focus = ? WHERE id = ?", (",".join(categories[:3]), worker_id))

    def focus(self, worker_id):
        w = self.worker(worker_id)
        return [c for c in (w["focus"].split(",") if w else []) if c]

    def set_signature(self, worker_id, signature):
        self._run("UPDATE workers SET signature = ? WHERE id = ?",
                  (json.dumps(signature) if signature else None, worker_id))

    def signatures(self):
        return {r["id"]: json.loads(r["signature"])
                for r in self._run("SELECT id, signature FROM workers WHERE signature IS NOT NULL")}

    def check_in(self, worker_id, bay="1", ts=None):
        self.upsert_worker(worker_id)
        self._run("INSERT INTO checkins (ts, worker_id, bay, kind) VALUES (?,?,?,'in')",
                  (ts if ts is not None else time.time(), worker_id, str(bay)))

    def check_out(self, bay="1", ts=None):
        self._run("INSERT INTO checkins (ts, worker_id, bay, kind) VALUES (?,'',?,'out')",
                  (ts if ts is not None else time.time(), str(bay)))

    def active_checkin(self, bay="1", now=None):
        """The latest check-in at this bay today, as a dict (worker_id, ts), or None."""
        now = now if now is not None else time.time()
        rows = self._run("SELECT * FROM checkins WHERE bay = ? AND ts >= ? AND ts <= ? "
                         "ORDER BY ts DESC, id DESC LIMIT 1", (str(bay), day_start(now), now))
        if rows and rows[0]["kind"] == "in":
            return dict(rows[0])
        return None

    def active_worker(self, bay="1", now=None):
        c = self.active_checkin(bay, now)
        return c["worker_id"] if c else None

    # --- reading (web app) ---
    def exposure(self, worker_id, t0, t1):
        """Totals between t0 and t1: seconds observed, seconds high-risk, seconds per region."""
        sums = ", ".join(f"COALESCE(SUM({r}_s), 0) AS {r}" for r in REGIONS)
        row = self._run(f"SELECT COALESCE(SUM(duration), 0) AS observed, COALESCE(SUM(risky_s), 0) AS risky, "
                        f"COALESCE(MAX(reba_max), 0) AS reba_max, {sums} FROM samples "
                        f"WHERE worker_id = ? AND ts >= ? AND ts < ?", (worker_id, t0, t1))[0]
        return {"observed": row["observed"], "risky": row["risky"], "reba_max": row["reba_max"],
                "regions": {r: row[r] for r in REGIONS}}

    def daily_risky(self, worker_id, t0, days=7):
        """High-risk seconds for each of `days` days starting at t0 (list of (day_ts, seconds))."""
        out = []
        for i in range(days):
            a = (datetime.datetime.fromtimestamp(t0) + datetime.timedelta(days=i)).timestamp()
            b = (datetime.datetime.fromtimestamp(t0) + datetime.timedelta(days=i + 1)).timestamp()
            out.append((a, self.exposure(worker_id, a, b)["risky"]))
        return out

    def top_moves(self, worker_id, t0, t1):
        """Posture types ranked by how often they happened (risky episodes) between t0 and t1."""
        rows = self._run("SELECT category, COUNT(*) AS n, SUM(duration) AS seconds, MAX(reba) AS reba "
                         "FROM events WHERE worker_id = ? AND kind = 'posture' AND category != '' "
                         "AND ts >= ? AND ts < ? GROUP BY category ORDER BY n DESC, seconds DESC",
                         (worker_id, t0, t1))
        return [dict(r) for r in rows]

    def events(self, worker_id, t0, t1, kinds=None, limit=50):
        sql = "SELECT * FROM events WHERE worker_id = ? AND ts >= ? AND ts < ?"
        args = [worker_id, t0, t1]
        if kinds:
            sql += f" AND kind IN ({', '.join('?' * len(kinds))})"
            args += list(kinds)
        return [dict(r) for r in self._run(sql + " ORDER BY ts DESC LIMIT ?", args + [limit])]

    def event_counts(self, worker_id, t0, t1):
        rows = self._run("SELECT kind, COUNT(*) AS n, COALESCE(SUM(reps), 0) AS reps, "
                         "COALESCE(SUM(alerted), 0) AS alerts FROM events "
                         "WHERE worker_id = ? AND ts >= ? AND ts < ? GROUP BY kind", (worker_id, t0, t1))
        return {r["kind"]: dict(r) for r in rows}

    def worker_ids_with_data(self):
        return [r[0] for r in self._run("SELECT DISTINCT worker_id FROM samples")]


class AsyncWriter:
    """Runs database writes on a background thread so the video loop never waits on disk."""

    def __init__(self, store):
        self.store = store
        self.q = queue.Queue()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while True:
            job = self.q.get()
            if job is None:
                return
            fn, args, kwargs = job
            try:
                getattr(self.store, fn)(*args, **kwargs)
            except Exception as e:  # never let a database problem stop the demo
                print(f"[ErgoWatch] database write failed ({fn}): {e}")

    def call(self, fn, *args, **kwargs):
        self.q.put((fn, args, kwargs))

    def close(self):
        self.q.put(None)
        self.thread.join(timeout=3)
