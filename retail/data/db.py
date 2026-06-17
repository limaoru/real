"""SQLite 持久化：事件、漏斗、分时、快照"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from retail.paths import DB_PATH


class StoreDB:
    def __init__(self, path: Path = DB_PATH):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        c = self.conn.cursor()
        c.executescript("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            cam TEXT,
            event_type TEXT,
            message TEXT,
            global_id INTEGER,
            track_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS funnel_daily (
            date TEXT, stage TEXT, count INTEGER DEFAULT 0,
            PRIMARY KEY (date, stage)
        );
        CREATE TABLE IF NOT EXISTS dwell_tiers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, cam TEXT, zone TEXT, tier TEXT, track_id INTEGER, global_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS hourly (
            date TEXT, hour TEXT, cam TEXT, count INTEGER DEFAULT 0,
            PRIMARY KEY (date, hour, cam)
        );
        CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, cam TEXT, persons INTEGER, line_in INTEGER, line_out INTEGER,
            queue INTEGER, groups INTEGER, bags INTEGER, peak INTEGER
        );
        CREATE TABLE IF NOT EXISTS cross_cam (
            global_id INTEGER PRIMARY KEY,
            first_cam TEXT, last_cam TEXT, first_ts TEXT, last_ts TEXT, visit_count INTEGER DEFAULT 1
        );
        CREATE TABLE IF NOT EXISTS behaviors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, cam TEXT, behavior TEXT, track_id INTEGER, global_id INTEGER
        );
        CREATE TABLE IF NOT EXISTS vlm_insights (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            insight TEXT
        );
        CREATE TABLE IF NOT EXISTS open_vocab_hits (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, cam TEXT, label TEXT, conf REAL
        );
        CREATE TABLE IF NOT EXISTS ocr_reads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT, cam TEXT, text TEXT
        );
        """)
        self.conn.commit()

    def log_event(self, cam: str, event_type: str, message: str,
                  global_id: Optional[int] = None, track_id: Optional[int] = None):
        self.conn.execute(
            "INSERT INTO events (ts, cam, event_type, message, global_id, track_id) VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), cam, event_type, message, global_id, track_id),
        )
        self.conn.commit()

    def bump_funnel(self, stage: str):
        d = datetime.now().strftime("%Y-%m-%d")
        self.conn.execute(
            "INSERT INTO funnel_daily (date, stage, count) VALUES (?,?,1) "
            "ON CONFLICT(date,stage) DO UPDATE SET count=count+1",
            (d, stage),
        )
        self.conn.commit()

    def log_dwell_tier(self, cam: str, zone: str, tier: str, track_id: int, global_id: Optional[int]):
        self.conn.execute(
            "INSERT INTO dwell_tiers (ts, cam, zone, tier, track_id, global_id) VALUES (?,?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), cam, zone, tier, track_id, global_id),
        )
        self.conn.commit()

    def bump_hourly(self, cam: str):
        d = datetime.now().strftime("%Y-%m-%d")
        h = datetime.now().strftime("%H:00")
        self.conn.execute(
            "INSERT INTO hourly (date, hour, cam, count) VALUES (?,?,?,1) "
            "ON CONFLICT(date, hour, cam) DO UPDATE SET count=count+1",
            (d, h, cam),
        )
        self.conn.commit()

    def log_snapshot(self, cam: str, m: dict):
        self.conn.execute(
            "INSERT INTO snapshots (ts, cam, persons, line_in, line_out, queue, groups, bags, peak) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (
                datetime.now().isoformat(timespec="seconds"), cam,
                m.get("Person", 0), m.get("line_in", 0), m.get("line_out", 0),
                m.get("queue", 0), m.get("groups", 0), m.get("bags", 0), m.get("peak", 0),
            ),
        )
        self.conn.commit()

    def upsert_cross_cam(self, gid: int, cam: str):
        ts = datetime.now().isoformat(timespec="seconds")
        row = self.conn.execute("SELECT global_id FROM cross_cam WHERE global_id=?", (gid,)).fetchone()
        if row:
            self.conn.execute(
                "UPDATE cross_cam SET last_cam=?, last_ts=?, visit_count=visit_count+1 WHERE global_id=?",
                (cam, ts, gid),
            )
        else:
            self.conn.execute(
                "INSERT INTO cross_cam (global_id, first_cam, last_cam, first_ts, last_ts) VALUES (?,?,?,?,?)",
                (gid, cam, cam, ts, ts),
            )
        self.conn.commit()

    def get_funnel_today(self) -> dict:
        d = datetime.now().strftime("%Y-%m-%d")
        rows = self.conn.execute("SELECT stage, count FROM funnel_daily WHERE date=?", (d,)).fetchall()
        return {r[0]: r[1] for r in rows}

    def get_recent_events(self, limit: int = 30) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, cam, event_type, message FROM events ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"ts": r[0], "cam": r[1], "type": r[2], "message": r[3]} for r in rows]

    def get_hourly_today(self, cam: Optional[str] = None) -> list[dict]:
        d = datetime.now().strftime("%Y-%m-%d")
        if cam:
            rows = self.conn.execute(
                "SELECT hour, count FROM hourly WHERE date=? AND cam=? ORDER BY hour", (d, cam)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT hour, SUM(count) FROM hourly WHERE date=? GROUP BY hour ORDER BY hour", (d,)
            ).fetchall()
        return [{"hour": r[0], "count": r[1]} for r in rows]

    def get_cross_cam_summary(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT global_id, first_cam, last_cam, visit_count, last_ts FROM cross_cam ORDER BY last_ts DESC LIMIT 20"
        ).fetchall()
        return [{"gid": r[0], "first": r[1], "last": r[2], "visits": r[3], "last_ts": r[4]} for r in rows]

    def log_behavior(self, cam: str, behavior: str, track_id: int, global_id: Optional[int] = None):
        self.conn.execute(
            "INSERT INTO behaviors (ts, cam, behavior, track_id, global_id) VALUES (?,?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), cam, behavior, track_id, global_id),
        )
        self.conn.commit()

    def log_vlm_insight(self, insight: str):
        self.conn.execute(
            "INSERT INTO vlm_insights (ts, insight) VALUES (?,?)",
            (datetime.now().isoformat(timespec="seconds"), insight),
        )
        self.conn.commit()

    def log_open_vocab(self, cam: str, label: str, conf: float):
        self.conn.execute(
            "INSERT INTO open_vocab_hits (ts, cam, label, conf) VALUES (?,?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), cam, label, conf),
        )
        self.conn.commit()

    def log_ocr(self, cam: str, text: str):
        self.conn.execute(
            "INSERT INTO ocr_reads (ts, cam, text) VALUES (?,?,?)",
            (datetime.now().isoformat(timespec="seconds"), cam, text),
        )
        self.conn.commit()

    def get_recent_behaviors(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT ts, cam, behavior, track_id FROM behaviors ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"ts": r[0], "cam": r[1], "behavior": r[2], "track_id": r[3]} for r in rows]

    def get_latest_vlm(self) -> Optional[str]:
        row = self.conn.execute("SELECT insight FROM vlm_insights ORDER BY id DESC LIMIT 1").fetchone()
        return row[0] if row else None
