"""单路摄像头区域、越线与轨迹分析。"""
from __future__ import annotations

import csv
import time
from collections import defaultdict, deque
from datetime import datetime

import cv2
import numpy as np

from retail.config.cameras import get_cam_lines, get_cam_zones
from retail.config.settings import (
    ENABLE_CSV_LOG,
    ENABLE_HEATMAP_SNAPSHOT,
    ENABLE_LINE_CROSSING,
    ENABLE_LOITER_ALERT,
    ENABLE_TRAILS,
    ENABLE_ZONES,
    LOG_INTERVAL_SEC,
    LOITER_SECONDS,
    SNAPSHOT_INTERVAL_SEC,
    TRAIL_LENGTH,
)
from retail.geometry import detect_line_cross, point_in_polygon
from retail.paths import LOG_DIR
from retail.ui.text import queue_text


class CameraAnalytics:
    """单路摄像头店铺分析状态机。"""

    def __init__(self, cam_name: str, width: int, height: int):
        self.cam_name = cam_name
        self.width = width
        self.height = height
        self.trails: dict = {}
        self.tracks: dict = {}
        self.zone_enter_total = defaultdict(int)
        self.zone_current = defaultdict(int)
        self.line_in = 0
        self.line_out = 0
        self.alerts: deque = deque(maxlen=5)
        self.hourly = defaultdict(int)
        self.peak_occupancy = 0
        self.loiter_ids: set = set()
        self.lying_alerted: set = set()
        self.queue_alert_active = False
        self.last_log_t = time.time()
        self.last_snapshot_t = time.time()
        self.session_start = datetime.now()
        self.zones = get_cam_zones(cam_name, width, height)
        self.lines = get_cam_lines(cam_name, width, height)

    def push_alert(self, msg: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self.alerts.appendleft(f"[{ts}] {msg}")

    def update_person(self, track_id: int, foot_xy, pose_status: str = "未知") -> None:
        now = time.time()
        cx, cy = int(foot_xy[0]), int(foot_xy[1])
        info = self.tracks.get(track_id)
        if info is None:
            info = {"first": now, "last": now, "prev": None, "zones": set(), "pose": pose_status}
            self.tracks[track_id] = info
        else:
            info["prev"] = info.get("curr")
            info["last"] = now
            info["pose"] = pose_status
        info["curr"] = (cx, cy)

        if ENABLE_TRAILS:
            trail = self.trails.setdefault(track_id, deque(maxlen=TRAIL_LENGTH))
            trail.append((cx, cy))

        if ENABLE_LINE_CROSSING:
            for line in self.lines:
                cross = detect_line_cross(info.get("prev"), (cx, cy), line["p1"], line["p2"])
                if cross == "in":
                    self.line_in += 1
                    self.push_alert(f"{line['name']} 进店 +1")
                elif cross == "out":
                    self.line_out += 1
                    self.push_alert(f"{line['name']} 离店 +1")

        if ENABLE_ZONES:
            for zone in self.zones:
                inside = point_in_polygon(cx, cy, zone["poly"])
                zname = zone["name"]
                if inside and zname not in info["zones"]:
                    info["zones"].add(zname)
                    self.zone_enter_total[zname] += 1
                if not inside and zname in info["zones"]:
                    info["zones"].discard(zname)

        dwell = now - info["first"]
        if ENABLE_LOITER_ALERT and dwell >= LOITER_SECONDS and track_id not in self.loiter_ids:
            self.loiter_ids.add(track_id)
            self.push_alert(f"徘徊告警 ID:{track_id} 停留{dwell:.0f}s")
        if pose_status == "躺" and track_id not in self.lying_alerted:
            self.lying_alerted.add(track_id)
            self.push_alert(f"跌倒/躺卧 ID:{track_id}")

    def refresh_zone_occupancy(self) -> None:
        self.zone_current = defaultdict(int)
        for info in self.tracks.values():
            for z in info.get("zones", []):
                self.zone_current[z] += 1

    def prune_stale_tracks(self, max_idle: float = 8.0) -> None:
        now = time.time()
        stale = [tid for tid, info in self.tracks.items() if now - info["last"] > max_idle]
        for tid in stale:
            self.tracks.pop(tid, None)
            self.trails.pop(tid, None)

    def draw_zones(self, frame: np.ndarray) -> None:
        if not ENABLE_ZONES:
            return
        for zone in self.zones:
            pts = np.array(zone["poly"], dtype=np.int32).reshape(-1, 1, 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], zone["color"])
            cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
            cv2.polylines(frame, [pts], True, zone["color"], 2, cv2.LINE_AA)
            cx = int(np.mean([p[0] for p in zone["poly"]]))
            cy = int(np.mean([p[1] for p in zone["poly"]]))
            queue_text(
                frame,
                f"{zone['name']}:{self.zone_current.get(zone['name'], 0)}",
                cx - 40,
                cy,
                zone["color"],
                18,
            )

    def draw_lines(self, frame: np.ndarray) -> None:
        if not ENABLE_LINE_CROSSING:
            return
        for line in self.lines:
            cv2.line(frame, line["p1"], line["p2"], (0, 255, 255), 2, cv2.LINE_AA)
            mx = (line["p1"][0] + line["p2"][0]) // 2
            my = (line["p1"][1] + line["p2"][1]) // 2 - 8
            queue_text(frame, line["name"], mx - 30, my, (0, 255, 255), 18)

    def draw_trails(self, frame: np.ndarray) -> None:
        if not ENABLE_TRAILS:
            return
        for tid, trail in self.trails.items():
            if len(trail) < 2:
                continue
            pts = np.array(trail, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(frame, [pts], False, (255, 180, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, str(tid), trail[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1)

    def maybe_log_csv(self, metrics: dict) -> None:
        if not ENABLE_CSV_LOG or time.time() - self.last_log_t < LOG_INTERVAL_SEC:
            return
        self.last_log_t = time.time()
        LOG_DIR.mkdir(exist_ok=True)
        path = LOG_DIR / f"{self.cam_name}_{datetime.now():%Y%m%d}.csv"
        new_file = not path.exists()
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(
                    ["time", "cam", "persons", "in", "out", "standing", "sitting", "groups", "queue", "bags", "peak"]
                )
            w.writerow([
                datetime.now().isoformat(timespec="seconds"),
                self.cam_name,
                metrics.get("Person", 0),
                self.line_in,
                self.line_out,
                metrics.get("standing", 0),
                metrics.get("sitting", 0),
                metrics.get("groups", 0),
                metrics.get("queue", 0),
                metrics.get("bags", 0),
                self.peak_occupancy,
            ])

    def maybe_snapshot_heatmap(self, heatmap_panel: np.ndarray) -> None:
        if not ENABLE_HEATMAP_SNAPSHOT or time.time() - self.last_snapshot_t < SNAPSHOT_INTERVAL_SEC:
            return
        self.last_snapshot_t = time.time()
        snap_dir = LOG_DIR / "heatmaps"
        snap_dir.mkdir(parents=True, exist_ok=True)
        path = snap_dir / f"{self.cam_name}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
        cv2.imwrite(str(path), heatmap_panel)
