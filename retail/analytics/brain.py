"""高级分析：漏斗、ReID、停留分级、预测、可疑行为"""
import time
from collections import defaultdict, deque
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from retail.data.db import StoreDB
from retail.analytics.events import EventEngine
from retail.config.settings import (
    DWELL_TIER_BROWSE,
    DWELL_TIER_INTENT,
    DWELL_TIER_INTEREST,
    ENABLE_CONVERSION_TRACK,
    ENABLE_CROSS_CAM_REID,
    ENABLE_CROWD_DENSITY,
    ENABLE_DATABASE,
    ENABLE_DEEP_REID,
    ENABLE_DIRECTION_ARROWS,
    ENABLE_DWELL_TIERS,
    ENABLE_EVENT_ENGINE,
    ENABLE_FALL_DETECT,
    ENABLE_FAMILY_GROUP,
    ENABLE_FORECAST,
    ENABLE_FUNNEL,
    ENABLE_SHELF_ENGAGEMENT,
    ENABLE_SPEED_DETECT,
    ENABLE_STAFF_DETECT,
    ENABLE_STAFF_EXCLUDE_COUNT,
    ENABLE_SUSPICIOUS,
    ENABLE_ZIGZAG_DETECT,
    FAMILY_DISTANCE,
    FAMILY_MIN_FRAMES,
    SHELF_IDLE_SECONDS,
    SPEED_FAST_PX,
    STAFF_DWELL_SECONDS,
    STORE_AREA_SQM,
    ZIGZAG_REVERSALS,
)

from retail.analytics.attribution import AttributionEngine
from retail.analytics.forecast import AdvancedForecaster


def zone_to_funnel_stage(zone_name: str) -> Optional[str]:
    if any(k in zone_name for k in ("入口", "进店")):
        return "入口"
    if any(k in zone_name for k in ("货架", "体验", "等候")):
        return "浏览"
    if any(k in zone_name for k in ("收银",)):
        return "收银"
    return None


def is_shelf_zone(name: str) -> bool:
    return any(k in name for k in ("货架", "体验"))


def is_checkout_zone(name: str) -> bool:
    return any(k in name for k in ("收银", "等候"))


class SimpleReID:
    def __init__(self):
        self.next_gid = 1
        self.track_map: dict[tuple, int] = {}
        self.gallery: deque = deque(maxlen=200)

    def _feature(self, frame, box):
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return None
        crop = frame[y1:y2, x1:x2]
        small = cv2.resize(crop, (48, 96))
        hist = cv2.calcHist([small], [0, 1, 2], None, [6, 6, 6], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        area = (x2 - x1) * (y2 - y1)
        return hist.flatten().astype(np.float32), area

    def assign(self, cam: str, track_id: int, frame, box) -> Optional[int]:
        key = (cam, track_id)
        if key in self.track_map:
            return self.track_map[key]
        feat = self._feature(frame, box)
        if feat is None:
            return None
        hist, area = feat
        now = time.time()
        best_gid, best_score = None, 0.52
        for g in self.gallery:
            if now - g["time"] > 180 or g["cam"] == cam:
                continue
            score = cv2.compareHist(
                hist.reshape(-1, 1), g["hist"].reshape(-1, 1), cv2.HISTCMP_CORREL
            )
            ar = min(area, g["area"]) / max(area, g["area"])
            if score > best_score and ar > 0.35:
                best_score, best_gid = score, g["gid"]
        gid = best_gid or self.next_gid
        if best_gid is None:
            self.next_gid += 1
        self.track_map[key] = gid
        self.gallery.append({"gid": gid, "cam": cam, "hist": hist, "area": area, "time": now})
        return gid


class StoreBrain:
    """店铺智能分析中枢"""

    def __init__(self):
        self.db = StoreDB() if ENABLE_DATABASE else None
        self.events = EventEngine() if ENABLE_EVENT_ENGINE else None
        if ENABLE_CROSS_CAM_REID:
            if ENABLE_DEEP_REID:
                from retail.vision.advanced import DeepReID
                self.reid = DeepReID()
            else:
                self.reid = SimpleReID()
        else:
            self.reid = None
        self.forecaster = AdvancedForecaster()
        self.attribution = AttributionEngine()
        self.funnel_journey: dict[tuple, set] = defaultdict(set)
        self.funnel_completed: set[tuple] = set()
        self.dwell_zone_time: dict[tuple, float] = defaultdict(float)
        self.staff_dwell: dict[tuple, float] = defaultdict(float)
        self.dwell_tier_hit: set[tuple] = set()
        self.dwell_tier_counts = {"浏览": 0, "兴趣": 0, "意向": 0}
        self.staff_ids: set[int] = set()
        self.suspicious_count = 0
        self.suspicious_alerted: set[int] = set()
        self.fast_movers = 0
        self.cross_cam_new = 0
        self._cross_announced: set[tuple] = set()
        self.last_positions: dict[tuple, tuple] = {}
        self.shelf_enter_t: dict[int, float] = {}
        self.global_summary = {}
        self._last_db_snap = 0.0
        self.fall_count = 0
        self.fall_alerted: set[int] = set()
        self.zigzag_count = 0
        self.move_history: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=8))
        self.family_pairs: dict[tuple, int] = defaultdict(int)
        self.family_group_count = 0
        self.shelf_engagement = 0
        self.frame_zigzag = 0
        self.frame_falls = 0
        self.direction_vectors: dict[tuple, tuple] = {}
        self.behavior_counts: dict[str, int] = {}

    def record_behavior(self, cam: str, track_id: int, behavior: str, gid: Optional[int] = None):
        self.behavior_counts[behavior] = self.behavior_counts.get(behavior, 0) + 1
        if self.db:
            self.db.log_behavior(cam, behavior, track_id, gid)

    def process_person(
        self, cam: str, track_id: int, box, foot, frame, zones_inside: list[str], pose: str
    ) -> dict:
        info = {"gid": None, "is_staff": False, "funnel_stage": None}
        gid = None
        if self.reid:
            gid = self.reid.assign(cam, track_id, frame, box)
            info["gid"] = gid
            if gid:
                cams = {g["cam"] for g in self.reid.gallery if g["gid"] == gid}
                if len(cams) > 1 and (gid, cam) not in self._cross_announced:
                    self._cross_announced.add((gid, cam))
                    self.cross_cam_new += 1
                if self.db:
                    self.db.upsert_cross_cam(gid, cam)

        # 漏斗
        if ENABLE_FUNNEL:
            for zname in zones_inside:
                stage = zone_to_funnel_stage(zname)
                if not stage:
                    continue
                key = (cam, track_id)
                self.funnel_journey[key].add(stage)
                journey = self.funnel_journey[key]
                if "入口" in journey and "浏览" in journey and "收银" in journey:
                    if key not in self.funnel_completed:
                        self.funnel_completed.add(key)
                        if self.db:
                            self.db.bump_funnel("完整动线")
                if stage and self.db:
                    self.db.bump_funnel(stage)
                info["funnel_stage"] = stage

        # 停留分级
        if ENABLE_DWELL_TIERS:
            for zname in zones_inside:
                zkey = (cam, track_id, zname)
                self.dwell_zone_time[zkey] += 1.0  # 每帧约1单位，按处理帧率粗略累计
                t = self.dwell_zone_time[zkey]
                tier = None
                if t >= DWELL_TIER_INTENT:
                    tier = "意向"
                elif t >= DWELL_TIER_INTEREST:
                    tier = "兴趣"
                elif t >= DWELL_TIER_BROWSE:
                    tier = "浏览"
                if tier and zkey not in self.dwell_tier_hit:
                    self.dwell_tier_hit.add(zkey)
                    self.dwell_tier_counts[tier] += 1
                    if self.db:
                        self.db.log_dwell_tier(cam, zname, tier, track_id, gid)

        # 速度
        if ENABLE_SPEED_DETECT:
            pkey = (cam, track_id)
            prev = self.last_positions.get(pkey)
            if prev:
                dist = ((foot[0] - prev[0]) ** 2 + (foot[1] - prev[1]) ** 2) ** 0.5
                if dist > SPEED_FAST_PX:
                    self.fast_movers += 1
            self.last_positions[pkey] = foot

        # 店员启发式
        if ENABLE_STAFF_DETECT:
            in_checkout = any(is_checkout_zone(z) for z in zones_inside)
            if in_checkout and pose == "站":
                skey = (cam, track_id)
                self.staff_dwell[skey] += 1
                if self.staff_dwell[skey] > STAFF_DWELL_SECONDS / 3:
                    self.staff_ids.add(track_id)
            info["is_staff"] = track_id in self.staff_ids

        # 可疑：货架区久站不动
        if ENABLE_SUSPICIOUS:
            in_shelf = any(is_shelf_zone(z) for z in zones_inside)
            if in_shelf:
                if track_id not in self.shelf_enter_t:
                    self.shelf_enter_t[track_id] = time.time()
                elif time.time() - self.shelf_enter_t[track_id] > SHELF_IDLE_SECONDS:
                    if track_id not in self.suspicious_alerted:
                        self.suspicious_alerted.add(track_id)
                        self.suspicious_count += 1
            else:
                self.shelf_enter_t.pop(track_id, None)

        if ENABLE_FALL_DETECT and pose == "躺" and track_id not in self.fall_alerted:
            self.fall_alerted.add(track_id)
            self.fall_count += 1
            self.frame_falls += 1
            if self.db:
                self.db.log_event(cam, "fall", f"跌倒/躺卧 ID:{track_id}", gid, track_id)

        if ENABLE_ZIGZAG_DETECT:
            pkey = (cam, track_id)
            hist = self.move_history[pkey]
            hist.append(foot)
            if len(hist) >= 4:
                reversals = 0
                for i in range(2, len(hist)):
                    v1 = (hist[i - 1][0] - hist[i - 2][0], hist[i - 1][1] - hist[i - 2][1])
                    v2 = (hist[i][0] - hist[i - 1][0], hist[i][1] - hist[i - 1][1])
                    dot = v1[0] * v2[0] + v1[1] * v2[1]
                    if dot < -400:
                        reversals += 1
                if reversals >= ZIGZAG_REVERSALS and track_id not in self.suspicious_alerted:
                    self.zigzag_count += 1
                    self.frame_zigzag += 1

        if ENABLE_DIRECTION_ARROWS:
            pkey = (cam, track_id)
            prev = self.last_positions.get(pkey)
            if prev:
                self.direction_vectors[pkey] = (foot[0] - prev[0], foot[1] - prev[1])

        return info

    def process_frame_end(self, cam: str, track_positions: dict[int, tuple], lying_ids: set[int]):
        """每帧结束后：同行组、货架参与度"""
        if ENABLE_FAMILY_GROUP and len(track_positions) >= 2:
            ids = list(track_positions.keys())
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    a, b = track_positions[ids[i]], track_positions[ids[j]]
                    dist = ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
                    if dist < FAMILY_DISTANCE:
                        pair = tuple(sorted((ids[i], ids[j])))
                        self.family_pairs[pair] += 1
                        if self.family_pairs[pair] == FAMILY_MIN_FRAMES:
                            self.family_group_count += 1

        if ENABLE_SHELF_ENGAGEMENT:
            self.shelf_engagement = sum(
                1 for tid in track_positions if tid in self.shelf_enter_t
            )

        if ENABLE_FALL_DETECT:
            for tid in lying_ids:
                if tid not in self.fall_alerted:
                    self.fall_alerted.add(tid)
                    self.frame_falls += 1

    def get_conversion_rate(self) -> float:
        if not ENABLE_CONVERSION_TRACK:
            return 0.0
        f = self.get_funnel_stats()
        entry, checkout = f.get("入口", 0), f.get("收银", 0)
        return round(checkout / entry * 100, 1) if entry else 0.0

    def is_staff(self, track_id: int) -> bool:
        return track_id in self.staff_ids

    def draw_direction_arrows(self, frame):
        if not ENABLE_DIRECTION_ARROWS:
            return
        for (cam, tid), (dx, dy) in self.direction_vectors.items():
            if abs(dx) + abs(dy) < 5:
                continue
            pos = self.last_positions.get((cam, tid))
            if not pos:
                continue
            end = (int(pos[0] + dx * 2), int(pos[1] + dy * 2))
            cv2.arrowedLine(frame, pos, end, (200, 255, 100), 2, tipLength=0.35)

    def build_event_context(self, cam: str, metrics: dict, analytics) -> dict:
        checkout_p = 0
        for zname, cnt in analytics.zone_current.items():
            if is_checkout_zone(zname):
                checkout_p += cnt
        funnel_stages = self.get_funnel_stats()
        conv = self.get_conversion_rate()
        ctx = {
            "persons": metrics.get("Person", 0),
            "queue": metrics.get("queue", 0),
            "checkout_persons": checkout_p,
            "dwell_intent": self.dwell_tier_counts.get("意向", 0),
            "suspicious": self.suspicious_count,
            "fast_movers": self.fast_movers,
            "cross_cam_new": self.cross_cam_new,
            "funnel_entry": funnel_stages.get("入口", 0),
            "funnel_checkout": funnel_stages.get("收银", 0),
            "falls": self.frame_falls,
            "family_groups": self.family_group_count,
            "zigzag": self.frame_zigzag,
            "conversion_pct": conv,
        }
        return ctx

    def run_events(self, cam: str, metrics: dict, analytics, push_alert_fn):
        if not self.events:
            return
        ctx = self.build_event_context(cam, metrics, analytics)
        for msg in self.events.evaluate(ctx):
            push_alert_fn(msg)
        self.fast_movers = 0
        self.cross_cam_new = 0
        self.frame_zigzag = 0
        self.frame_falls = 0

    def get_funnel_stats(self) -> dict:
        stats = defaultdict(int)
        for journey in self.funnel_journey.values():
            for s in journey:
                stats[s] += 1
        stats["完整动线"] = len(self.funnel_completed)
        if self.db and ENABLE_DATABASE:
            for k, v in self.db.get_funnel_today().items():
                stats[k] = max(stats[k], v)
        return dict(stats)

    def forecast_next_hour(self, hourly: dict) -> int:
        if not ENABLE_FORECAST or not hourly:
            return 0
        return self.forecaster.predict_next_hour(hourly)

    def crowd_density(self, persons: int) -> float:
        if not ENABLE_CROWD_DENSITY or STORE_AREA_SQM <= 0:
            return 0.0
        return round(persons / STORE_AREA_SQM, 3)

    def get_cam_summary(self, cam: str, metrics: dict, analytics) -> dict:
        funnel = self.get_funnel_stats()
        density = self.crowd_density(metrics.get("Person", 0))
        forecast = self.forecast_next_hour(dict(analytics.hourly))
        return {
            "funnel": funnel,
            "dwell_tiers": dict(self.dwell_tier_counts),
            "density": density,
            "forecast_next_hour": forecast,
            "staff_count": len(self.staff_ids),
            "suspicious": self.suspicious_count,
            "cross_cam": self.db.get_cross_cam_summary() if self.db else [],
            "conversion_pct": self.get_conversion_rate(),
            "family_groups": self.family_group_count,
            "falls": self.fall_count,
            "zigzag": self.zigzag_count,
            "shelf_engagement": self.shelf_engagement,
            "behaviors": dict(self.behavior_counts),
            "attribution": self.attribution.get_insights(),
            "forecast_rest_day": self.forecaster.predict_rest_of_day(dict(analytics.hourly)),
        }

    def maybe_persist(self, cam: str, metrics: dict, analytics):
        if not self.db or time.time() - self._last_db_snap < 15:
            return
        self._last_db_snap = time.time()
        snap = dict(metrics)
        snap["line_in"] = analytics.line_in
        snap["line_out"] = analytics.line_out
        snap["peak"] = analytics.peak_occupancy
        self.db.log_snapshot(cam, snap)

    def push_attribution(self, metrics: dict, conversion_pct: float):
        m = dict(metrics)
        m["conversion_pct"] = conversion_pct
        self.attribution.push(m)

    def get_global_export(self) -> dict:
        from retail.config import settings as cfg

        return {
            "store_name": cfg.STORE_NAME,
            "funnel": self.get_funnel_stats(),
            "dwell_tiers": dict(self.dwell_tier_counts),
            "store_area_sqm": STORE_AREA_SQM,
            "conversion_pct": self.get_conversion_rate(),
            "family_groups": self.family_group_count,
            "falls": self.fall_count,
            "zigzag": self.zigzag_count,
            "behaviors": dict(self.behavior_counts),
            "attribution": self.attribution.export(),
            "features": {
                "boss_all_in": cfg.BOSS_ALL_IN,
                "funnel": ENABLE_FUNNEL,
                "reid": ENABLE_CROSS_CAM_REID,
                "deep_reid": ENABLE_DEEP_REID,
                "events": ENABLE_EVENT_ENGINE,
                "forecast": ENABLE_FORECAST,
                "fall": ENABLE_FALL_DETECT,
                "family": ENABLE_FAMILY_GROUP,
                "zigzag": ENABLE_ZIGZAG_DETECT,
                "webhook": cfg.ENABLE_WEBHOOK,
                "vlm": cfg.ENABLE_VLM_ANALYSIS,
                "open_vocab": cfg.ENABLE_OPEN_VOCAB,
                "ocr": cfg.ENABLE_OCR_SHELF,
                "event_clips": cfg.ENABLE_EVENT_CLIPS,
                "digital_twin": cfg.ENABLE_DIGITAL_TWIN,
                "multiview": cfg.ENABLE_MULTIVIEW_FUSION,
                "multistore": cfg.ENABLE_MULTISTORE,
                "face_blur": cfg.ENABLE_FACE_BLUR,
                "active_learning": cfg.ENABLE_ACTIVE_LEARNING,
            },
        }
