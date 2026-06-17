"""轮廓与骨架时序平滑，减轻闪烁。"""
from __future__ import annotations

import numpy as np

from retail.config.settings import OVERLAY_SMOOTH_ALPHA, SKELETON_HOLD_FRAMES


def _torso_center(kp: np.ndarray) -> tuple[float, float]:
    if kp.shape[0] > 12 and kp[11][1] > 0 and kp[12][1] > 0:
        return float((kp[11][0] + kp[12][0]) / 2), float((kp[11][1] + kp[12][1]) / 2)
    if kp[0][1] > 0:
        return float(kp[0][0]), float(kp[0][1])
    return -1.0, -1.0


def _point_in_box(px: float, py: float, box, margin: float = 0.08) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - w * margin) <= px <= (x2 + w * margin) and (y1 - h * margin) <= py <= (y2 + h * margin)


def match_poses_to_tracks(
    kps_xy: np.ndarray,
    person_boxes: dict[int, np.ndarray],
) -> dict[int, int]:
    """pose 索引 → track_id"""
    mapping: dict[int, int] = {}
    used: set[int] = set()
    for pi in range(len(kps_xy)):
        cx, cy = _torso_center(kps_xy[pi])
        if cx < 0:
            continue
        best_tid, best_score = None, -1.0
        for tid, box in person_boxes.items():
            if tid in used:
                continue
            if _point_in_box(cx, cy, box):
                score = 1000.0
            else:
                bx = (box[0] + box[2]) / 2
                by = box[3]
                dist = (cx - bx) ** 2 + (cy - by) ** 2
                score = -dist
            if score > best_score:
                best_score, best_tid = score, tid
        if best_tid is not None and best_score > -200**2:
            mapping[pi] = best_tid
            used.add(best_tid)
    return mapping


class OverlaySmoothCache:
    """按 (cam, track_id) 平滑 mask 与 keypoints。"""

    def __init__(self, alpha: float = OVERLAY_SMOOTH_ALPHA):
        self.alpha = alpha
        self._masks: dict[tuple, np.ndarray] = {}
        self._keypoints: dict[tuple, np.ndarray] = {}
        self._kp_conf: dict[tuple, np.ndarray] = {}
        self._mask_age: dict[tuple, int] = {}
        self._kp_age: dict[tuple, int] = {}

    def smooth_mask(self, cam: str, track_id: int, mask_xy) -> np.ndarray | None:
        key = (cam, track_id)
        if mask_xy is None or len(mask_xy) < 3:
            age = self._mask_age.get(key, SKELETON_HOLD_FRAMES + 1)
            if age <= SKELETON_HOLD_FRAMES and key in self._masks:
                self._mask_age[key] = age + 1
                return self._masks[key]
            return None

        pts = np.asarray(mask_xy, dtype=np.float32)
        if pts.ndim == 1:
            return None
        if key in self._masks and self._masks[key].shape == pts.shape:
            pts = self.alpha * pts + (1.0 - self.alpha) * self._masks[key]
        self._masks[key] = pts.copy()
        self._mask_age[key] = 0
        return pts

    def update_keypoints(
        self,
        cam: str,
        person_boxes: dict[int, np.ndarray],
        kps_xy: np.ndarray | None,
        kps_conf: np.ndarray | None,
    ) -> None:
        active = set()
        if kps_xy is not None and len(kps_xy) > 0 and person_boxes:
            mapping = match_poses_to_tracks(kps_xy, person_boxes)
            for pi, tid in mapping.items():
                key = (cam, tid)
                active.add(key)
                kp = kps_xy[pi].astype(np.float32)
                conf = kps_conf[pi].astype(np.float32) if kps_conf is not None else None
                if key in self._keypoints:
                    prev = self._keypoints[key]
                    vis = (kp[:, 0] > 0) & (kp[:, 1] > 0)
                    merged = prev.copy()
                    merged[vis] = self.alpha * kp[vis] + (1.0 - self.alpha) * prev[vis]
                    kp = merged
                    if conf is not None and key in self._kp_conf:
                        pc = self._kp_conf[key]
                        cvis = conf > 0.05
                        conf = np.where(cvis, self.alpha * conf + (1.0 - self.alpha) * pc, pc)
                self._keypoints[key] = kp
                if conf is not None:
                    self._kp_conf[key] = conf
                self._kp_age[key] = 0

        for key in list(self._keypoints.keys()):
            if not key[0] == cam:
                continue
            if key in active:
                continue
            age = self._kp_age.get(key, 0) + 1
            self._kp_age[key] = age
            if age > SKELETON_HOLD_FRAMES:
                self._keypoints.pop(key, None)
                self._kp_conf.pop(key, None)
                self._kp_age.pop(key, None)

    def prune_masks(self, cam: str, active_track_ids: set[int]) -> None:
        for key in list(self._masks.keys()):
            if key[0] != cam:
                continue
            if key[1] not in active_track_ids:
                age = self._mask_age.get(key, 0) + 1
                self._mask_age[key] = age
                if age > SKELETON_HOLD_FRAMES:
                    self._masks.pop(key, None)
                    self._mask_age.pop(key, None)
            elif key in self._masks:
                self._mask_age[key] = 0

    def get_keypoints(self, cam: str, track_id: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        key = (cam, track_id)
        return self._keypoints.get(key), self._kp_conf.get(key)

    def iter_track_keypoints(self, cam: str, track_ids: set[int]):
        for tid in track_ids:
            kp, conf = self.get_keypoints(cam, tid)
            if kp is not None:
                yield tid, kp, conf
