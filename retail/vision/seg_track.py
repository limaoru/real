"""Seg + ByteTrack：每 N 帧跑一次分割检测，中间帧用 Kalman 插值。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ultralytics.engine.results import Boxes


@dataclass
class TrackFrame:
    boxes: np.ndarray | None
    clss: np.ndarray | None
    track_ids: np.ndarray | None
    masks_xy: list | None
    conf: np.ndarray | None
    is_seg_frame: bool


class ByteTrackSegRunner:
    """每 ``every_n`` 帧 YOLO seg+track 一次，其余帧仅 ByteTrack 预测。"""

    def __init__(self, model, track_kw: dict, *, every_n: int = 3):
        self.model = model
        self.track_kw = {**track_kw, "persist": True}
        self.every_n = max(1, every_n)
        self._frame = 0
        self._mask_cache: dict[int, np.ndarray] = {}
        self._box_cache: dict[int, np.ndarray] = {}

    def _get_tracker(self):
        predictor = getattr(self.model, "predictor", None)
        if predictor is None or not getattr(predictor, "trackers", None):
            return None
        return predictor.trackers[0]

    def _shift_mask(self, track_id: int, new_box: np.ndarray) -> np.ndarray | None:
        mask = self._mask_cache.get(track_id)
        old_box = self._box_cache.get(track_id)
        if mask is None:
            return None
        if old_box is None:
            return mask
        dx = (new_box[0] + new_box[2]) / 2 - (old_box[0] + old_box[2]) / 2
        dy = (new_box[1] + new_box[3]) / 2 - (old_box[1] + old_box[3]) / 2
        if abs(dx) < 0.5 and abs(dy) < 0.5:
            return mask
        shifted = mask.copy()
        shifted[:, 0] += dx
        shifted[:, 1] += dy
        return shifted

    def _update_caches(
        self,
        track_ids: np.ndarray,
        clss: np.ndarray,
        boxes: np.ndarray,
        masks_xy: list | None,
    ) -> None:
        active: set[int] = set()
        if track_ids is None:
            return
        for idx, (tid, cls_id, box) in enumerate(zip(track_ids, clss, boxes)):
            if tid is None:
                continue
            tid = int(tid)
            active.add(tid)
            self._box_cache[tid] = box.copy()
            if cls_id == 0 and masks_xy is not None and idx < len(masks_xy):
                raw = masks_xy[idx]
                if raw is not None and len(raw) >= 3:
                    self._mask_cache[tid] = np.asarray(raw, dtype=np.float32)
        stale = [tid for tid in self._mask_cache if tid not in active]
        for tid in stale:
            self._mask_cache.pop(tid, None)
            self._box_cache.pop(tid, None)

    def _pack_yolo_result(self, r0, *, is_seg_frame: bool) -> TrackFrame:
        if r0.boxes is None or r0.boxes.data.numel() == 0:
            return TrackFrame(None, None, None, None, None, is_seg_frame)

        boxes = r0.boxes.xyxy.cpu().numpy()
        clss = r0.boxes.cls.cpu().numpy().astype(int)
        conf = r0.boxes.conf.cpu().numpy()
        if r0.boxes.id is not None:
            track_ids = r0.boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.array([None] * len(clss), dtype=object)

        masks_xy = r0.masks.xy if r0.masks is not None else None
        if is_seg_frame:
            self._update_caches(track_ids, clss, boxes, masks_xy)
        return TrackFrame(boxes, clss, track_ids, masks_xy, conf, is_seg_frame)

    def _pack_tracker_output(self, tracks: np.ndarray, h: int, w: int) -> TrackFrame:
        if tracks is None or len(tracks) == 0:
            return TrackFrame(None, None, None, None, None, False)

        boxes = tracks[:, :4].astype(np.float32)
        track_ids = tracks[:, 4].astype(int)
        conf = tracks[:, 5].astype(np.float32)
        clss = tracks[:, 6].astype(int)

        masks_xy = []
        for tid, box in zip(track_ids, boxes):
            self._box_cache[int(tid)] = box.copy()
            masks_xy.append(self._shift_mask(int(tid), box))

        return TrackFrame(boxes, clss, track_ids, masks_xy, conf, False)

    def run(self, frame: np.ndarray) -> TrackFrame:
        self._frame += 1
        run_seg = self._frame == 1 or (self._frame % self.every_n == 1)

        if run_seg or self._get_tracker() is None:
            results = self.model.track(source=frame, **self.track_kw)
            return self._pack_yolo_result(results[0], is_seg_frame=True)

        h, w = frame.shape[:2]
        empty = Boxes(np.zeros((0, 6), dtype=np.float32), orig_shape=(h, w))
        tracks = self._get_tracker().update(empty, frame)
        return self._pack_tracker_output(tracks, h, w)
