"""推理进程 IPC：TrackFrame / 姿态结果 序列化。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from retail.vision.seg_track import TrackFrame


@dataclass(frozen=True)
class InferJob:
    seq: int
    infer_frame: np.ndarray
    inv_scale: float
    run_pose: bool


@dataclass(frozen=True)
class InferResult:
    cam_name: str
    seq: int
    inv_scale: float
    track_frame: TrackFrame
    kps_xy_np: np.ndarray | None
    kps_conf_np: np.ndarray | None
    infer_ms: float


def track_frame_to_payload(tf: TrackFrame) -> dict:
    masks = tf.masks_xy
    if masks is not None:
        masks = [
            None if m is None else np.asarray(m, dtype=np.float32)
            for m in masks
        ]
    return {
        "boxes": None if tf.boxes is None else np.asarray(tf.boxes, dtype=np.float32),
        "clss": None if tf.clss is None else np.asarray(tf.clss),
        "track_ids": None if tf.track_ids is None else np.asarray(tf.track_ids),
        "masks_xy": masks,
        "conf": None if tf.conf is None else np.asarray(tf.conf, dtype=np.float32),
        "is_seg_frame": tf.is_seg_frame,
    }


def payload_to_track_frame(payload: dict) -> TrackFrame:
    masks = payload.get("masks_xy")
    if masks is not None:
        masks = [None if m is None else np.asarray(m) for m in masks]
    return TrackFrame(
        payload.get("boxes"),
        payload.get("clss"),
        payload.get("track_ids"),
        masks,
        payload.get("conf"),
        payload.get("is_seg_frame", False),
    )
