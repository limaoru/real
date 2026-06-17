"""姿态估计：骨架绘制与站/坐/躺分类。"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from retail.config.models import SKELETON


def draw_skeleton(
    frame: np.ndarray,
    keypoints_xy,
    keypoints_conf=None,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    conf_thresh: float = 0.5,
) -> None:
    if keypoints_xy is None:
        return
    kps = keypoints_xy.cpu().numpy() if hasattr(keypoints_xy, "cpu") else np.asarray(keypoints_xy)
    confs = None
    if keypoints_conf is not None:
        confs = (
            keypoints_conf.cpu().numpy()
            if hasattr(keypoints_conf, "cpu")
            else np.asarray(keypoints_conf)
        )

    for p_idx, kp in enumerate(kps):
        kp_conf = confs[p_idx] if confs is not None else None
        for i, j in SKELETON:
            if kp[i][0] <= 0 or kp[i][1] <= 0 or kp[j][0] <= 0 or kp[j][1] <= 0:
                continue
            if kp_conf is not None and (kp_conf[i] < conf_thresh or kp_conf[j] < conf_thresh):
                continue
            cv2.line(
                frame,
                (int(kp[i][0]), int(kp[i][1])),
                (int(kp[j][0]), int(kp[j][1])),
                color,
                thickness,
                cv2.LINE_AA,
            )
        for x, y in kp:
            if x > 0 and y > 0:
                cv2.circle(frame, (int(x), int(y)), 3, color, -1, cv2.LINE_AA)


def classify_pose(
    kp_xy: np.ndarray,
    kp_conf: Optional[np.ndarray] = None,
    conf_thresh: float = 0.35,
) -> str:
    def ok(i: int) -> bool:
        if kp_xy[i][0] <= 0 or kp_xy[i][1] <= 0:
            return False
        if kp_conf is not None and kp_conf[i] < conf_thresh:
            return False
        return True

    if not (ok(5) and ok(6) and ok(11) and ok(12)):
        return "Unknown"

    shoulder_y = float((kp_xy[5][1] + kp_xy[6][1]) / 2)
    hip_y = float((kp_xy[11][1] + kp_xy[12][1]) / 2)
    torso = hip_y - shoulder_y
    if torso <= 0:
        return "Unknown"

    shoulder_w = abs(float(kp_xy[5][0] - kp_xy[6][0])) if ok(5) and ok(6) else 0.0
    if shoulder_w > 0 and torso < shoulder_w * 0.35:
        return "Lying"

    if ok(15) and ok(16):
        ankle_y = float((kp_xy[15][1] + kp_xy[16][1]) / 2)
        leg = ankle_y - hip_y
        if leg > 0:
            ratio = torso / leg
            if ratio > 0.9:
                return "Sitting"
            return "Standing"

    return "Standing" if torso > 0 else "Unknown"
