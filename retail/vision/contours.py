"""人体分割轮廓绘制。"""
from __future__ import annotations

import cv2
import numpy as np


def draw_person_contour(
    frame: np.ndarray,
    mask_xy,
    color: tuple[int, int, int],
    thickness: int = 2,
    fill_alpha: float = 0.22,
) -> tuple[int, int] | None:
    if mask_xy is None or len(mask_xy) < 3:
        return None
    pts = np.asarray(mask_xy, dtype=np.int32).reshape(-1, 1, 2)
    if fill_alpha > 0:
        fill_layer = frame.copy()
        cv2.fillPoly(fill_layer, [pts], color)
        cv2.addWeighted(fill_layer, fill_alpha, frame, 1 - fill_alpha, 0, frame)
    cv2.polylines(frame, [pts], True, color, thickness, cv2.LINE_AA)
    top_y = int(np.min(pts[:, 0, 1]))
    top_x = int(np.mean(pts[:, 0, 0]))
    return top_x, top_y
