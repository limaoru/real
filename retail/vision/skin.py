"""肤色检测与可视化。"""
from __future__ import annotations

import cv2
import numpy as np

from retail.config.settings import SKIN_OVERLAY_ALPHA


def _skin_mask_bgr(roi_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2YCrCb)
    lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
    upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([0, 40, 80], dtype=np.uint8)
    upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)

    mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)
    return mask


def detect_and_draw_skin(
    frame: np.ndarray,
    mask_xy,
    label_pos=None,
) -> tuple[float, tuple | None]:
    if mask_xy is None or len(mask_xy) < 3:
        return 0.0, label_pos

    pts = np.asarray(mask_xy, dtype=np.int32).reshape(-1, 1, 2)
    x, y, w, h = cv2.boundingRect(pts)
    if w <= 2 or h <= 2:
        return 0.0, label_pos

    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
    if x2 - x1 <= 2 or y2 - y1 <= 2:
        return 0.0, label_pos

    roi = frame[y1:y2, x1:x2]
    max_w = 360
    scale = 1.0
    if roi.shape[1] > max_w:
        scale = max_w / float(roi.shape[1])
        roi_small = cv2.resize(roi, (int(roi.shape[1] * scale), int(roi.shape[0] * scale)))
    else:
        roi_small = roi

    pts_small = (pts - np.array([[x1, y1]], dtype=np.int32)) * scale
    pts_small = pts_small.astype(np.int32)
    person_mask = np.zeros((roi_small.shape[0], roi_small.shape[1]), dtype=np.uint8)
    cv2.fillPoly(person_mask, [pts_small], 255)

    skin_mask = _skin_mask_bgr(roi_small)
    skin_in_person = cv2.bitwise_and(skin_mask, person_mask)
    person_area = int(cv2.countNonZero(person_mask))
    skin_area = int(cv2.countNonZero(skin_in_person))
    skin_ratio = (skin_area / person_area) if person_area > 0 else 0.0

    if SKIN_OVERLAY_ALPHA > 0 and skin_area > 0:
        skin_up = skin_in_person
        if scale != 1.0:
            skin_up = cv2.resize(skin_in_person, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
        overlay = roi.copy()
        overlay[skin_up > 0] = (0, 0, 255)
        cv2.addWeighted(overlay, SKIN_OVERLAY_ALPHA, roi, 1 - SKIN_OVERLAY_ALPHA, 0, roi)

    if label_pos is None:
        label_pos = (x1 + (x2 - x1) // 2, y1)
    return skin_ratio, label_pos
