"""画面叠加层：BI 面板、热力图、FPS。"""
from __future__ import annotations

import cv2
import numpy as np

from retail.analytics.cam import CameraAnalytics
from retail.ui.text import flush_text, queue_text


def draw_analytics_panel(
    frame: np.ndarray,
    cam_analytics: CameraAnalytics,
    customer_count: int,
    metrics: dict,
    brain_summary: dict | None = None,
) -> None:
    lines = [
        f"累计客流: {customer_count}",
        f"进店/离店: {cam_analytics.line_in}/{cam_analytics.line_out}",
        f"峰值人数: {cam_analytics.peak_occupancy}",
        f"站立/坐下: {metrics.get('standing', 0)}/{metrics.get('sitting', 0)}",
        f"聚集/排队/携包: {metrics.get('groups', 0)}/{metrics.get('queue', 0)}/{metrics.get('bags', 0)}",
    ]
    if brain_summary:
        funnel = brain_summary.get("funnel", {})
        if funnel:
            lines.append(
                f"漏斗 入/览/收: {funnel.get('入口', 0)}/{funnel.get('浏览', 0)}/{funnel.get('收银', 0)}"
            )
            lines.append(f"转化率: {brain_summary.get('conversion_pct', 0)}%")
            lines.append(f"完整动线: {funnel.get('完整动线', 0)}")
        lines.append(
            f"同行组: {brain_summary.get('family_groups', 0)} "
            f"货架互动: {brain_summary.get('shelf_engagement', 0)}"
        )
        tiers = brain_summary.get("dwell_tiers", {})
        if tiers:
            lines.append(
                f"停留 览/趣/向: {tiers.get('浏览', 0)}/{tiers.get('兴趣', 0)}/{tiers.get('意向', 0)}"
            )
        lines.append(f"坪效 {brain_summary.get('density', 0):.2f}人/㎡")
        lines.append(f"下小时预测: ~{brain_summary.get('forecast_next_hour', 0)}人")
        if brain_summary.get("suspicious", 0):
            lines.append(f"可疑: {brain_summary['suspicious']}")
    for zname, cnt in list(cam_analytics.zone_enter_total.items())[:3]:
        lines.append(f"{zname}累计: {cnt}")

    panel_h = 20 + min(len(lines), 12) * 20
    cv2.rectangle(frame, (10, 10), (360, panel_h), (0, 0, 0), -1)
    for i, text in enumerate(lines[:12]):
        queue_text(frame, text, 20, 32 + i * 20, (0, 255, 255), 16)

    if cam_analytics.alerts:
        show_alerts = list(cam_analytics.alerts)[:3]
        alert_y = frame.shape[0] - 10 - len(show_alerts) * 22
        cv2.rectangle(
            frame,
            (frame.shape[1] - 420, alert_y - 10),
            (frame.shape[1] - 10, frame.shape[0] - 10),
            (0, 0, 80),
            -1,
        )
        for i, alert in enumerate(show_alerts):
            queue_text(frame, alert[:40], frame.shape[1] - 410, alert_y + i * 22 + 18, (0, 200, 255), 16)


def build_heatmap_views(frame: np.ndarray, accum_heatmap: np.ndarray, width: int):
    if not np.any(accum_heatmap):
        return frame.copy(), np.zeros_like(frame)

    blur_k = max(51, int(51 * width / 1280) | 1)
    blur_heatmap = cv2.GaussianBlur(accum_heatmap, (blur_k, blur_k), 0)
    heatmap_fixed = np.clip(blur_heatmap * 4, 0, 255).astype(np.uint8)
    color_heatmap = cv2.applyColorMap(heatmap_fixed, cv2.COLORMAP_JET)
    overlay_frame = cv2.addWeighted(frame, 0.65, color_heatmap, 0.35, 0)
    heatmap_panel = cv2.addWeighted(frame, 0.12, color_heatmap, 0.88, 0)
    return overlay_frame, heatmap_panel


def draw_fps_hud(frame: np.ndarray, fps: float, model_label: str) -> None:
    fps_text = f"FPS: {fps:.1f}"
    model_text = f"Model: {model_label}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    fps_size, _ = cv2.getTextSize(fps_text, font, 0.8, 2)
    model_size, _ = cv2.getTextSize(model_text, font, 0.55, 1)
    panel_w = max(fps_size[0], model_size[0]) + 24
    panel_h = fps_size[1] + model_size[1] + 28
    x1 = frame.shape[1] - panel_w - 10
    y1 = 10
    x2 = frame.shape[1] - 10
    y2 = y1 + panel_h
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (x1 + 12, y1 + fps_size[1] + 8), font, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, model_text, (x1 + 12, y2 - 10), font, 0.55, (200, 200, 200), 1)
    