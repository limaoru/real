"""实时状态导出（供 Web 仪表盘）。"""
from __future__ import annotations

from datetime import datetime

from retail.config.settings import ENABLE_WEB_DASHBOARD
from retail.data.serialize import dumps_json
from retail.paths import LIVE_STATE_PATH
from retail.services.multistore import export_chain_summary


def export_live_state(
    cam_analytics,
    customer_counts,
    fps_state,
    metrics_by_cam,
    brain=None,
    brain_summaries=None,
    extras=None,
    stream_online=None,
) -> None:
    if not ENABLE_WEB_DASHBOARD:
        return
    extras = extras or {}
    stream_online = stream_online or {}
    payload = {
        "updated": datetime.now().isoformat(timespec="seconds"),
        "cameras": {},
        "global": brain.get_global_export() if brain else {},
    }
    if brain and brain.db:
        payload["recent_events"] = brain.db.get_recent_events(20)
        payload["hourly_chart"] = brain.db.get_hourly_today()
        payload["cross_cam"] = brain.db.get_cross_cam_summary()
        payload["behaviors"] = brain.db.get_recent_behaviors(15)
        vlm_db = brain.db.get_latest_vlm()
        if vlm_db:
            payload["vlm_insight"] = vlm_db
    payload.update(extras)
    for cam_name, analytics in cam_analytics.items():
        m = metrics_by_cam.get(cam_name, {})
        entry = {
            "customers_total": customer_counts.get(cam_name, 0),
            "line_in": analytics.line_in,
            "line_out": analytics.line_out,
            "peak": analytics.peak_occupancy,
            "standing": m.get("standing", 0),
            "sitting": m.get("sitting", 0),
            "groups": m.get("groups", 0),
            "queue": m.get("queue", 0),
            "bags": m.get("bags", 0),
            "persons": m.get("Person", 0),
            "zone_current": dict(analytics.zone_current),
            "zone_enter_total": dict(analytics.zone_enter_total),
            "hourly": dict(analytics.hourly),
            "alerts": list(analytics.alerts),
            "fps": round(fps_state.get(cam_name, {}).get("fps", 0), 1),
            "stream_online": stream_online.get(cam_name, False),
        }
        if brain_summaries and cam_name in brain_summaries:
            entry["plus"] = brain_summaries[cam_name]
        payload["cameras"][cam_name] = entry
    LIVE_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    LIVE_STATE_PATH.write_text(
        dumps_json(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    export_chain_summary(payload)
