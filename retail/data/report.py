"""每日经营日报 HTML 生成"""
from datetime import datetime
from pathlib import Path
from typing import Optional

from retail.paths import REPORT_DIR
from retail.config.settings import STORE_NAME
from retail.data.db import StoreDB
from retail.services.digital_twin import DigitalTwin
from retail.services.vlm import VLMBridge


def generate_daily_report(db: Optional[StoreDB] = None) -> Path:
    db = db or StoreDB()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    funnel = db.get_funnel_today()
    hourly = db.get_hourly_today()
    events = db.get_recent_events(50)
    cross = db.get_cross_cam_summary()
    behaviors = db.get_recent_behaviors(30)
    vlm = db.get_latest_vlm() or VLMBridge.get_latest().get("insight", "")
    twin = DigitalTwin.load()
    date = datetime.now().strftime("%Y-%m-%d")
    entry = funnel.get("入口", 0)
    checkout = funnel.get("收银", 0)
    conv = f"{checkout / entry * 100:.1f}%" if entry else "N/A"

    rows_h = "".join(f"<tr><td>{h['hour']}</td><td>{h['count']}</td></tr>" for h in hourly)
    rows_e = "".join(f"<tr><td>{e['ts']}</td><td>{e['cam']}</td><td>{e['message']}</td></tr>" for e in events[:20])
    rows_c = "".join(
        f"<tr><td>G{r['gid']}</td><td>{r['first']}→{r['last']}</td><td>{r['visits']}</td></tr>" for r in cross[:10]
    ) or "<tr><td colspan=3>无</td></tr>"
    rows_b = "".join(
        f"<tr><td>{b['ts']}</td><td>{b['cam']}</td><td>{b['behavior']}</td></tr>" for b in behaviors[:15]
    ) or "<tr><td colspan=3>无</td></tr>"
    sim_hint = twin.get("simulation", {}).get("hint", "暂无") if twin else "暂无"

    html = f"""<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">
<title>{STORE_NAME} 日报 {date}</title>
<style>body{{font-family:sans-serif;padding:24px;background:#f5f5f5}}
.card{{background:#fff;padding:16px;margin:12px 0;border-radius:8px;box-shadow:0 1px 4px #0001}}
table{{border-collapse:collapse;width:100%}}td,th{{border:1px solid #ddd;padding:8px}}</style></head>
<body><h1>{STORE_NAME} 经营日报</h1><p>{date}</p>
<div class="card"><h2>动线漏斗</h2>
<p>入口 {funnel.get('入口',0)} → 浏览 {funnel.get('浏览',0)} → 收银 {checkout} → 完整动线 {funnel.get('完整动线',0)}</p>
<p><b>浏览→收银转化率: {conv}</b></p></div>
<div class="card"><h2>VLM 场景摘要</h2><p>{vlm or '未生成'}</p></div>
<div class="card"><h2>数字孪生建议</h2><p>{sim_hint}</p></div>
<div class="card"><h2>分时客流</h2><table><tr><th>时段</th><th>人数</th></tr>{rows_h}</table></div>
<div class="card"><h2>行为识别</h2><table><tr><th>时间</th><th>摄像头</th><th>行为</th></tr>{rows_b}</table></div>
<div class="card"><h2>跨摄像头顾客</h2><table><tr><th>GID</th><th>路径</th><th>访问</th></tr>{rows_c}</table></div>
<div class="card"><h2>重要事件</h2><table><tr><th>时间</th><th>摄像头</th><th>内容</th></tr>{rows_e}</table></div>
</body></html>"""
    path = REPORT_DIR / f"report_{date}.html"
    path.write_text(html, encoding="utf-8")
    return path
