#!/usr/bin/env python3
"""
店铺零售 Web 仪表盘（视觉进阶全功能版）

运行：python -m retail dashboard
访问：http://127.0.0.1:5050
实时画面：http://<IP>:5050/live/<摄像头名>
"""
import csv
import json
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from retail.config.settings import DASHBOARD_HOST, DASHBOARD_PORT, LIVE_STREAM_FPS
from retail.paths import CHAIN_SUMMARY_PATH, CLIP_DIR, LIVE_STATE_PATH as LIVE_STATE, LOG_DIR
from retail.services.digital_twin import DigitalTwin
from retail.services.frame_hub import FrameHub
from retail.services.multistore import list_stores
from retail.services.vlm import VLMBridge

HTML = """<!DOCTYPE html>
<html lang="zh"><head><meta charset="utf-8"/><title>店铺智能仪表盘</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,sans-serif;background:#0b0f14;color:#e8eaed;padding:20px;line-height:1.5}
h1{font-size:1.5rem;margin-bottom:4px}
.sub{color:#8b949e;font-size:.85rem;margin-bottom:20px}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
.card{background:#151b23;border:1px solid #30363d;border-radius:12px;padding:16px}
.card h2{font-size:.95rem;color:#58a6ff;margin-bottom:10px}
.stat{display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #21262d;font-size:.88rem}
.val{color:#3fb950;font-weight:600}
.tag{display:inline-block;background:#1f3d2f;color:#3fb950;padding:2px 8px;border-radius:99px;font-size:.72rem;margin:2px}
.warn{color:#f85149}.section{margin-top:18px}
table{width:100%;border-collapse:collapse;font-size:.8rem}
th,td{padding:6px 8px;border-bottom:1px solid #21262d;text-align:left}
th{color:#8b949e}
.funnel-bar{display:flex;height:22px;border-radius:6px;overflow:hidden;margin:8px 0}
.funnel-bar div{display:flex;align-items:center;justify-content:center;font-size:.7rem;color:#fff}
.insight{background:#1c2128;padding:10px;border-radius:8px;font-size:.85rem;line-height:1.6}
.qa-box{display:flex;gap:8px;margin-top:8px}
.qa-box input{flex:1;padding:8px;border-radius:8px;border:1px solid #30363d;background:#0d1117;color:#e8eaed}
.qa-box button{padding:8px 14px;border-radius:8px;border:none;background:#238636;color:#fff;cursor:pointer}
.clip-link{color:#58a6ff;font-size:.8rem;display:block;margin:4px 0}
.live-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:16px}
.live-box h3{font-size:.9rem;color:#8b949e;margin-bottom:8px}
.live-box img{width:100%;border-radius:8px;background:#000;min-height:180px}
.live-label{font-size:.75rem;color:#6e7681;margin:6px 0 2px}
</style></head><body>
<h1>店铺智能 BI 仪表盘</h1>
<p class="sub">更新：<span id="u">-</span> · 5秒刷新 · Tailscale/Mac 可看实时 MJPEG</p>
<div class="section card"><h2>实时分析画面</h2>
<p class="sub" style="margin-bottom:12px">需 PC 运行 <code>python -m retail run</code>；单路直达 <code>/live/摄像头名</code></p>
<div class="live-grid" id="live"></div></div>
<div class="grid" id="cards"></div>
<div class="section card"><h2>今日动线漏斗</h2><div id="funnel"></div>
<p style="margin-top:8px" id="features"></p></div>
<div class="grid section">
  <div class="card"><h2>VLM 场景洞察</h2><div class="insight" id="vlm">加载中…</div></div>
  <div class="card"><h2>智能归因</h2><div id="attrib" class="insight"></div></div>
  <div class="card"><h2>数字孪生仿真</h2><div id="twin"></div></div>
</div>
<div class="grid section">
  <div class="card"><h2>行为识别</h2><table id="behaviors"><thead><tr><th>时间</th><th>行为</th><th>ID</th></tr></thead><tbody></tbody></table></div>
  <div class="card"><h2>告警录像</h2><div id="clips"></div></div>
  <div class="card"><h2>多店总览</h2><div id="chain"></div></div>
</div>
<div class="section card"><h2>视频问答（VLM）</h2>
<div class="qa-box"><input id="q" placeholder="例：今天下午收银台为什么排队？"/><button onclick="ask()">提问</button></div>
<div class="insight" id="qa" style="margin-top:10px">填写 API Key 后可用云端 VLM；未配置时使用本地摘要。</div></div>
<div class="grid section">
  <div class="card"><h2>分时客流</h2><table id="hourly"><thead><tr><th>时段</th><th>人数</th></tr></thead><tbody></tbody></table></div>
  <div class="card"><h2>跨摄像头顾客</h2><table id="cross"><thead><tr><th>GID</th><th>路径</th><th>次数</th></tr></thead><tbody></tbody></table></div>
  <div class="card"><h2>最近事件</h2><div id="events" style="font-size:.82rem"></div></div>
</div>
<div class="section card"><h2>CSV 历史</h2>
<table id="hist"><thead><tr><th>时间</th><th>摄像头</th><th>人数</th><th>进</th><th>出</th><th>排队</th></tr></thead><tbody></tbody></table></div>
<script>
async function ask(){
  const q=document.getElementById('q').value;
  if(!q)return;
  const r=await fetch('/api/qa?q='+encodeURIComponent(q)).then(r=>r.json());
  document.getElementById('qa').textContent=r.answer||'无回答';
}
async function go(){
  const d=await fetch('/api/all').then(r=>r.json());
  document.getElementById('u').textContent=d.updated||'-';
  let h='';
  for(const [cam,v] of Object.entries(d.cameras||{})){
    const p=v.plus||{};
    const f=p.funnel||{};
    h+=`<div class="card"><h2>${cam} <span class="tag">FPS ${v.fps||0}</span></h2>
      <div class="stat"><span>累计客流</span><span class="val">${v.customers_total||0}</span></div>
      <div class="stat"><span>进店/离店</span><span class="val">${v.line_in||0} / ${v.line_out||0}</span></div>
      <div class="stat"><span>当前/峰值</span><span class="val">${v.persons||0} / ${v.peak||0}</span></div>
      <div class="stat"><span>站/坐/排队</span><span class="val">${v.standing||0}/${v.sitting||0}/${v.queue||0}</span></div>
      <div class="stat"><span>坪效</span><span class="val">${(p.density||0).toFixed(2)} 人/㎡</span></div>
      <div class="stat"><span>下小时/今日余</span><span class="val">~${p.forecast_next_hour||0} / ${p.forecast_rest_day||0}</span></div>
      <div class="stat"><span>停留 览/趣/向</span><span class="val">${(p.dwell_tiers||{}).浏览||0}/${(p.dwell_tiers||{}).兴趣||0}/${(p.dwell_tiers||{}).意向||0}</span></div>
      <div class="stat"><span>漏斗 入/览/收</span><span class="val">${f.入口||0}/${f.浏览||0}/${f.收银||0}</span></div>
      <div style="margin-top:8px;font-size:.8rem" class="warn">${(v.alerts||[]).slice(0,3).join('<br>')||'无告警'}</div></div>`;
  }
  document.getElementById('cards').innerHTML=h||'<p>请先运行 python -m retail run</p>';
  const cams=Object.keys(d.cameras||{});
  document.getElementById('live').innerHTML=cams.length?cams.map(cam=>{
    const enc=encodeURIComponent(cam);
    return `<div class="live-box"><h3>${cam}</h3>
      <div class="live-label">分析叠加</div>
      <img src="/live/${enc}" alt="${cam}"/>
      <div class="live-label">热力图</div>
      <img src="/live/${enc}/heatmap" alt="${cam} heatmap"/></div>`;
  }).join(''):'<p>等待主程序推流…</p>';
  const gf=d.global?.funnel||{};
  const mx=Math.max(1,gf.入口||0,gf.浏览||0,gf.收银||0);
  document.getElementById('funnel').innerHTML=`
    <div class="funnel-bar">
      <div style="width:${(gf.入口||0)/mx*33}%;background:#238636">入口 ${gf.入口||0}</div>
      <div style="width:${(gf.浏览||0)/mx*33}%;background:#1f6feb">浏览 ${gf.浏览||0}</div>
      <div style="width:${(gf.收银||0)/mx*34}%;background:#8957e5">收银 ${gf.收银||0}</div>
    </div>
    <div class="stat"><span>完整动线</span><span class="val">${gf.完整动线||0}</span></div>
    <div class="stat"><span>转化率</span><span class="val">${d.global?.conversion_pct||0}%</span></div>
    <div class="stat"><span>同行组</span><span class="val">${d.global?.family_groups||0}</span></div>
    <div class="stat"><span>跌倒/折返</span><span class="val">${d.global?.falls||0} / ${d.global?.zigzag||0}</span></div>`;
  const feats=d.global?.features||{};
  document.getElementById('features').innerHTML=Object.entries(feats).map(([k,v])=>
    `<span class="tag" style="background:${v?'#1f3d2f':'#3d1f1f'};color:${v?'#3fb950':'#f85149'}">${k}</span>`).join(' ');
  const vlm=(d.vlm||{}).insight||d.vlm_insight||'暂无洞察，等待 VLM 周期分析…';
  document.getElementById('vlm').textContent=vlm;
  const ins=(d.attribution||{}).insights||d.global?.attribution?.insights||[];
  document.getElementById('attrib').innerHTML=ins.length?ins.map(i=>`<div>• ${i}</div>`).join(''):'数据积累中…';
  const twin=d.twin||{};
  document.getElementById('twin').innerHTML=twin.simulation?
    `<div class="stat"><span>热点数</span><span class="val">${(twin.hotspots||[]).length}</span></div>
     <div class="stat"><span>仿真建议</span><span class="val">${twin.simulation.hint||'-'}</span></div>`:'等待数字孪生数据…';
  document.querySelector('#behaviors tbody').innerHTML=(d.behaviors||[]).map(b=>
    `<tr><td>${b.ts}</td><td>${b.behavior}</td><td>${b.track_id}</td></tr>`).join('')||'<tr><td colspan=3>暂无</td></tr>';
  document.getElementById('clips').innerHTML=(d.clips||[]).map(c=>
    `<span class="clip-link">[${c.ts}] ${c.cam}: ${c.message}</span>`).join('')||'暂无告警录像';
  document.getElementById('chain').innerHTML=(d.chain_stores||[]).map(s=>
    `<div class="stat"><span>${s.name}</span><span class="val">${s.persons||0}人 · 转化${s.conversion_pct||0}%</span></div>`).join('')||'单店模式';
  document.querySelector('#hourly tbody').innerHTML=(d.hourly_chart||[]).map(r=>
    `<tr><td>${r.hour}</td><td>${r.count}</td></tr>`).join('');
  document.querySelector('#cross tbody').innerHTML=(d.cross_cam||[]).map(r=>
    `<tr><td>G${r.gid}</td><td>${r.first}→${r.last}</td><td>${r.visits}</td></tr>`).join('')||'<tr><td colspan=3>暂无</td></tr>';
  document.getElementById('events').innerHTML=(d.recent_events||[]).map(e=>
    `<div>[${e.ts}] ${e.cam} ${e.message}</div>`).join('')||'暂无';
  document.querySelector('#hist tbody').innerHTML=(d.history||[]).map(r=>
    `<tr><td>${r.time}</td><td>${r.cam}</td><td>${r.persons}</td><td>${r.in}</td><td>${r.out}</td><td>${r.queue}</td></tr>`).join('');
}
go();setInterval(go,5000);
</script></body></html>"""


def read_live():
    if not LIVE_STATE.exists():
        return {"updated": None, "cameras": {}, "global": {}}
    try:
        data = json.loads(LIVE_STATE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"updated": None, "cameras": {}, "global": {}}
    chain_path = CHAIN_SUMMARY_PATH
    if chain_path.exists():
        try:
            chain = json.loads(chain_path.read_text(encoding="utf-8"))
            data["chain_stores"] = chain.get("stores", [])
        except (json.JSONDecodeError, OSError):
            pass
    data.setdefault("chain_stores", [{"name": s.get("name"), "store_id": s.get("store_id")} for s in list_stores()])
    if not data.get("twin"):
        data["twin"] = DigitalTwin.load()
    return data


def read_history(limit=20):
    rows = []
    if not LOG_DIR.exists():
        return rows
    for p in sorted(LOG_DIR.glob("*.csv"), reverse=True):
        try:
            with open(p, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    rows.append({
                        "time": row.get("time", ""), "cam": row.get("cam", ""),
                        "persons": row.get("persons", ""), "in": row.get("in", ""),
                        "out": row.get("out", ""), "queue": row.get("queue", ""),
                    })
        except OSError:
            pass
    rows.sort(key=lambda r: r["time"], reverse=True)
    return rows[:limit]


def _parse_live_path(path: str) -> str | None:
    """/live/Cam_xxx 或 /live/Cam_xxx/heatmap → FrameHub stream key"""
    if not path.startswith("/live/"):
        return None
    rest = unquote(path[len("/live/"):].strip("/"))
    if not rest:
        return None
    if rest.endswith("/heatmap"):
        cam = rest[: -len("/heatmap")].strip("/")
        return FrameHub.heatmap_key(cam) if cam else None
    return rest


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def _stream_mjpeg(self, stream_key: str) -> None:
        boundary = b"frame"
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={boundary.decode()}")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Connection", "close")
        self.end_headers()
        interval = 1.0 / max(1, LIVE_STREAM_FPS)
        try:
            while True:
                jpeg = FrameHub.get_jpeg(stream_key)
                if jpeg:
                    self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_GET(self):
        path = urlparse(self.path).path
        stream_key = _parse_live_path(path)
        if stream_key is not None:
            self._stream_mjpeg(stream_key)
            return
        if path == "/api/all":
            data = read_live()
            data["history"] = read_history()
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif path == "/api/qa":
            qs = parse_qs(urlparse(self.path).query)
            question = (qs.get("q") or [""])[0]
            live = read_live()
            ctx = json.dumps(live.get("global", {}), ensure_ascii=False)[:500]
            answer = VLMBridge().ask(question, context=ctx)
            body = json.dumps({"answer": answer}, ensure_ascii=False).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(body)
        elif path.startswith("/clips/"):
            fname = path.split("/clips/", 1)[-1]
            fpath = CLIP_DIR / fname
            if fpath.exists() and fpath.suffix == ".mp4":
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_response(404)
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML.encode("utf-8"))


def serve(host: str | None = None, port: int | None = None) -> None:
    host = host or DASHBOARD_HOST
    port = port or DASHBOARD_PORT
    LOG_DIR.mkdir(exist_ok=True)
    print(f"仪表盘: http://127.0.0.1:{port}")
    print(f"实时画面: http://127.0.0.1:{port}/live/<摄像头名>")
    print("Tailscale: 将 127.0.0.1 换成 PC 的 100.x.x.x")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    serve()
