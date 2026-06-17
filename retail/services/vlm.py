"""VLM 多模态：场景摘要、告警复核、视频问答"""
import base64
import json
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from retail.config.settings import (
    ENABLE_VLM_ANALYSIS,
    ENABLE_VLM_QA,
    STORE_NAME,
    VLM_API_BASE,
    VLM_API_KEY,
    VLM_INTERVAL_SEC,
    VLM_MODEL,
)
from retail.paths import VLM_DIR, VLM_STATE


class VLMBridge:
    def __init__(self):
        self.last_run = 0.0
        self.last_insight = ""
        VLM_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _encode_frame(frame, max_w: int = 960) -> str:
        h, w = frame.shape[:2]
        if w > max_w:
            scale = max_w / w
            frame = cv2.resize(frame, (max_w, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        if not ok:
            return ""
        return base64.b64encode(buf.tobytes()).decode("ascii")

    def _call_api(self, prompt: str, image_b64: Optional[str] = None) -> Optional[str]:
        if not VLM_API_KEY or not VLM_API_BASE:
            return None
        url = VLM_API_BASE.rstrip("/") + "/chat/completions"
        content = [{"type": "text", "text": prompt}]
        if image_b64:
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            })
        body = json.dumps({
            "model": VLM_MODEL,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 400,
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {VLM_API_KEY}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, OSError, KeyError, json.JSONDecodeError):
            return None

    def _local_summary(self, metrics: dict, alerts: list[str]) -> str:
        parts = [
            f"{STORE_NAME} 场景快照：当前 {metrics.get('Person', 0)} 人，",
            f"排队 {metrics.get('queue', 0)}，进店累计相关指标正常。"]
        if metrics.get("queue", 0) >= 3:
            parts.append("收银区排队偏长，建议增开通道。")
        if alerts:
            parts.append(f"近期告警：{'; '.join(alerts[:3])}")
        return "".join(parts)

    def maybe_analyze(self, frame, metrics: dict, alerts: list[str]) -> str:
        if not ENABLE_VLM_ANALYSIS:
            return self.last_insight
        now = datetime.now().timestamp()
        if now - self.last_run < VLM_INTERVAL_SEC:
            return self.last_insight
        self.last_run = now
        prompt = (
            f"你是零售店铺视觉分析师。请用2-4句中文描述画面：客流、排队、货架、异常。"
            f"数据参考：人数{metrics.get('Person',0)} 排队{metrics.get('queue',0)}。"
        )
        b64 = self._encode_frame(frame)
        text = self._call_api(prompt, b64) if b64 else None
        if not text:
            text = self._local_summary(metrics, alerts)
        self.last_insight = text
        payload = {"ts": datetime.now().isoformat(timespec="seconds"), "insight": text}
        VLM_STATE.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return text

    def ask(self, question: str, frame=None, context: str = "") -> str:
        if not ENABLE_VLM_QA:
            return "VLM 问答未启用"
        prompt = f"店铺：{STORE_NAME}。上下文：{context}\n问题：{question}\n请简短中文回答。"
        b64 = self._encode_frame(frame) if frame is not None else None
        ans = self._call_api(prompt, b64)
        return ans or f"（本地模式）根据当前数据：{context or '暂无'}。建议查看仪表盘分时客流与告警录像。"

    @staticmethod
    def get_latest() -> dict:
        if not VLM_STATE.exists():
            return {}
        try:
            return json.loads(VLM_STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
