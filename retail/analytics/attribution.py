"""指标归因：排队、客流、转化率相关分析"""
from collections import defaultdict
from typing import Optional

from retail.config.settings import ENABLE_ATTRIBUTION


class AttributionEngine:
    """基于滑动窗口的简单因果提示（无需 POS）"""

    def __init__(self, window: int = 30):
        self.window = window
        self.history: list[dict] = []
        self.insights: list[str] = []

    def push(self, metrics: dict):
        if not ENABLE_ATTRIBUTION:
            return
        snap = {
            "persons": metrics.get("Person", 0),
            "queue": metrics.get("queue", 0),
            "standing": metrics.get("standing", 0),
            "groups": metrics.get("groups", 0),
            "conversion_pct": metrics.get("conversion_pct", 0),
        }
        self.history.append(snap)
        if len(self.history) > self.window:
            self.history.pop(0)
        if len(self.history) < 8:
            return
        self._analyze()

    def _analyze(self):
        recent = self.history[-8:]
        prev = self.history[-16:-8] if len(self.history) >= 16 else self.history[:8]
        q_now = sum(s["queue"] for s in recent) / len(recent)
        q_prev = sum(s["queue"] for s in prev) / len(prev) if prev else q_now
        p_now = sum(s["persons"] for s in recent) / len(recent)
        p_prev = sum(s["persons"] for s in prev) / len(prev) if prev else p_now
        insights = []
        if q_now > q_prev + 1.5 and p_now >= p_prev:
            insights.append("排队上升且店内人数未减 → 可能收银处理变慢或通道不足")
        if p_now > p_prev + 2 and q_now <= q_prev:
            insights.append("客流上升但排队未增 → 当前接待能力尚可")
        if p_now < p_prev - 2:
            insights.append("店内人数下降 → 关注时段促销或动线吸引力")
        g_now = sum(s["groups"] for s in recent) / len(recent)
        if g_now >= 2 and q_now > 2:
            insights.append("同行组较多且排队 → 家庭客增多，可考虑合并结账")
        self.insights = insights[-5:]

    def get_insights(self) -> list[str]:
        return list(self.insights)

    def export(self) -> dict:
        return {"insights": self.insights, "samples": len(self.history)}
