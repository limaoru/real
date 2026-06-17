"""可配置事件规则引擎"""
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class EventRule:
    rule_id: str
    name: str
    check: Callable[[dict], bool]
    message: Callable[[dict], str]
    once_key: Optional[str] = None  # 同会话只触发一次


DEFAULT_RULES = []


def _build_default_rules():
    return [
        EventRule("crowd", "店内拥挤", lambda ctx: ctx.get("persons", 0) >= 8,
                  lambda ctx: f"拥挤告警 当前{ctx['persons']}人", "crowd"),
        EventRule("queue", "排队过长", lambda ctx: ctx.get("queue", 0) >= 3,
                  lambda ctx: f"排队{ctx['queue']}人 建议加开收银", None),
        EventRule("no_staff", "收银区无人", lambda ctx: ctx.get("checkout_persons", 0) == 0 and ctx.get("queue", 0) >= 2,
                  lambda ctx: "收银区无人但有排队", "no_staff"),
        EventRule("high_browse", "浏览热区", lambda ctx: ctx.get("dwell_intent", 0) >= 2,
                  lambda ctx: f"强意向顾客 {ctx['dwell_intent']}人", None),
        EventRule("suspicious", "可疑停留", lambda ctx: ctx.get("suspicious", 0) > 0,
                  lambda ctx: f"可疑行为 {ctx['suspicious']}起", None),
        EventRule("fast_move", "快速移动", lambda ctx: ctx.get("fast_movers", 0) > 0,
                  lambda ctx: f"快速移动 {ctx['fast_movers']}人", None),
        EventRule("cross_store", "跨区顾客", lambda ctx: ctx.get("cross_cam_new", 0) > 0,
                  lambda ctx: f"跨摄像头识别 {ctx['cross_cam_new']}人", None),
        EventRule("fall", "跌倒检测", lambda ctx: ctx.get("falls", 0) > 0,
                  lambda ctx: f"跌倒/躺卧 {ctx['falls']}起", None),
        EventRule("family", "同行客群", lambda ctx: ctx.get("family_groups", 0) >= 2,
                  lambda ctx: f"同行组 {ctx['family_groups']}组", None),
        EventRule("zigzag", "折返行为", lambda ctx: ctx.get("zigzag", 0) > 0,
                  lambda ctx: f"折返可疑 {ctx['zigzag']}人", None),
        EventRule("low_conversion", "转化率低", lambda ctx: ctx.get("conversion_pct", 100) < 15 and ctx.get("funnel_entry", 0) > 8,
                  lambda ctx: f"转化率仅{ctx['conversion_pct']:.0f}% 优化动线", "low_conv"),
        EventRule("peak_hour", "客流高峰", lambda ctx: ctx.get("persons", 0) >= 10,
                  lambda ctx: f"高峰 {ctx['persons']}人 注意排班", None),
        EventRule("funnel_drop", "漏斗流失", lambda ctx: ctx.get("funnel_entry", 0) > 5 and ctx.get("funnel_checkout", 0) == 0,
                  lambda ctx: "有进店但未到达收银 检查动线", "funnel_drop"),
    ]


class EventEngine:
    def __init__(self):
        self.rules = _build_default_rules()
        self.fired_once: set[str] = set()

    def evaluate(self, ctx: dict) -> list[str]:
        alerts = []
        for rule in self.rules:
            try:
                if not rule.check(ctx):
                    continue
                if rule.once_key and rule.once_key in self.fired_once:
                    continue
                msg = rule.message(ctx)
                alerts.append(f"[{rule.name}] {msg}")
                if rule.once_key:
                    self.fired_once.add(rule.once_key)
            except (KeyError, TypeError, ZeroDivisionError):
                continue
        return alerts

    def reset_daily(self):
        self.fired_once.clear()
