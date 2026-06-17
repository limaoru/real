"""外部集成：Webhook 告警推送"""
import json
import urllib.error
import urllib.request

from retail.config.settings import ENABLE_WEBHOOK, WEBHOOK_URL


def push_alert_webhook(message: str, store_name: str = ""):
    if not ENABLE_WEBHOOK or not WEBHOOK_URL:
        return
    text = f"[{store_name}] {message}" if store_name else message
    payloads = [
        json.dumps({"msgtype": "text", "text": {"content": text}}).encode("utf-8"),
        json.dumps({"text": text}).encode("utf-8"),
    ]
    for body in payloads:
        try:
            req = urllib.request.Request(
                WEBHOOK_URL, data=body, headers={"Content-Type": "application/json"}, method="POST"
            )
            urllib.request.urlopen(req, timeout=5)
            return
        except (urllib.error.URLError, TimeoutError, OSError):
            continue
