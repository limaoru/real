"""多店 SaaS：门店注册表 + 汇总导出"""
import json
from datetime import datetime
from pathlib import Path

from retail.config.settings import ENABLE_MULTISTORE, STORE_CHAIN_ID, STORE_ID, STORE_NAME
from retail.data.serialize import dumps_json
from retail.paths import CHAIN_SUMMARY_PATH, STORES_REGISTRY_PATH as REGISTRY_PATH


def _default_registry() -> dict:
    return {
        "chain_id": STORE_CHAIN_ID,
        "stores": [
            {
                "store_id": STORE_ID,
                "name": STORE_NAME,
                "role": "primary",
                "dashboard_port": 5050,
            }
        ],
    }


def load_registry() -> dict:
    if not REGISTRY_PATH.exists():
        reg = _default_registry()
        REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
        return reg
    try:
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _default_registry()


def export_chain_summary(live_state: dict) -> dict:
    if not ENABLE_MULTISTORE:
        return {}
    reg = load_registry()
    summary = {
        "chain_id": reg.get("chain_id", STORE_CHAIN_ID),
        "updated": datetime.now().isoformat(timespec="seconds"),
        "stores": [],
    }
    g = live_state.get("global", {})
    summary["stores"].append({
        "store_id": STORE_ID,
        "name": STORE_NAME,
        "persons": sum(c.get("persons", 0) for c in live_state.get("cameras", {}).values()),
        "conversion_pct": g.get("conversion_pct", 0),
        "funnel": g.get("funnel", {}),
        "falls": g.get("falls", 0),
    })
    CHAIN_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHAIN_SUMMARY_PATH.write_text(
        dumps_json(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def list_stores() -> list[dict]:
    return load_registry().get("stores", [])
