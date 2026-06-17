"""项目路径常量（单一来源）。"""
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

LOG_DIR = PROJECT_ROOT / "analytics_logs"
ZONE_CONFIG_PATH = PROJECT_ROOT / "zone_config.json"
DB_PATH = LOG_DIR / "store.db"
CLIP_DIR = LOG_DIR / "clips"
REPORT_DIR = LOG_DIR / "reports"
CHAIN_SUMMARY_PATH = LOG_DIR / "chain_summary.json"
AGG_STATE = CHAIN_SUMMARY_PATH  # 别名
LIVE_STATE_PATH = LOG_DIR / "live_state.json"
STORES_REGISTRY_PATH = PROJECT_ROOT / "stores_registry.json"
TWIN_STATE_PATH = LOG_DIR / "twin_state.json"
VLM_DIR = LOG_DIR / "vlm"
VLM_STATE_PATH = VLM_DIR / "latest_insight.json"
VLM_STATE = VLM_STATE_PATH
ACTIVE_LEARNING_DIR = LOG_DIR / "active_learning"
ACTIVE_LEARNING_INDEX_PATH = ACTIVE_LEARNING_DIR / "index.json"
