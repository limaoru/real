"""主动学习：低置信度 / 告警帧导出供标注"""
import json
from datetime import datetime
from pathlib import Path

import cv2

from retail.config.settings import ACTIVE_LEARNING_MIN_CONF, ENABLE_ACTIVE_LEARNING
from retail.paths import ACTIVE_LEARNING_DIR, ACTIVE_LEARNING_INDEX_PATH as INDEX_PATH


class ActiveLearningExporter:
    def __init__(self):
        self.exported = 0
        ACTIVE_LEARNING_DIR.mkdir(parents=True, exist_ok=True)

    def maybe_export(self, frame, cam: str, reason: str, conf: float = 1.0):
        if not ENABLE_ACTIVE_LEARNING or conf >= ACTIVE_LEARNING_MIN_CONF:
            return
        if self.exported > 200:
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = ACTIVE_LEARNING_DIR / f"{cam}_{ts}.jpg"
        cv2.imwrite(str(path), frame)
        meta = {
            "path": str(path),
            "cam": cam,
            "reason": reason,
            "conf": conf,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        items = []
        if INDEX_PATH.exists():
            try:
                items = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                items = []
        items.insert(0, meta)
        INDEX_PATH.write_text(json.dumps(items[:150], ensure_ascii=False, indent=2), encoding="utf-8")
        self.exported += 1

    @staticmethod
    def list_pending(limit: int = 20) -> list[dict]:
        if not INDEX_PATH.exists():
            return []
        try:
            return json.loads(INDEX_PATH.read_text(encoding="utf-8"))[:limit]
        except (json.JSONDecodeError, OSError):
            return []
