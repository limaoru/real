"""告警事件短视频回放（环形缓冲 + 触发录制）"""
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2

from retail.config.settings import (
    CLIP_AFTER_SEC,
    CLIP_BUFFER_SEC,
    CLIP_MAX_PER_HOUR,
    ENABLE_EVENT_CLIPS,
)
from retail.paths import CLIP_DIR


class ClipRecorder:
    def __init__(self, cam: str, fps: float = 10.0):
        self.cam = cam
        self.fps = max(5.0, fps)
        self.buffer: deque = deque(maxlen=int(self.fps * CLIP_BUFFER_SEC))
        self.pending_after: dict[str, int] = {}
        self.writers: dict[str, cv2.VideoWriter] = {}
        self.clip_meta: dict[str, dict] = {}
        self.hour_count = 0
        self.hour_key = datetime.now().strftime("%Y%m%d%H")
        CLIP_DIR.mkdir(parents=True, exist_ok=True)

    def push_frame(self, frame):
        if not ENABLE_EVENT_CLIPS:
            return
        self.buffer.append(frame.copy())
        done = []
        for tag, left in list(self.pending_after.items()):
            w = self.writers.get(tag)
            if w:
                w.write(frame)
            self.pending_after[tag] = left - 1
            if self.pending_after[tag] <= 0:
                done.append(tag)
        for tag in done:
            self._finalize(tag)

    def trigger(self, event_msg: str, fps: Optional[float] = None) -> Optional[str]:
        if not ENABLE_EVENT_CLIPS or not self.buffer:
            return None
        hk = datetime.now().strftime("%Y%m%d%H")
        if hk != self.hour_key:
            self.hour_key = hk
            self.hour_count = 0
        if self.hour_count >= CLIP_MAX_PER_HOUR:
            return None
        if fps:
            self.fps = fps
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in event_msg[:30])
        tag = f"{self.cam}_{ts}_{safe}"
        h, w = self.buffer[0].shape[:2]
        path = CLIP_DIR / f"{tag}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, self.fps, (w, h))
        if not writer.isOpened():
            return None
        for f in self.buffer:
            writer.write(f)
        self.writers[tag] = writer
        self.pending_after[tag] = int(self.fps * CLIP_AFTER_SEC)
        self.clip_meta[tag] = {
            "path": str(path),
            "cam": self.cam,
            "message": event_msg,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        self.hour_count += 1
        return str(path)

    def _finalize(self, tag: str):
        w = self.writers.pop(tag, None)
        if w:
            w.release()
        meta = self.clip_meta.pop(tag, None)
        self.pending_after.pop(tag, None)
        if meta:
            idx_path = CLIP_DIR / "index.json"
            import json
            items = []
            if idx_path.exists():
                try:
                    items = json.loads(idx_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    items = []
            items.insert(0, meta)
            idx_path.write_text(json.dumps(items[:100], ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def list_clips(limit: int = 20) -> list[dict]:
        idx = CLIP_DIR / "index.json"
        if not idx.exists():
            return []
        try:
            import json
            return json.loads(idx.read_text(encoding="utf-8"))[:limit]
        except (json.JSONDecodeError, OSError):
            return []