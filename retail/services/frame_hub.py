"""跨进程共享最新 JPEG 帧（写磁盘），供 MJPEG Web 流使用。"""
from __future__ import annotations

import re
import threading
import time
from typing import Optional

import cv2
import numpy as np

from retail.config.settings import LIVE_STREAM_JPEG_QUALITY, LIVE_STREAM_MAX_WIDTH
from retail.paths import LIVE_FRAMES_DIR


def _safe_filename(stream_key: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", stream_key)


class FrameHub:
    """每路摄像头保存最新一帧 JPEG；runner 写入、dashboard 读取（跨进程）。"""

    _lock = threading.Lock()
    _mem: dict[str, bytes] = {}

    @classmethod
    def _path_for(cls, stream_key: str):
        LIVE_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
        return LIVE_FRAMES_DIR / f"{_safe_filename(stream_key)}.jpg"

    @classmethod
    def publish(cls, stream_key: str, bgr_frame: np.ndarray, quality: int | None = None) -> None:
        if bgr_frame is None or bgr_frame.size == 0:
            return
        q = quality if quality is not None else LIVE_STREAM_JPEG_QUALITY
        frame = bgr_frame
        h, w = frame.shape[:2]
        if LIVE_STREAM_MAX_WIDTH > 0 and w > LIVE_STREAM_MAX_WIDTH:
            scale = LIVE_STREAM_MAX_WIDTH / w
            frame = cv2.resize(frame, (LIVE_STREAM_MAX_WIDTH, int(h * scale)))
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
        if not ok:
            return
        jpeg = buf.tobytes()
        path = cls._path_for(stream_key)
        tmp = path.with_suffix(".jpg.tmp")
        tmp.write_bytes(jpeg)
        tmp.replace(path)
        with cls._lock:
            cls._mem[stream_key] = jpeg

    @classmethod
    def get_jpeg(cls, stream_key: str) -> Optional[bytes]:
        with cls._lock:
            if stream_key in cls._mem:
                return cls._mem[stream_key]
        path = cls._path_for(stream_key)
        if path.exists():
            try:
                return path.read_bytes()
            except OSError:
                return None
        return None

    @classmethod
    def list_streams(cls) -> list[str]:
        if not LIVE_FRAMES_DIR.exists():
            return []
        return [p.stem for p in LIVE_FRAMES_DIR.glob("*.jpg")]

    @classmethod
    def heatmap_key(cls, cam_name: str) -> str:
        return f"{cam_name}__heatmap"
