"""FFmpeg 子进程拉 RTSP 帧（OpenCV 无 FFmpeg 后端时的共用实现）。"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from typing import Callable, Iterator

import numpy as np

_FFMPEG_BIN = shutil.which("ffmpeg") or "/usr/local/bin/ffmpeg"


def iter_rtsp_frames(
    rtsp_url: str,
    width: int,
    height: int,
    *,
    stop_flag: Callable[[], bool] | None = None,
) -> Iterator[np.ndarray]:
    """阻塞生成 BGR 帧；stop_flag() 为 True 时退出。"""
    if not os.path.isfile(_FFMPEG_BIN):
        raise FileNotFoundError(f"找不到 ffmpeg: {_FFMPEG_BIN}")

    frame_bytes = width * height * 3
    cmd = [
        _FFMPEG_BIN,
        "-hide_banner",
        "-loglevel",
        "error",
        "-rtsp_transport",
        "tcp",
        "-i",
        rtsp_url,
        "-an",
        "-sn",
        "-vf",
        f"scale={width}:{height}",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-",
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=frame_bytes * 2,
    )
    try:
        while stop_flag is None or not stop_flag():
            raw = proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                break
            yield np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 3)).copy()
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)
