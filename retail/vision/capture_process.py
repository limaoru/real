"""RTSP 采集：multiprocessing.Process + Queue（与推理进程解耦）。"""
from __future__ import annotations

import multiprocessing as mp
import time
from queue import Empty, Full

import numpy as np

from retail.vision.rtsp_ffmpeg import iter_rtsp_frames


def _push_latest_frame(frame_queue: mp.Queue, frame: np.ndarray) -> None:
    """入队最新帧；满则丢旧帧。绝不因背压抛错中断 RTSP。"""
    for _ in range(8):
        try:
            frame_queue.put(frame, block=False)
            return
        except Full:
            try:
                frame_queue.get_nowait()
            except Empty:
                time.sleep(0.002)
    # 最后兜底：阻塞替换队首
    try:
        frame_queue.get_nowait()
    except Empty:
        pass
    frame_queue.put(frame, block=True, timeout=2.0)


def _capture_worker(
    name: str,
    rtsp_url: str,
    width: int,
    height: int,
    frame_queue: mp.Queue,
    stop_event: mp.Event,
    online_flag,
) -> None:
    print(f"📡 [采集进程] {name} 启动 FFmpeg …")
    retry = 0
    while not stop_event.is_set():
        try:
            connected = False
            for frame in iter_rtsp_frames(
                rtsp_url, width, height, stop_flag=stop_event.is_set
            ):
                if stop_event.is_set():
                    break
                if not connected:
                    print(f"✅ [采集进程] {name} RTSP 已连接 ({width}x{height})")
                    connected = True
                    retry = 0
                online_flag.value = True
                _push_latest_frame(frame_queue, frame)

            # FFmpeg 正常 EOF（断流/摄像头重启），短退避后重连
            if connected and not stop_event.is_set():
                online_flag.value = False
                retry += 1
                if retry <= 3 or retry % 10 == 0:
                    print(f"⚠️ [{name}] RTSP 断流，{0.5}s 后重连…")
                time.sleep(0.5)

        except FileNotFoundError as exc:
            online_flag.value = False
            retry += 1
            print(f"⚠️ [{name}] 采集失败: {exc}")
            time.sleep(2)
        except OSError as exc:
            online_flag.value = False
            retry += 1
            if retry == 1 or retry % 5 == 0:
                print(f"⚠️ [{name}] 采集 I/O 错误: {exc}")
            time.sleep(2)


class RTSPCaptureProcess:
    """每路摄像头一个独立采集进程。"""

    def __init__(
        self,
        name: str,
        rtsp_url: str,
        width: int,
        height: int,
        *,
        ctx: mp.context.BaseContext | None = None,
        queue_depth: int = 2,
    ):
        self.name = name
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self._ctx = ctx or mp.get_context("spawn")
        self.frame_queue: mp.Queue = self._ctx.Queue(maxsize=max(2, queue_depth))
        self.stop_event = self._ctx.Event()
        self.online = self._ctx.Value("b", False)
        self._process: mp.Process | None = None

    def start(self) -> None:
        if self._process is not None and self._process.is_alive():
            return
        self._process = self._ctx.Process(
            target=_capture_worker,
            args=(
                self.name,
                self.rtsp_url,
                self.width,
                self.height,
                self.frame_queue,
                self.stop_event,
                self.online,
            ),
            daemon=True,
            name=f"capture-{self.name}",
        )
        self._process.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self._process is not None:
            self._process.join(timeout=3)
            if self._process.is_alive():
                self._process.terminate()
            self._process = None

    @property
    def is_online(self) -> bool:
        return bool(self.online.value)

    def read_latest(self) -> np.ndarray | None:
        latest: np.ndarray | None = None
        while True:
            try:
                latest = self.frame_queue.get_nowait()
            except Empty:
                break
        return latest
