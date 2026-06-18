"""RTSP 视频流后台读取（threading 版，供 zones 等工具使用）。"""
from __future__ import annotations

import os
import threading
import time
from queue import Empty, Queue

import cv2
import numpy as np

from retail.vision.rtsp_ffmpeg import _FFMPEG_BIN, iter_rtsp_frames

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|stimeout;10000000",
)


def _opencv_has_ffmpeg() -> bool:
    try:
        return "FFMPEG:                      YES" in cv2.getBuildInformation()
    except Exception:
        return False


class RTSPStreamReader(threading.Thread):
    def __init__(self, name: str, rtsp_url: str, width: int, height: int):
        super().__init__(daemon=True)
        self.name = name
        self.rtsp_url = rtsp_url
        self.width = width
        self.height = height
        self.frame_queue: Queue = Queue(maxsize=3)
        self.stopped = False
        self.cap = None
        self._use_ffmpeg = not _opencv_has_ffmpeg()
        self.online = False
        self.last_frame_at = 0.0

    def _open_capture(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _enqueue(self, frame: np.ndarray) -> None:
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except Empty:
                pass
        self.frame_queue.put(frame)

    def _run_ffmpeg(self) -> None:
        print(
            f"📡 [线程] 正在通过 FFmpeg 连接 {self.name} "
            f"(OpenCV 无 FFmpeg 后端) …"
        )
        retry = 0
        while not self.stopped:
            try:
                connected = False
                for frame in iter_rtsp_frames(
                    self.rtsp_url,
                    self.width,
                    self.height,
                    stop_flag=lambda: self.stopped,
                ):
                    if self.stopped:
                        break
                    if not connected:
                        print(f"✅ [{self.name}] RTSP 已连接 ({self.width}x{self.height})")
                        connected = True
                        retry = 0
                    self.online = True
                    self.last_frame_at = time.time()
                    self._enqueue(frame)
            except FileNotFoundError:
                self.online = False
                retry += 1
                if retry == 1 or retry % 5 == 0:
                    print(f"⚠️ [{self.name}] 找不到 ffmpeg: {_FFMPEG_BIN}")
                time.sleep(2)

    def _run_opencv(self) -> None:
        print(f"📡 [线程] 正在通过 OpenCV 连接 {self.name} …")
        retry = 0
        self.cap = self._open_capture()
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.online = False
                retry += 1
                if retry == 1 or retry % 5 == 0:
                    print(f"⚠️ [{self.name}] RTSP 读帧失败，重连中… (第 {retry} 次)")
                if self.cap is not None:
                    self.cap.release()
                time.sleep(2)
                self.cap = self._open_capture()
                continue
            if retry > 0:
                print(f"✅ [{self.name}] RTSP 已连接")
            retry = 0
            self.online = True
            self.last_frame_at = time.time()
            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))
            self._enqueue(frame)
        if self.cap is not None:
            self.cap.release()

    def run(self) -> None:
        if self._use_ffmpeg:
            self._run_ffmpeg()
        else:
            self._run_opencv()

    def stop(self) -> None:
        self.stopped = True
