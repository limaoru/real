"""RTSP 视频流后台读取。"""
from __future__ import annotations

import threading
import time
from queue import Empty, Queue

import cv2


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
        self.online = False
        self.last_frame_at = 0.0

    def _open_capture(self):
        cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def run(self) -> None:
        print(f"📡 [线程启动] 正在通过 OpenCV 连接 {self.name} ...")
        self.cap = self._open_capture()

        while not self.stopped:
            if not self.cap.isOpened():
                self.online = False
                time.sleep(2)
                self.cap = self._open_capture()
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                self.online = False
                self.cap.release()
                time.sleep(2)
                self.cap = self._open_capture()
                continue

            self.online = True
            self.last_frame_at = time.time()

            if frame.shape[1] != self.width or frame.shape[0] != self.height:
                frame = cv2.resize(frame, (self.width, self.height))

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except Empty:
                    pass
            self.frame_queue.put(frame)

        if self.cap is not None:
            self.cap.release()

    def stop(self) -> None:
        self.stopped = True
