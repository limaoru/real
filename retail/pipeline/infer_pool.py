"""多进程推理池：主循环非阻塞 submit / poll。"""
from __future__ import annotations

import multiprocessing as mp
from queue import Empty, Full

from retail.config.settings import INFER_MP_QUEUE_DEPTH
from retail.pipeline.infer_ipc import InferJob, InferResult
from retail.pipeline.infer_worker import _worker_entry


class CameraInferPool:
    """每路摄像头一个推理子进程 + 独立 ByteTrack 状态。"""

    def __init__(self, cam_names: list[str], ctx: mp.context.BaseContext | None = None):
        self._ctx = ctx or mp.get_context("spawn")
        self._cam_names = cam_names
        self._in_queues: dict[str, mp.Queue] = {}
        self._out_queues: dict[str, mp.Queue] = {}
        self._stop_events: dict[str, mp.Event] = {}
        self._processes: dict[str, mp.Process] = {}
        depth = max(1, INFER_MP_QUEUE_DEPTH)

        for name in cam_names:
            self._in_queues[name] = self._ctx.Queue(maxsize=depth)
            self._out_queues[name] = self._ctx.Queue(maxsize=depth + 2)
            self._stop_events[name] = self._ctx.Event()

    def start(self) -> None:
        for name in self._cam_names:
            if name in self._processes and self._processes[name].is_alive():
                continue
            proc = self._ctx.Process(
                target=_worker_entry,
                args=(
                    name,
                    self._in_queues[name],
                    self._out_queues[name],
                    self._stop_events[name],
                ),
                daemon=True,
                name=f"infer-{name}",
            )
            proc.start()
            self._processes[name] = proc
        print(f"🚀 推理进程池已启动：{len(self._cam_names)} 路 (multiprocessing spawn)")

    def stop(self) -> None:
        for ev in self._stop_events.values():
            ev.set()
        for proc in self._processes.values():
            proc.join(timeout=3)
            if proc.is_alive():
                proc.terminate()
        self._processes.clear()

    def submit(
        self,
        cam_name: str,
        seq: int,
        infer_frame,
        inv_scale: float,
        run_pose: bool,
    ) -> bool:
        job = InferJob(seq=seq, infer_frame=infer_frame, inv_scale=inv_scale, run_pose=run_pose)
        q = self._in_queues[cam_name]
        try:
            q.put_nowait(job)
            return True
        except Full:
            try:
                q.get_nowait()
            except Empty:
                pass
            try:
                q.put_nowait(job)
                return True
            except Full:
                return False

    def poll_all(self) -> dict[str, InferResult]:
        latest: dict[str, InferResult] = {}
        for name, q in self._out_queues.items():
            while True:
                try:
                    latest[name] = q.get_nowait()
                except Empty:
                    break
        return latest
