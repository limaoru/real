"""每路摄像头独立推理进程（seg + ByteTrack + pose）。"""
from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from queue import Empty

import numpy as np

from retail.pipeline.infer_ipc import InferJob, InferResult


def camera_infer_worker(
    cam_name: str,
    in_queue: mp.Queue,
    out_queue: mp.Queue,
    stop_event: mp.Event,
) -> None:
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

    import torch

    from retail.config.models import POSE_MODEL, SEG_MODEL
    from retail.config.settings import POSE_EVERY_N_FRAMES, SEG_EVERY_N_FRAMES
    from retail.vision.infer_backend import infer_profile, load_yolo_model, track_kwargs, yolo_kwargs
    from retail.vision.seg_track import ByteTrackSegRunner

    profile = infer_profile()
    print(f"🧠 [推理进程] {cam_name} 加载模型 ({profile.label}) …")
    track_kw = track_kwargs()
    seg_runner = ByteTrackSegRunner(
        load_yolo_model(SEG_MODEL), track_kw, every_n=SEG_EVERY_N_FRAMES
    )
    pose_model = load_yolo_model(POSE_MODEL)
    pose_kw = yolo_kwargs(conf=0.4)
    print(f"✅ [推理进程] {cam_name} 就绪")

    with torch.inference_mode():
        while not stop_event.is_set():
            try:
                job: InferJob = in_queue.get(timeout=0.25)
            except Empty:
                continue

            # 只保留最新任务，丢弃积压
            while True:
                try:
                    job = in_queue.get_nowait()
                except Empty:
                    break

            t0 = time.perf_counter()
            track_frame = seg_runner.run(job.infer_frame)
            kps_xy_np = None
            kps_conf_np = None

            has_person = (
                track_frame.clss is not None
                and np.any(track_frame.clss == 0)
            )
            run_pose = job.run_pose or track_frame.is_seg_frame
            if run_pose and has_person:
                pose_results = pose_model(job.infer_frame, **pose_kw)[0]
                if pose_results.keypoints is not None:
                    kps_xy = pose_results.keypoints.xy
                    kps_conf = pose_results.keypoints.conf
                    kps_xy_np = (
                        kps_xy.cpu().numpy()
                        if hasattr(kps_xy, "cpu")
                        else np.asarray(kps_xy)
                    )
                    kps_conf_np = (
                        kps_conf.cpu().numpy()
                        if kps_conf is not None and hasattr(kps_conf, "cpu")
                        else (np.asarray(kps_conf) if kps_conf is not None else None)
                    )

            infer_ms = (time.perf_counter() - t0) * 1000.0
            result = InferResult(
                cam_name=cam_name,
                seq=job.seq,
                inv_scale=job.inv_scale,
                track_frame=track_frame,
                kps_xy_np=kps_xy_np,
                kps_conf_np=kps_conf_np,
                infer_ms=infer_ms,
            )

            if out_queue.full():
                try:
                    out_queue.get_nowait()
                except Empty:
                    pass
            try:
                out_queue.put(result, block=False)
            except Exception:
                pass


def _worker_entry(cam_name, in_q, out_q, stop_event):
    """spawn 入口：捕获异常避免静默退出。"""
    try:
        camera_infer_worker(cam_name, in_q, out_q, stop_event)
    except Exception:
        traceback.print_exc()
