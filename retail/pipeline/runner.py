"""零售视觉分析主流水线。"""
from __future__ import annotations

import os

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import time
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from retail.analytics.brain import StoreBrain
from retail.analytics.cam import CameraAnalytics
from retail.config.cameras import CAMERA_CONFIGS, load_zone_config
from retail.config.models import (
    BAG_CLASS_IDS,
    CLASS_MAPPING,
    INFER_IMGSZ,
    POSE_MODEL,
    POSE_ZH,
    SEG_MODEL,
    TARGET_CLASSES,
)
from retail.config.settings import (
    ANALYSIS_SCALE,
    DAILY_REPORT_HOUR,
    ENABLE_ANALYSIS_DOWNSCALE,
    ENABLE_BAG_ASSOCIATION,
    ENABLE_DAILY_REPORT,
    ENABLE_EVENT_CLIPS,
    ENABLE_FACE_BLUR,
    ENABLE_GROUP_DETECT,
    ENABLE_OCR_SHELF,
    ENABLE_OPEN_VOCAB,
    ENABLE_QUEUE_DETECT,
    ENABLE_SKIN_DETECT,
    ENABLE_STAFF_EXCLUDE_COUNT,
    ENABLE_VLM_ANALYSIS,
    ENABLE_WEB_DASHBOARD,
    GROUP_DISTANCE,
    HEATMAP_DECAY,
    INFER_DEVICE,
    MODEL_SIZE,
    POSE_EVERY_N_FRAMES,
    QUEUE_MIN_PEOPLE,
    SKIN_MIN_RATIO,
    STORE_NAME,
    WEB_EXPORT_INTERVAL,
)
from retail.data.export import export_live_state
from retail.data.report import generate_daily_report
from retail.geometry import box_contains_point, detect_groups, point_in_polygon, scale_boxes
from retail.paths import LOG_DIR
from retail.services.active_learning import ActiveLearningExporter
from retail.services.clips import ClipRecorder
from retail.services.digital_twin import DigitalTwin
from retail.services.multiview import MultiViewFusion
from retail.services.vlm import VLMBridge
from retail.services.webhook import push_alert_webhook
from retail.ui.overlay import build_heatmap_views, draw_analytics_panel, draw_fps_hud
from retail.ui.text import flush_text, queue_text
from retail.vision.advanced import VisionAdvancedHub, blur_faces
from retail.vision.contours import draw_person_contour
from retail.vision.pose import classify_pose, draw_skeleton
from retail.vision.skin import detect_and_draw_skin
from retail.vision.stream import RTSPStreamReader

load_zone_config()


def run():
    if not torch.cuda.is_available():
        print("❌ 未检测到可用的 NVIDIA CUDA 环境！")
        return
    
    print(f"🔥 {STORE_NAME} 零售视界全功能启动！模型: {MODEL_SIZE} | 分割: {SEG_MODEL} | 姿态: {POSE_MODEL}")
    print("📐 区域标定: python -m retail zones  |  📊 仪表盘: python -m retail dashboard")
    print("🧠 配置见 retail/config/settings.py")
    model = YOLO(SEG_MODEL)
    pose_model = YOLO(POSE_MODEL)
    model_label = f"11{MODEL_SIZE}"
    brain = StoreBrain()
    vision_hub = VisionAdvancedHub()
    vlm = VLMBridge()
    multiview = MultiViewFusion()
    twin = DigitalTwin()
    active_learn = ActiveLearningExporter()
    clip_recorders = {}
    frame_counters = {}

    readers = {}
    heatmaps = {}
    customer_counts = {}
    counted_ids = {}
    fps_state = {}
    cam_analytics = {}

    for cam_name, config in CAMERA_CONFIGS.items():
        cv2.namedWindow(f"Store Analytics Platform - {cam_name}", cv2.WINDOW_NORMAL)
        cv2.namedWindow(f"Heatmap - {cam_name}", cv2.WINDOW_NORMAL)
        reader = RTSPStreamReader(cam_name, config["rtsp"], config["width"], config["height"])
        reader.daemon = True
        reader.start()
        readers[cam_name] = reader
        
        heatmaps[cam_name] = np.zeros((config["height"], config["width"]), dtype=np.float32)
        customer_counts[cam_name] = 0
        counted_ids[cam_name] = set()
        fps_state[cam_name] = {"fps": 0.0}
        cam_analytics[cam_name] = CameraAnalytics(cam_name, config["width"], config["height"])
        clip_recorders[cam_name] = ClipRecorder(cam_name)
        frame_counters[cam_name] = 0

    LOG_DIR.mkdir(exist_ok=True)
    last_web_export = 0.0
    metrics_by_cam = {}
    brain_summaries = {}
    gid_labels = {}
    last_report_date = None

    def push_alert_hook(analytics, msg):
        analytics.push_alert(msg)
        push_alert_webhook(msg, STORE_NAME)
        if brain.db:
            brain.db.log_event(analytics.cam_name, "alert", msg)
        if ENABLE_EVENT_CLIPS:
            rec = clip_recorders.get(analytics.cam_name)
            if rec:
                fps = fps_state.get(analytics.cam_name, {}).get("fps", 10) or 10
                rec.trigger(msg, fps=fps)

    with torch.inference_mode():
        while True:
            pending_frames = []
            for cam_name, reader in readers.items():
                if not reader.frame_queue.empty():
                    pending_frames.append((cam_name, reader.frame_queue.get()))

            for cam_name, frame in pending_frames:
                reader = readers[cam_name]
                t0 = time.perf_counter()
                frame_counters[cam_name] += 1
                if ENABLE_EVENT_CLIPS:
                    clip_recorders[cam_name].push_frame(frame)
                analytics = cam_analytics[cam_name]
                accum_heatmap = heatmaps[cam_name]
                frame_w = reader.width
                frame_h = reader.height
                accum_heatmap *= HEATMAP_DECAY

                infer_frame = frame
                inv_scale = 1.0
                if ENABLE_ANALYSIS_DOWNSCALE and ANALYSIS_SCALE < 1.0:
                    inv_scale = ANALYSIS_SCALE
                    infer_frame = cv2.resize(
                        frame, (int(frame_w * inv_scale), int(frame_h * inv_scale))
                    )

                current_metrics = {name: 0 for name, _ in CLASS_MAPPING.values()}
                current_metrics.update({"standing": 0, "sitting": 0, "lying": 0, "groups": 0, "queue": 0, "bags": 0})
                person_centers = []
                person_boxes = {}
                object_boxes = []

                results = model.track(
                    source=infer_frame, persist=True, device=INFER_DEVICE, verbose=False, imgsz=INFER_IMGSZ,
                    classes=TARGET_CLASSES, tracker="bytetrack.yaml", conf=0.35, iou=0.4
                )

                pose_by_foot = {}
                kps_list_for_blur = []

                if results[0].boxes is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    if inv_scale != 1.0:
                        boxes = scale_boxes(boxes, inv_scale)
                    clss = results[0].boxes.cls.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else [None] * len(clss)
                    masks_xy = results[0].masks.xy if results[0].masks is not None else None

                    for idx, (box, cls_id, track_id) in enumerate(zip(boxes, clss, track_ids)):
                        if cls_id not in CLASS_MAPPING:
                            continue

                        label, color = CLASS_MAPPING[cls_id]
                        current_metrics[label] += 1
                        x1, y1, x2, y2 = box
                        id_str = f" ID:{track_id}" if track_id is not None else ""

                        if cls_id == 0:
                            foot = (int((x1 + x2) / 2), int(y2))
                            person_centers.append(foot)
                            if track_id is not None:
                                person_boxes[track_id] = box
                            mask_xy = masks_xy[idx] if masks_xy is not None and idx < len(masks_xy) else None
                            label_pos = draw_person_contour(frame, mask_xy, color)
                            if label_pos is None:
                                label_pos = (int(x1), int(y1))
                            skin_ratio = 0.0
                            if ENABLE_SKIN_DETECT:
                                skin_ratio, label_pos = detect_and_draw_skin(frame, mask_xy, label_pos=label_pos)
                            queue_text(frame, f"{label}{id_str}", label_pos[0], label_pos[1] - 10, color, 18)
                            if ENABLE_SKIN_DETECT and skin_ratio >= SKIN_MIN_RATIO:
                                queue_text(frame, f"肤色:{skin_ratio*100:.0f}%", label_pos[0], label_pos[1] + 28,
                                           (0, 0, 255), 18)
                            if track_id is not None and track_id not in counted_ids[cam_name]:
                                if not (ENABLE_STAFF_EXCLUDE_COUNT and brain.is_staff(track_id)):
                                    counted_ids[cam_name].add(track_id)
                                    customer_counts[cam_name] += 1
                                    hour_key = datetime.now().strftime("%H:00")
                                    analytics.hourly[hour_key] += 1
                                    if brain.db:
                                        brain.db.bump_hourly(cam_name)
                            cx, cy = foot
                            radius = max(22, int(22 * frame_w / 1280))
                            cv2.circle(accum_heatmap, (cx, cy), radius=radius, color=0.4, thickness=-1)
                        else:
                            object_boxes.append((cls_id, box))
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                            queue_text(frame, f"{label}{id_str}", int(x1), int(y1) - 10, color, 18)

                run_pose = (frame_counters[cam_name] % max(1, POSE_EVERY_N_FRAMES) == 0)
                if current_metrics["Person"] > 0 and run_pose:
                    pose_results = pose_model(infer_frame, verbose=False, device=INFER_DEVICE, conf=0.35, imgsz=INFER_IMGSZ)[0]
                    if pose_results.keypoints is not None:
                        kps_xy = pose_results.keypoints.xy
                        kps_conf = pose_results.keypoints.conf
                        kps_xy_np = kps_xy.cpu().numpy() if hasattr(kps_xy, "cpu") else np.asarray(kps_xy)
                        if inv_scale != 1.0:
                            kps_xy_np = kps_xy_np / inv_scale
                        kps_conf_np = kps_conf.cpu().numpy() if kps_conf is not None and hasattr(kps_conf, "cpu") else (
                            np.asarray(kps_conf) if kps_conf is not None else None
                        )
                        kps_list_for_blur = list(kps_xy_np)
                        for pi, kp in enumerate(kps_xy_np):
                            conf = kps_conf_np[pi] if kps_conf_np is not None else None
                            status_en = classify_pose(kp, conf)
                            status = POSE_ZH.get(status_en, "未知")
                            if status_en == "Standing":
                                current_metrics["standing"] += 1
                            elif status_en == "Sitting":
                                current_metrics["sitting"] += 1
                            elif status_en == "Lying":
                                current_metrics["lying"] += 1
                            nose_x, nose_y = int(kp[0][0]), int(kp[0][1])
                            if nose_x > 0 and nose_y > 0:
                                queue_text(frame, status, nose_x, max(20, nose_y - 18), (255, 255, 255), 22)
                            if kp[11][1] > 0 and kp[12][1] > 0:
                                foot_est = (int((kp[11][0] + kp[12][0]) / 2), int(max(kp[11][1], kp[12][1]) + 40))
                                pose_by_foot[foot_est] = status
                        draw_skeleton(frame, kps_xy_np, kps_conf_np,
                                      color=CLASS_MAPPING[0][1], thickness=3)

                if ENABLE_FACE_BLUR and kps_list_for_blur:
                    blur_faces(frame, kps_list_for_blur)

                # 开放词汇 / OCR
                ov_hits = vision_hub.open_vocab.detect(frame)
                for hit in ov_hits[:5]:
                    bx = hit["box"]
                    queue_text(frame, hit["label"][:12], int(bx[0]), int(bx[1]) - 8, (255, 128, 0), 14)
                    if brain.db:
                        brain.db.log_open_vocab(cam_name, hit["label"], hit["conf"])
                if ENABLE_OCR_SHELF:
                    shelf_z = next((z for z in analytics.zones if "货架" in z["name"]), None)
                    roi = None
                    if shelf_z:
                        xs = [p[0] for p in shelf_z["poly"]]
                        ys = [p[1] for p in shelf_z["poly"]]
                        roi = (min(xs), min(ys), max(xs), max(ys))
                    texts = vision_hub.ocr.scan(frame, roi)
                    for i, txt in enumerate(texts[:3]):
                        queue_text(frame, txt[:20], 400, 80 + i * 22, (200, 255, 200), 14)
                        if brain.db:
                            brain.db.log_ocr(cam_name, txt)

                lying_ids = set()
                for track_id, box in person_boxes.items():
                    foot = (int((box[0] + box[2]) / 2), int(box[3]))
                    pose_status = "未知"
                    for pf, ps in pose_by_foot.items():
                        if abs(pf[0] - foot[0]) < 80 and abs(pf[1] - foot[1]) < 120:
                            pose_status = ps
                            break
                    if pose_status == "躺":
                        lying_ids.add(track_id)
                    analytics.update_person(track_id, foot, pose_status)
                    zones_inside = list(analytics.tracks.get(track_id, {}).get("zones", []))
                    in_shelf = any("货架" in z or "体验" in z for z in zones_inside)
                    speed_px = 0.0
                    prev = brain.last_positions.get((cam_name, track_id))
                    if prev:
                        speed_px = ((foot[0] - prev[0]) ** 2 + (foot[1] - prev[1]) ** 2) ** 0.5
                    pinfo = brain.process_person(
                        cam_name, track_id, box, foot, frame, zones_inside, pose_status
                    )
                    if vision_hub.behavior and kps_list_for_blur:
                        best_kp, best_d = None, 1e9
                        for kp in kps_list_for_blur:
                            if kp[11][1] > 0 and kp[12][1] > 0:
                                est = (int((kp[11][0] + kp[12][0]) / 2), int(max(kp[11][1], kp[12][1]) + 40))
                            else:
                                est = (int(kp[0][0]), int(kp[0][1]))
                            d = (est[0] - foot[0]) ** 2 + (est[1] - foot[1]) ** 2
                            if d < best_d:
                                best_d, best_kp = d, kp
                        if best_kp is not None and best_d < 120 ** 2:
                            beh = vision_hub.behavior.update(
                                cam_name, track_id, best_kp, None, in_shelf, speed_px
                            )
                            if beh and beh != "idle":
                                brain.record_behavior(cam_name, track_id, beh, pinfo.get("gid"))
                                queue_text(frame, vision_hub.behavior.zh(beh), foot[0] + 15, foot[1] - 30,
                                           (180, 255, 180), 14)
                    gid = pinfo.get("gid")
                    if gid:
                        gid_labels[track_id] = gid
                        queue_text(frame, f"G{gid}", foot[0] - 10, foot[1] + 20, (255, 180, 0), 16)

                track_positions = {tid: (int((person_boxes[tid][0]+person_boxes[tid][2])/2), int(person_boxes[tid][3]))
                                   for tid in person_boxes}
                multiview.update(cam_name, track_positions, frame_w, frame_h)
                twin.ingest_frame(track_positions, frame_w, frame_h, analytics.zones)
                brain.process_frame_end(cam_name, track_positions, lying_ids)
                brain.run_events(cam_name, current_metrics, analytics,
                                 lambda m: push_alert_hook(analytics, m))
                brain.maybe_persist(cam_name, current_metrics, analytics)
                brain_summaries[cam_name] = brain.get_cam_summary(cam_name, current_metrics, analytics)
                brain.push_attribution(current_metrics, brain_summaries[cam_name].get("conversion_pct", 0))

                if ENABLE_VLM_ANALYSIS and frame_counters[cam_name] % 150 == 0:
                    insight = vlm.maybe_analyze(frame, current_metrics, list(analytics.alerts))
                    if insight and brain.db:
                        brain.db.log_vlm_insight(insight)

                conf_tensor = (
                    results[0].boxes.conf
                    if results[0].boxes is not None and results[0].boxes.conf is not None
                    else None
                )
                if conf_tensor is not None and conf_tensor.numel() > 0:
                    low_conf = float(conf_tensor.min().cpu().item())
                    active_learn.maybe_export(frame, cam_name, "low_conf", low_conf)

                analytics.prune_stale_tracks()
                analytics.refresh_zone_occupancy()
                analytics.peak_occupancy = max(analytics.peak_occupancy, current_metrics["Person"])

                if ENABLE_GROUP_DETECT and len(person_centers) >= 2:
                    current_metrics["groups"] = detect_groups(person_centers)

                if ENABLE_QUEUE_DETECT:
                    queue_zone = next((z for z in analytics.zones if "等候" in z["name"] or "收银" in z["name"]), None)
                    if queue_zone:
                        q_cnt = sum(1 for c in person_centers if point_in_polygon(c[0], c[1], queue_zone["poly"]))
                        current_metrics["queue"] = q_cnt
                        if q_cnt >= QUEUE_MIN_PEOPLE and not analytics.queue_alert_active:
                            analytics.queue_alert_active = True
                            analytics.push_alert(f"排队告警 {q_cnt}人")
                        elif q_cnt < QUEUE_MIN_PEOPLE:
                            analytics.queue_alert_active = False

                if ENABLE_BAG_ASSOCIATION:
                    bag_cnt = 0
                    for cls_id, obox in object_boxes:
                        if cls_id not in BAG_CLASS_IDS:
                            continue
                        bx, by = (obox[0] + obox[2]) / 2, (obox[1] + obox[3]) / 2
                        for pbox in person_boxes.values():
                            if box_contains_point(pbox, bx, by):
                                bag_cnt += 1
                                break
                    current_metrics["bags"] = bag_cnt

                analytics.draw_zones(frame)
                analytics.draw_lines(frame)
                analytics.draw_trails(frame)
                brain.draw_direction_arrows(frame)
                flush_text(frame)

                overlay_frame, heatmap_panel = build_heatmap_views(frame, accum_heatmap, frame_w)

                dt = time.perf_counter() - t0
                if dt > 0:
                    instant_fps = 1.0 / dt
                    state = fps_state[cam_name]
                    state["fps"] = state["fps"] * 0.85 + instant_fps * 0.15 if state["fps"] > 0 else instant_fps

                draw_analytics_panel(overlay_frame, analytics, customer_counts[cam_name], current_metrics,
                                     brain_summaries.get(cam_name))
                draw_fps_hud(overlay_frame, fps_state[cam_name]["fps"], model_label)
                flush_text(overlay_frame)
                analytics.maybe_log_csv(current_metrics)
                analytics.maybe_snapshot_heatmap(heatmap_panel)
                metrics_by_cam[cam_name] = dict(current_metrics)

                cv2.imshow(f"Store Analytics Platform - {cam_name}", overlay_frame)
                cv2.imshow(f"Heatmap - {cam_name}", heatmap_panel)

            now = time.time()
            if ENABLE_WEB_DASHBOARD and now - last_web_export >= WEB_EXPORT_INTERVAL:
                twin_payload = twin.export()
                extras = {
                    "vlm": vlm.get_latest(),
                    "clips": ClipRecorder.list_clips(10),
                    "twin": twin_payload,
                    "multiview": multiview.export(),
                    "attribution": brain.attribution.export(),
                    "active_learning": active_learn.list_pending(8),
                }
                export_live_state(cam_analytics, customer_counts, fps_state, metrics_by_cam, brain, brain_summaries, extras)
                last_web_export = now

            if ENABLE_DAILY_REPORT and datetime.now().hour == DAILY_REPORT_HOUR:
                today = datetime.now().date()
                if last_report_date != today:
                    try:
                        path = generate_daily_report(brain.db)
                        print(f"📄 日报已生成: {path}")
                        last_report_date = today
                    except OSError as e:
                        print(f"日报生成失败: {e}")

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for reader in readers.values(): reader.stop()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    run()
