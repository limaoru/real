"""视觉进阶：Deep ReID、行为识别、开放词汇、OCR、人脸模糊、SAM 精修"""
import time
from collections import defaultdict, deque
from typing import Optional

import cv2
import numpy as np

from retail.config.settings import (
    DEEP_REID_THRESHOLD,
    ENABLE_BEHAVIOR_RECOGNITION,
    ENABLE_DEEP_REID,
    ENABLE_FACE_BLUR,
    ENABLE_OCR_SHELF,
    ENABLE_OPEN_VOCAB,
    ENABLE_SAM_REFINE,
    FACE_BLUR_STRENGTH,
    OCR_INTERVAL_SEC,
    OPEN_VOCAB_INTERVAL,
    OPEN_VOCAB_MODEL,
    OPEN_VOCAB_PROMPTS,
    SAM_MODEL,
)
from retail.vision.infer_backend import load_yolo_model, torch_device, use_fp16, yolo_kwargs

_BEHAVIOR_ZH = {
    "browse_shelf": "浏览货架",
    "pick_up": "拿取",
    "put_back": "放回",
    "carry": "携带",
    "try_on": "试穿",
    "walk_fast": "快走",
    "idle": "停留",
}


class DeepReID:
    """基于 CNN 嵌入的跨镜 ReID（比颜色直方图稳）"""

    def __init__(self):
        self.next_gid = 1
        self.track_map: dict[tuple, int] = {}
        self.gallery: deque = deque(maxlen=300)
        self._embedder = None
        self._device = None

    def _load_embedder(self):
        if self._embedder is not None:
            return
        import torch
        from torchvision import models, transforms

        self._device = torch_device()
        net = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        net.classifier = torch.nn.Identity()
        net.eval().to(self._device)
        if use_fp16():
            net.half()
        self._embedder = (
            net,
            transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 64)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]),
        )

    def _embed(self, frame, box) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = map(int, box)
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 - x1 < 12 or y2 - y1 < 20:
            return None
        crop = frame[y1:y2, x1:x2]
        if self._embedder is None:
            self._load_embedder()
        import torch
        net, tfm = self._embedder
        dtype = torch.float16 if use_fp16() else torch.float32
        with torch.inference_mode():
            t = tfm(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self._device, dtype=dtype)
            vec = net(t).float().cpu().numpy().flatten()
        norm = np.linalg.norm(vec) + 1e-6
        return (vec / norm).astype(np.float32)

    @staticmethod
    def _sim(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            return 0.0
        return float(np.dot(a, b))

    def assign(self, cam: str, track_id: int, frame, box) -> Optional[int]:
        key = (cam, track_id)
        if key in self.track_map:
            return self.track_map[key]
        emb = self._embed(frame, box)
        if emb is None:
            return None
        now = time.time()
        best_gid, best_score = None, DEEP_REID_THRESHOLD
        area = (box[2] - box[0]) * (box[3] - box[1])
        for g in self.gallery:
            if now - g["time"] > 240 or g["cam"] == cam:
                continue
            score = self._sim(emb, g["emb"])
            ar = min(area, g["area"]) / max(area, g["area"])
            if score > best_score and ar > 0.3:
                best_score, best_gid = score, g["gid"]
        gid = best_gid or self.next_gid
        if best_gid is None:
            self.next_gid += 1
        self.track_map[key] = gid
        self.gallery.append({"gid": gid, "cam": cam, "emb": emb, "area": area, "time": now})
        return gid


class BehaviorRecognizer:
    """姿态序列 + 区域 → 零售行为语义"""

    def __init__(self):
        self.wrist_hist: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=12))
        self.last_behavior: dict[tuple, str] = {}
        self.counts: dict[str, int] = defaultdict(int)

    def update(
        self, cam: str, track_id: int, kp, conf, in_shelf: bool, speed_px: float
    ) -> Optional[str]:
        if not ENABLE_BEHAVIOR_RECOGNITION or kp is None:
            return None
        key = (cam, track_id)
        lw, rw = kp[9], kp[10]
        ls, rs = kp[5], kp[6]
        if lw[1] <= 0 or rw[1] <= 0 or ls[1] <= 0:
            return None
        shoulder_y = (ls[1] + rs[1]) / 2
        wrist_y = (lw[1] + rw[1]) / 2
        wrist_x = (lw[0] + rw[0]) / 2
        self.wrist_hist[key].append((wrist_x, wrist_y, shoulder_y))

        behavior = "idle"
        if speed_px > 70:
            behavior = "walk_fast"
        elif in_shelf:
            if wrist_y < shoulder_y - 30:
                behavior = "pick_up"
            elif len(self.wrist_hist[key]) >= 4:
                ys = [p[1] for p in self.wrist_hist[key]]
                if max(ys) - min(ys) > 40:
                    behavior = "put_back"
                else:
                    behavior = "browse_shelf"
            else:
                behavior = "browse_shelf"
        prev = self.last_behavior.get(key)
        if behavior != prev:
            self.counts[behavior] += 1
            self.last_behavior[key] = behavior
        return behavior

    def zh(self, behavior: str) -> str:
        return _BEHAVIOR_ZH.get(behavior, behavior)

    def get_counts(self) -> dict[str, int]:
        return dict(self.counts)


class OpenVocabDetector:
    """YOLO-World 开放词汇检测（缺货/货架物体）"""

    def __init__(self):
        self.model = None
        self.last_run = 0.0
        self.last_hits: list[dict] = []

    def _ensure(self):
        if self.model is not None or not ENABLE_OPEN_VOCAB:
            return
        try:
            self.model = load_yolo_model(OPEN_VOCAB_MODEL)
            self.model.set_classes(OPEN_VOCAB_PROMPTS)
        except Exception as e:
            print(f"⚠️ 开放词汇模型未加载: {e}")
            self.model = False

    def detect(self, frame) -> list[dict]:
        if not ENABLE_OPEN_VOCAB:
            return []
        now = time.time()
        if now - self.last_run < OPEN_VOCAB_INTERVAL:
            return self.last_hits
        self._ensure()
        if not self.model:
            return []
        self.last_run = now
        try:
            res = self.model(frame, **yolo_kwargs(conf=0.25))[0]
            hits = []
            if res.boxes is not None:
                for box, cls_id, conf in zip(
                    res.boxes.xyxy.cpu().numpy(),
                    res.boxes.cls.cpu().numpy().astype(int),
                    res.boxes.conf.cpu().numpy(),
                ):
                    label = OPEN_VOCAB_PROMPTS[int(cls_id)] if int(cls_id) < len(OPEN_VOCAB_PROMPTS) else "obj"
                    hits.append({"label": label, "box": box.tolist(), "conf": float(conf)})
            self.last_hits = hits
            return hits
        except Exception:
            return self.last_hits


class ShelfOCR:
    """价签 OCR（EasyOCR 可选，无则跳过）"""

    def __init__(self):
        self.reader = None
        self.last_run = 0.0
        self.last_texts: list[str] = []

    def _ensure(self):
        if self.reader is not None or not ENABLE_OCR_SHELF:
            return
        try:
            import easyocr
            self.reader = easyocr.Reader(["ch_sim", "en"], gpu=False, verbose=False)
        except Exception:
            self.reader = False

    def scan(self, frame, roi: Optional[tuple] = None) -> list[str]:
        if not ENABLE_OCR_SHELF:
            return []
        now = time.time()
        if now - self.last_run < OCR_INTERVAL_SEC:
            return self.last_texts
        self._ensure()
        if not self.reader:
            return []
        self.last_run = now
        img = frame
        if roi:
            x1, y1, x2, y2 = map(int, roi)
            img = frame[y1:y2, x1:x2]
        try:
            results = self.reader.readtext(img, detail=0, paragraph=True)
            self.last_texts = [t for t in results if len(t) >= 2][:8]
            return self.last_texts
        except Exception:
            return self.last_texts


class SAMRefiner:
    """可选 SAM 精修轮廓"""

    def __init__(self):
        self.model = None

    def _ensure(self):
        if self.model is not None or not ENABLE_SAM_REFINE:
            return
        try:
            from ultralytics import SAM
            self.model = SAM(SAM_MODEL)
        except Exception:
            self.model = False

    def refine_mask(self, frame, box) -> Optional[np.ndarray]:
        if not ENABLE_SAM_REFINE:
            return None
        self._ensure()
        if not self.model:
            return None
        try:
            x1, y1, x2, y2 = map(int, box)
            res = self.model(
                frame, bboxes=[[x1, y1, x2, y2]], **yolo_kwargs(),
            )[0]
            if res.masks is not None and len(res.masks.data):
                return res.masks.data[0].cpu().numpy()
        except Exception:
            pass
        return None


def blur_faces(frame, pose_keypoints_list) -> int:
    """隐私：根据鼻子/眼关键点模糊面部"""
    if not ENABLE_FACE_BLUR:
        return 0
    blurred = 0
    h, w = frame.shape[:2]
    for kp in pose_keypoints_list:
        if kp is None or len(kp) < 5:
            continue
        nose, le, re = kp[0], kp[1], kp[2]
        if nose[0] <= 0 or nose[1] <= 0:
            continue
        cx, cy = int(nose[0]), int(nose[1])
        eye_dist = 40
        if le[0] > 0 and re[0] > 0:
            eye_dist = max(30, int(abs(re[0] - le[0]) * 1.8))
        x1 = max(0, cx - eye_dist)
        y1 = max(0, cy - eye_dist)
        x2 = min(w, cx + eye_dist)
        y2 = min(h, cy + int(eye_dist * 1.2))
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (0, 0), FACE_BLUR_STRENGTH)
        blurred += 1
    return blurred


class VisionAdvancedHub:
    """统一入口"""

    def __init__(self):
        self.reid = DeepReID() if ENABLE_DEEP_REID else None
        self.behavior = BehaviorRecognizer() if ENABLE_BEHAVIOR_RECOGNITION else None
        self.open_vocab = OpenVocabDetector()
        self.ocr = ShelfOCR()
        self.sam = SAMRefiner()

    def assign_gid(self, cam, track_id, frame, box) -> Optional[int]:
        if self.reid:
            return self.reid.assign(cam, track_id, frame, box)
        return None
