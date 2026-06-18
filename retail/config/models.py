"""YOLO 检测与姿态相关常量。"""

from retail.config.settings import INFER_IMGSZ, MODEL_SIZE

TARGET_CLASSES = [0, 24, 26, 28, 39, 41, 67]

CLASS_MAPPING = {
    0: ("Person", (0, 255, 0)),
    24: ("Backpack", (255, 165, 0)),
    26: ("Handbag", (255, 0, 255)),
    28: ("Suitcase", (0, 165, 255)),
    39: ("Bottle", (0, 255, 255)),
    41: ("Cup", (0, 128, 255)),
    67: ("Phone", (0, 0, 255)),
}

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

MODEL_PRESETS = {
    "n": {"seg": "yolo11n-seg.pt", "pose": "yolo11n-pose.pt", "imgsz": 640},
    "s": {"seg": "yolo11s-seg.pt", "pose": "yolo11s-pose.pt", "imgsz": 768},
    "m": {"seg": "yolo11m-seg.pt", "pose": "yolo11m-pose.pt", "imgsz": 960},
    "l": {"seg": "yolo11l-seg.pt", "pose": "yolo11l-pose.pt", "imgsz": 1152},
    "x": {"seg": "yolo11x-seg.pt", "pose": "yolo11x-pose.pt", "imgsz": 1280},
}

_active = MODEL_PRESETS.get(MODEL_SIZE, MODEL_PRESETS["x"])
SEG_MODEL = _active["seg"]
POSE_MODEL = _active["pose"]
# INFER_IMGSZ 由 settings.py 统一配置（不再随 MODEL_SIZE 档位自动升高）

POSE_ZH = {"Standing": "站", "Sitting": "坐", "Lying": "躺", "Unknown": "未知"}
BAG_CLASS_IDS = {24, 26, 28}
