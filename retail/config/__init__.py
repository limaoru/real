"""配置：功能开关、摄像头、检测模型。"""

from retail.config.cameras import (
    CAMERA_CONFIGS,
    CONFIG_PATH,
    get_cam_lines,
    get_cam_zones,
    load_zone_config,
    save_zone_config,
)
from retail.config.models import (
    BAG_CLASS_IDS,
    CLASS_MAPPING,
    INFER_IMGSZ,
    MODEL_PRESETS,
    POSE_MODEL,
    POSE_ZH,
    SEG_MODEL,
    SKELETON,
    TARGET_CLASSES,
)
from retail.config.settings import *  # noqa: F403

__all__ = [
    "CAMERA_CONFIGS",
    "CONFIG_PATH",
    "get_cam_lines",
    "get_cam_zones",
    "load_zone_config",
    "save_zone_config",
    "BAG_CLASS_IDS",
    "CLASS_MAPPING",
    "INFER_IMGSZ",
    "MODEL_PRESETS",
    "POSE_MODEL",
    "POSE_ZH",
    "SEG_MODEL",
    "SKELETON",
    "TARGET_CLASSES",
]
