"""店铺监控共享配置：摄像头、区域、越线 JSON 读写"""
import json
from pathlib import Path
from typing import Optional

from retail.paths import ZONE_CONFIG_PATH

ROOT_DIR = ZONE_CONFIG_PATH.parent
CONFIG_PATH = ZONE_CONFIG_PATH

CAMERA_CONFIGS = {
    "Cam_231_Counter": {
        "rtsp": "rtsp://Mikhail:999ookjdf@127.0.0.1:10201/stream2",
        "width": 1280,
        "height": 720,
    },
    "Cam_104_Specialty": {
        "rtsp": "rtsp://Mikhail:999ookjdf@127.0.0.1:10200/stream2",
        "width": 1280,
        "height": 720,
    },
}

ZONE_TEMPLATES = {
    "default": [
        {"name": "入口区", "poly": [[0.0, 0.65], [0.45, 0.65], [0.45, 1.0], [0.0, 1.0]], "color": [0, 255, 255]},
        {"name": "收银区", "poly": [[0.45, 0.35], [1.0, 0.35], [1.0, 0.85], [0.45, 0.85]], "color": [255, 200, 0]},
        {"name": "等候区", "poly": [[0.0, 0.35], [0.45, 0.35], [0.45, 0.65], [0.0, 0.65]], "color": [255, 0, 255]},
    ],
    "Cam_231_Counter": [
        {"name": "入口", "poly": [[0.05, 0.7], [0.5, 0.7], [0.5, 1.0], [0.05, 1.0]], "color": [0, 255, 255]},
        {"name": "收银台", "poly": [[0.5, 0.3], [0.95, 0.3], [0.95, 0.75], [0.5, 0.75]], "color": [255, 200, 0]},
    ],
    "Cam_104_Specialty": [
        {"name": "货架区", "poly": [[0.1, 0.2], [0.9, 0.2], [0.9, 0.6], [0.1, 0.6]], "color": [0, 200, 255]},
        {"name": "体验区", "poly": [[0.1, 0.6], [0.9, 0.6], [0.9, 1.0], [0.1, 1.0]], "color": [200, 100, 255]},
    ],
}

LINE_TEMPLATES = {
    "default": [{"name": "进出线", "p1": [0.25, 0.65], "p2": [0.75, 0.65]}],
    "Cam_231_Counter": [{"name": "进店线", "p1": [0.1, 0.68], "p2": [0.55, 0.68]}],
    "Cam_104_Specialty": [{"name": "进店线", "p1": [0.15, 0.62], "p2": [0.85, 0.62]}],
}

ZONE_COLORS = [(0, 255, 255), (255, 200, 0), (255, 0, 255), (0, 200, 255), (200, 100, 255)]


def default_config() -> dict:
    return {"zones": ZONE_TEMPLATES, "lines": LINE_TEMPLATES}


def load_zone_config(path: Path = CONFIG_PATH) -> dict:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    cfg = default_config()
    save_zone_config(cfg, path)
    return cfg


def save_zone_config(data: dict, path: Path = CONFIG_PATH) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _zone_list_for_cam(config: dict, cam_name: str) -> list:
    zones = config.get("zones", {})
    return zones.get(cam_name) or zones.get("default", ZONE_TEMPLATES["default"])


def _line_list_for_cam(config: dict, cam_name: str) -> list:
    lines = config.get("lines", {})
    return lines.get(cam_name) or lines.get("default", LINE_TEMPLATES["default"])


def get_cam_zones(cam_name: str, width: int, height: int, config: Optional[dict] = None) -> list:
    config = config or load_zone_config()
    result = []
    for z in _zone_list_for_cam(config, cam_name):
        poly = [(int(p[0] * width), int(p[1] * height)) for p in z["poly"]]
        color = tuple(z.get("color", ZONE_COLORS[0]))
        result.append({"name": z["name"], "poly": poly, "color": color})
    return result


def get_cam_lines(cam_name: str, width: int, height: int, config: Optional[dict] = None) -> list:
    config = config or load_zone_config()
    result = []
    for ln in _line_list_for_cam(config, cam_name):
        p1 = (int(ln["p1"][0] * width), int(ln["p1"][1] * height))
        p2 = (int(ln["p2"][0] * width), int(ln["p2"][1] * height))
        result.append({"name": ln["name"], "p1": p1, "p2": p2})
    return result


def pixel_poly_to_norm(poly: list, width: int, height: int) -> list:
    return [[round(x / width, 4), round(y / height, 4)] for x, y in poly]


def pixel_line_to_norm(p1, p2, width: int, height: int) -> dict:
    return {
        "p1": [round(p1[0] / width, 4), round(p1[1] / height, 4)],
        "p2": [round(p2[0] / width, 4), round(p2[1] / height, 4)],
    }
