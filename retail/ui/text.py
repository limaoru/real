"""中文友好文字渲染（OpenCV + Pillow）。"""
from __future__ import annotations

import os

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

_TEXT_BATCH: dict[int, list] = {}
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def _get_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                _FONT_CACHE[size] = font
                return font
            except OSError:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def queue_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int] = (255, 255, 255),
    size: int = 20,
) -> None:
    """收集文字，稍后 :func:`flush_text` 一次性绘制。"""
    _TEXT_BATCH.setdefault(id(img), []).append((text, x, y, color, size))


def flush_text(img: np.ndarray) -> None:
    """将 :func:`queue_text` 收集的文字绘制到图像上。"""
    items = _TEXT_BATCH.pop(id(img), [])
    if not items:
        return

    ascii_items, cjk_items = [], []
    for item in items:
        (cjk_items if _has_cjk(item[0]) else ascii_items).append(item)

    for text, x, y, color, size in ascii_items:
        scale = max(0.35, size / 22.0)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    if not cjk_items:
        return

    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    for text, x, y, color, size in cjk_items:
        font = _get_cjk_font(size)
        bbox = draw.textbbox((0, 0), text, font=font)
        th = bbox[3] - bbox[1]
        draw.text((x, y - th), text, font=font, fill=(color[2], color[1], color[0]))
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
