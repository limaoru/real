"""将任意运行时对象转为可 JSON 序列化的结构。"""
from __future__ import annotations

import base64
import json
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np


def to_json_safe(obj: Any) -> Any:
    """递归转换 bytes / numpy / Path / deque 等为 JSON 兼容类型。"""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8")
        except UnicodeDecodeError:
            return base64.b64encode(obj).decode("ascii")
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {to_json_safe(k): to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, deque, set)):
        return [to_json_safe(x) for x in obj]
    return str(obj)


def dumps_json(obj: Any, **kwargs) -> str:
    return json.dumps(to_json_safe(obj), **kwargs)
