"""数字孪生：店铺占用网格 + 动线仿真数据导出"""
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from retail.config.settings import ENABLE_DIGITAL_TWIN, STORE_AREA_SQM, STORE_NAME
from retail.data.serialize import dumps_json
from retail.paths import TWIN_STATE_PATH as TWIN_STATE


class DigitalTwin:
    def __init__(self, grid_w: int = 24, grid_h: int = 16):
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.occupancy = np.zeros((grid_h, grid_w), dtype=np.float32)
        self.flow_vectors: list[dict] = []

    def ingest_frame(self, track_positions: dict, frame_w: int, frame_h: int, zones: list):
        if not ENABLE_DIGITAL_TWIN:
            return
        for tid, (x, y) in track_positions.items():
            gx = min(self.grid_w - 1, max(0, int(x / max(1, frame_w) * self.grid_w)))
            gy = min(self.grid_h - 1, max(0, int(y / max(1, frame_h) * self.grid_h)))
            self.occupancy[gy, gx] += 1.0
        self.occupancy *= 0.992

    def simulate_checkout_move(self, from_gx: int, to_gx: int) -> dict:
        """简易仿真：移动收银台对热点的影响（规划用）"""
        if not ENABLE_DIGITAL_TWIN:
            return {}
        occ = self.occupancy.copy()
        shift = min(3, max(-3, to_gx - from_gx))
        occ = np.roll(occ, shift, axis=1)
        return {
            "scenario": "checkout_relocate",
            "shift_cols": shift,
            "peak_before": float(self.occupancy.max()),
            "peak_after": float(occ.max()),
            "hint": "右移收银台可能缓解入口侧堆积" if shift > 0 else "左移可能缩短货架到收银路径",
        }

    def export(self) -> dict:
        hot = []
        if self.occupancy.max() > 0.5:
            ys, xs = np.where(self.occupancy > self.occupancy.max() * 0.55)
            for x, y in zip(xs[:30], ys[:30]):
                hot.append({"x": int(x), "y": int(y), "heat": round(float(self.occupancy[y, x]), 2)})
        payload = {
            "store": STORE_NAME,
            "area_sqm": STORE_AREA_SQM,
            "grid": [self.grid_w, self.grid_h],
            "hotspots": hot,
            "updated": datetime.now().isoformat(timespec="seconds"),
            "simulation": self.simulate_checkout_move(6, 10),
        }
        TWIN_STATE.parent.mkdir(parents=True, exist_ok=True)
        TWIN_STATE.write_text(dumps_json(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return payload

    @staticmethod
    def load() -> dict:
        if not TWIN_STATE.exists():
            return {}
        try:
            return json.loads(TWIN_STATE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
