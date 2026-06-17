"""多摄像头轨迹融合（归一化坐标 + 全局热力网格）"""
from collections import defaultdict
from typing import Optional

import numpy as np

from retail.config.settings import ENABLE_MULTIVIEW_FUSION, MULTIVIEW_GRID_SIZE


class MultiViewFusion:
    def __init__(self, grid_size: int = MULTIVIEW_GRID_SIZE):
        self.grid_size = grid_size
        self.global_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.cam_positions: dict[str, dict[int, tuple]] = defaultdict(dict)
        self.fused_tracks: dict[int, list] = defaultdict(list)

    def update(self, cam: str, track_positions: dict[int, tuple], frame_w: int, frame_h: int):
        if not ENABLE_MULTIVIEW_FUSION:
            return
        self.cam_positions[cam] = {}
        for tid, (x, y) in track_positions.items():
            nx = min(self.grid_size - 1, max(0, int(x / max(1, frame_w) * self.grid_size)))
            ny = min(self.grid_size - 1, max(0, int(y / max(1, frame_h) * self.grid_size)))
            self.cam_positions[cam][tid] = (nx, ny)
            self.global_grid[ny, nx] += 0.15
        self.global_grid *= 0.995

    def get_heatmap_png_bytes(self) -> Optional[bytes]:
        import cv2
        g = np.clip(self.global_grid / (self.global_grid.max() + 1e-6) * 255, 0, 255).astype(np.uint8)
        color = cv2.applyColorMap(g, cv2.COLORMAP_JET)
        ok, buf = cv2.imencode(".png", color)
        return buf.tobytes() if ok else None

    def export(self) -> dict:
        hot = []
        if self.global_grid.max() > 0.01:
            ys, xs = np.where(self.global_grid > self.global_grid.max() * 0.5)
            for x, y in zip(xs[:20], ys[:20]):
                hot.append({"gx": int(x), "gy": int(y), "v": round(float(self.global_grid[y, x]), 3)})
        return {
            "grid_size": self.grid_size,
            "hotspots": hot,
            "cam_count": len(self.cam_positions),
        }
