#!/usr/bin/env python3
"""
区域/越线鼠标标定工具

操作说明：
  左键点击   - 添加多边形顶点 / 设置越线端点
  右键       - 完成当前多边形
  拖拽顶点   - 按住左键在顶点附近拖动调整
  Z          - 新建区域模式（输入名称）
  L          - 新建越线模式（输入名称，点两个端点）
  D          - 删除最后一个区域或越线
  S          - 保存到 zone_config.json
  N          - 切换摄像头
  R          - 从 JSON 重新加载
  Q / ESC    - 退出

运行：python3 zone_editor.py [摄像头名称]
"""
import os
import sys

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
import numpy as np

from retail.config.cameras import (
    CAMERA_CONFIGS,
    CONFIG_PATH,
    ZONE_COLORS,
    get_cam_lines,
    get_cam_zones,
    load_zone_config,
    pixel_line_to_norm,
    pixel_poly_to_norm,
    save_zone_config,
)


def _nearest_point(points, x, y, thresh=12):
    best_i, best_d = -1, thresh * thresh
    for i, (px, py) in enumerate(points):
        d = (px - x) ** 2 + (py - y) ** 2
        if d < best_d:
            best_d, best_i = d, i
    return best_i


class ZoneEditor:
    def __init__(self, cam_name: str):
        if cam_name not in CAMERA_CONFIGS:
            cam_name = list(CAMERA_CONFIGS.keys())[0]
        self.cam_name = cam_name
        self.cfg = CAMERA_CONFIGS[cam_name]
        self.w, self.h = self.cfg["width"], self.cfg["height"]
        self.config = load_zone_config()
        self.zones_norm = list(self.config.get("zones", {}).get(cam_name, []))
        self.lines_norm = list(self.config.get("lines", {}).get(cam_name, []))
        self.mode = "edit"  # edit | new_zone | new_line
        self.drawing_pts = []
        self.pending_name = ""
        self.drag = None  # (kind, index, pt_index)
        self.line_step = 0
        self.win = f"Zone Editor - {cam_name}"

    def _zones_px(self):
        return get_cam_zones(self.cam_name, self.w, self.h, {"zones": {self.cam_name: self.zones_norm}, "lines": {}})

    def _lines_px(self):
        return get_cam_lines(self.cam_name, self.w, self.h, {"lines": {self.cam_name: self.lines_norm}, "zones": {}})

    def _save(self):
        self.config.setdefault("zones", {})[self.cam_name] = self.zones_norm
        self.config.setdefault("lines", {})[self.cam_name] = self.lines_norm
        save_zone_config(self.config)
        print(f"已保存 -> {CONFIG_PATH}")

    def _reload(self):
        self.config = load_zone_config()
        self.zones_norm = list(self.config.get("zones", {}).get(self.cam_name, []))
        self.lines_norm = list(self.config.get("lines", {}).get(self.cam_name, []))
        print("已重新加载配置")

    def _finish_zone(self):
        if len(self.drawing_pts) < 3:
            return
        color = ZONE_COLORS[len(self.zones_norm) % len(ZONE_COLORS)]
        self.zones_norm.append({
            "name": self.pending_name or f"区域{len(self.zones_norm)+1}",
            "poly": pixel_poly_to_norm(self.drawing_pts, self.w, self.h),
            "color": list(color),
        })
        self.drawing_pts = []
        self.mode = "edit"
        self.pending_name = ""

    def _finish_line(self):
        if len(self.drawing_pts) != 2:
            return
        norm = pixel_line_to_norm(self.drawing_pts[0], self.drawing_pts[1], self.w, self.h)
        self.lines_norm.append({
            "name": self.pending_name or f"越线{len(self.lines_norm)+1}",
            **norm,
        })
        self.drawing_pts = []
        self.line_step = 0
        self.mode = "edit"
        self.pending_name = ""

    def on_mouse(self, event, x, y, flags, param):
        frame = param

        if event == cv2.EVENT_LBUTTONDOWN:
            if self.mode == "new_zone":
                self.drawing_pts.append((x, y))
            elif self.mode == "new_line":
                self.drawing_pts.append((x, y))
                if len(self.drawing_pts) == 2:
                    self._finish_line()
            else:
                for zi, z in enumerate(self._zones_px()):
                    idx = _nearest_point(z["poly"], x, y)
                    if idx >= 0:
                        self.drag = ("zone", zi, idx)
                        return
                for li, ln in enumerate(self._lines_px()):
                    for pi, pt in enumerate([ln["p1"], ln["p2"]]):
                        if _nearest_point([pt], x, y) >= 0:
                            self.drag = ("line", li, pi)
                            return

        elif event == cv2.EVENT_MOUSEMOVE and self.drag:
            kind, idx, pi = self.drag
            if kind == "zone":
                poly = [[p[0], p[1]] for p in self.zones_norm[idx]["poly"]]
                poly[pi] = [x / self.w, y / self.h]
                self.zones_norm[idx]["poly"] = poly
            else:
                key = "p1" if pi == 0 else "p2"
                self.lines_norm[idx][key] = [round(x / self.w, 4), round(y / self.h, 4)]

        elif event == cv2.EVENT_LBUTTONUP:
            self.drag = None

        elif event == cv2.EVENT_RBUTTONDOWN and self.mode == "new_zone":
            self._finish_zone()

    def draw(self, frame):
        vis = frame.copy()
        for z in self._zones_px():
            pts = np.array(z["poly"], dtype=np.int32).reshape(-1, 1, 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], z["color"])
            cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
            cv2.polylines(vis, [pts], True, z["color"], 2)
            for p in z["poly"]:
                cv2.circle(vis, p, 6, z["color"], -1)
            cv2.putText(vis, z["name"], z["poly"][0], cv2.FONT_HERSHEY_SIMPLEX, 0.6, z["color"], 2)

        for ln in self._lines_px():
            cv2.line(vis, ln["p1"], ln["p2"], (0, 255, 255), 2)
            cv2.circle(vis, ln["p1"], 6, (0, 255, 255), -1)
            cv2.circle(vis, ln["p2"], 6, (0, 255, 255), -1)
            mx = (ln["p1"][0] + ln["p2"][0]) // 2
            my = (ln["p1"][1] + ln["p2"][1]) // 2
            cv2.putText(vis, ln["name"], (mx, my - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        if self.drawing_pts:
            for p in self.drawing_pts:
                cv2.circle(vis, p, 5, (0, 0, 255), -1)
            if len(self.drawing_pts) >= 2:
                pts = np.array(self.drawing_pts, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(vis, [pts], False, (0, 0, 255), 2)

        help_lines = [
            f"Cam: {self.cam_name}  Mode: {self.mode}",
            "Z:新区域 L:新越线 D:删除 S:保存 N:切换摄像头 R:重载 Q:退出",
            "左键:标点/拖拽顶点  右键:完成区域",
        ]
        for i, t in enumerate(help_lines):
            cv2.putText(vis, t, (10, 25 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return vis

    def run(self):
        cap = cv2.VideoCapture(self.cfg["rtsp"], cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                blank = np.zeros((self.h, self.w, 3), dtype=np.uint8)
                cv2.putText(blank, "No Signal - use last frame layout", (40, self.h // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                frame = blank
            else:
                frame = cv2.resize(frame, (self.w, self.h))

            cv2.setMouseCallback(self.win, self.on_mouse, frame.copy())
            cv2.imshow(self.win, self.draw(frame))
            key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                self._save()
            elif key == ord("r"):
                self._reload()
            elif key == ord("d"):
                if self.zones_norm:
                    self.zones_norm.pop()
                elif self.lines_norm:
                    self.lines_norm.pop()
            elif key == ord("z"):
                self.mode = "new_zone"
                self.drawing_pts = []
                self.pending_name = input("区域名称: ").strip() or f"区域{len(self.zones_norm)+1}"
            elif key == ord("l"):
                self.mode = "new_line"
                self.drawing_pts = []
                self.pending_name = input("越线名称: ").strip() or f"越线{len(self.lines_norm)+1}"
            elif key == ord("n"):
                names = list(CAMERA_CONFIGS.keys())
                idx = (names.index(self.cam_name) + 1) % len(names)
                self.cam_name = names[idx]
                self.cfg = CAMERA_CONFIGS[self.cam_name]
                self.w, self.h = self.cfg["width"], self.cfg["height"]
                self._reload()
                self.win = f"Zone Editor - {self.cam_name}"
                cv2.destroyWindow(self.win)
                cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)

        cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> None:
    import sys

    args = argv if argv is not None else sys.argv[1:]
    cam = args[0] if args else list(CAMERA_CONFIGS.keys())[0]
    ZoneEditor(cam).run()


if __name__ == "__main__":
    main()
