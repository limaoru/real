"""二维几何工具：多边形、越线、聚集检测。"""
from collections import defaultdict

from retail.config.settings import GROUP_DISTANCE


def point_in_polygon(px: float, py: float, polygon: list) -> bool:
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1 + 1e-6) + x1):
            inside = not inside
    return inside


def line_side(p, a, b) -> float:
    return (p[0] - a[0]) * (b[1] - a[1]) - (p[1] - a[1]) * (b[0] - a[0])


def detect_line_cross(prev_pt, curr_pt, a, b) -> str | None:
    if prev_pt is None:
        return None
    s0, s1 = line_side(prev_pt, a, b), line_side(curr_pt, a, b)
    if s0 == 0 or s1 == 0 or s0 * s1 >= 0:
        return None
    return "in" if s1 > 0 else "out"


def detect_groups(centers: list, dist_thresh: float = GROUP_DISTANCE) -> int:
    n = len(centers)
    if n < 2:
        return 0
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            dx, dy = centers[i][0] - centers[j][0], centers[i][1] - centers[j][1]
            if dx * dx + dy * dy <= dist_thresh * dist_thresh:
                pi, pj = find(i), find(j)
                if pi != pj:
                    parent[pj] = pi
    clusters: dict[int, int] = defaultdict(int)
    for i in range(n):
        clusters[find(i)] += 1
    return sum(1 for c in clusters.values() if c >= 2)


def box_contains_point(box, px: float, py: float, margin: float = 0.15) -> bool:
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    return (x1 - w * margin) <= px <= (x2 + w * margin) and (y1 - h * margin) <= py <= (y2 + h * margin)


def scale_boxes(boxes, scale: float):
    if scale == 1.0:
        return boxes
    return boxes / scale
