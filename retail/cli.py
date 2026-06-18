"""命令行入口。"""
from __future__ import annotations

import argparse
import sys

from retail.config.settings import DASHBOARD_PORT


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="retail",
        description="零售店铺视觉智能分析系统",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="启动双路 RTSP 分析主程序")
    sub.add_parser("dashboard", help=f"启动 Web 仪表盘 (端口 {DASHBOARD_PORT})")
    sub.add_parser("export-openvino", help="导出 seg/pose 权重为 OpenVINO IR")
    zones = sub.add_parser("zones", help="区域/越线鼠标标定工具")
    zones.add_argument("camera", nargs="?", help="摄像头名称")

    args = parser.parse_args(argv)

    if args.command == "run":
        from retail.pipeline.runner import run

        run()
        return 0

    if args.command == "dashboard":
        from retail.apps.dashboard import serve

        serve()
        return 0

    if args.command == "zones":
        from retail.apps.zone_editor import main as zones_main

        zones_main([args.camera] if args.camera else [])
        return 0

    if args.command == "export-openvino":
        from retail.config.models import POSE_MODEL, SEG_MODEL
        from retail.vision.infer_backend import export_all_openvino

        export_all_openvino(SEG_MODEL, POSE_MODEL)
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
