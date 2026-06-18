"""推理后端：CUDA / CPU / OpenVINO 统一加载与参数。"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO

from retail.config.settings import (
    INFER_BACKEND,
    INFER_DEVICE,
    INFER_IMGSZ,
    OPENVINO_AUTO_EXPORT,
    SEG_CONF_MIN,
    SEG_IOU,
)
from retail.paths import BYTETRACK_CONFIG, OPENVINO_MODEL_DIR, PROJECT_ROOT


@dataclass(frozen=True)
class InferProfile:
    backend: str
    device: str | int
    half: bool
    label: str


def get_infer_profile() -> InferProfile:
    backend = INFER_BACKEND.strip().lower()
    if backend == "openvino":
        return InferProfile(
            backend="openvino",
            device="cpu",
            half=False,
            label="OpenVINO/CPU",
        )
    if backend == "cpu":
        return InferProfile(
            backend="cpu",
            device="cpu",
            half=False,
            label="PyTorch/CPU",
        )
    if backend == "cuda":
        if torch.cuda.is_available():
            return InferProfile(
                backend="cuda",
                device=INFER_DEVICE,
                half=True,
                label=f"PyTorch/CUDA:{INFER_DEVICE} FP16",
            )
        print("⚠️ INFER_BACKEND=cuda 但未检测到 CUDA，回退到 CPU")
        return InferProfile(
            backend="cpu",
            device="cpu",
            half=False,
            label="PyTorch/CPU (CUDA 不可用)",
        )
    raise ValueError(f"未知 INFER_BACKEND={INFER_BACKEND!r}，可选: cuda | cpu | openvino")


_PROFILE = get_infer_profile()


def infer_profile() -> InferProfile:
    return _PROFILE


def torch_device() -> torch.device:
    p = infer_profile()
    if p.backend == "cuda":
        return torch.device(f"cuda:{p.device}" if isinstance(p.device, int) else p.device)
    return torch.device("cpu")


def use_fp16() -> bool:
    return infer_profile().half


def yolo_kwargs(**extra: Any) -> dict[str, Any]:
    p = infer_profile()
    kw: dict[str, Any] = {
        "device": p.device,
        "half": p.half,
        "verbose": False,
        "imgsz": INFER_IMGSZ,
    }
    kw.update(extra)
    return kw


def track_kwargs(**extra: Any) -> dict[str, Any]:
    from retail.config.models import TARGET_CLASSES

    kw = yolo_kwargs(
        classes=TARGET_CLASSES,
        tracker=str(BYTETRACK_CONFIG),
        conf=SEG_CONF_MIN,
        iou=SEG_IOU,
    )
    kw.update(extra)
    return kw


def _resolve_pt_path(pt_name: str) -> Path:
    pt_path = Path(pt_name)
    if not pt_path.is_file() and (PROJECT_ROOT / pt_name).is_file():
        pt_path = PROJECT_ROOT / pt_name
    if not pt_path.is_file():
        raise FileNotFoundError(f"找不到权重: {pt_name}")
    return pt_path


def _openvino_dir_for(pt_name: str) -> Path:
    stem = Path(pt_name).stem
    return OPENVINO_MODEL_DIR / f"{stem}_openvino_model"


def _find_openvino_xml(ov_dir: Path) -> Path | None:
    if not ov_dir.is_dir():
        return None
    xmls = sorted(ov_dir.glob("*.xml"))
    return xmls[0] if xmls else None


def _candidate_openvino_dirs(pt_name: str) -> list[Path]:
    """Ultralytics 可能把 IR 放在 openvino_models/ 或权重同目录/项目根。"""
    stem = Path(pt_name).stem
    folder = f"{stem}_openvino_model"
    pt_path = _resolve_pt_path(pt_name)
    candidates = [
        OPENVINO_MODEL_DIR / folder,
        PROJECT_ROOT / folder,
        pt_path.parent / folder,
    ]
    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(path)
    return unique


def _find_openvino_dir(pt_name: str) -> Path | None:
    for ov_dir in _candidate_openvino_dirs(pt_name):
        if _find_openvino_xml(ov_dir):
            return ov_dir
    return None


def _canonicalize_openvino_dir(pt_name: str, found: Path) -> Path:
    """将已导出的 IR 归位到 openvino_models/（Ultralytics 常忽略 project/name）。"""
    target = _openvino_dir_for(Path(pt_name).name)
    if found.resolve() == target.resolve():
        return target
    OPENVINO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if target.exists():
        shutil.rmtree(target)
    shutil.move(str(found), str(target))
    print(f"📁 OpenVINO IR 已移至: {target}")
    return target


def _read_openvino_imgsz(ov_dir: Path) -> int | None:
    meta = ov_dir / "metadata.yaml"
    if meta.is_file():
        m = re.search(r"imgsz:\s*\n- (\d+)", meta.read_text(encoding="utf-8"))
        if m:
            return int(m.group(1))
    xml = _find_openvino_xml(ov_dir)
    if xml is not None:
        m = re.search(r'shape="1,3,(\d+),\1"', xml.read_text(encoding="utf-8"))
        if m:
            return int(m.group(1))
    return None


def _ensure_openvino_ready(pt_name: str, imgsz: int, *, allow_reexport: bool) -> Path | None:
    """已有 IR 且 imgsz 与配置一致则返回目录；尺寸不符时按 allow_reexport 重导或报错。"""
    found = _find_openvino_dir(pt_name)
    if not found:
        return None
    found = _canonicalize_openvino_dir(pt_name, found)
    got = _read_openvino_imgsz(found)
    if got == imgsz:
        return found
    if got is not None:
        msg = (
            f"OpenVINO IR 输入尺寸 {got} 与 INFER_IMGSZ={imgsz} 不一致: {found.name}"
        )
        if not allow_reexport:
            raise RuntimeError(f"{msg}。请运行: python -m retail export-openvino")
        print(f"♻️ {msg}，重新导出 {Path(pt_name).name} …")
        shutil.rmtree(found)
    return None


def export_openvino(pt_name: str, *, imgsz: int | None = None) -> Path:
    """从 .pt 导出 OpenVINO IR 到 openvino_models/。"""
    imgsz = imgsz or INFER_IMGSZ
    pt_path = _resolve_pt_path(pt_name)

    ready = _ensure_openvino_ready(pt_path.name, imgsz, allow_reexport=True)
    if ready:
        return ready

    try:
        import openvino  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "OpenVINO 未安装。请执行: pip install 'retail-analytics[openvino]'"
        ) from e

    OPENVINO_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target = _openvino_dir_for(pt_path.name)
    print(f"📦 导出 OpenVINO IR: {pt_path.name} → {target.name}/ (imgsz={imgsz}) …")
    model = YOLO(str(pt_path))
    model.export(
        format="openvino",
        imgsz=imgsz,
        half=False,
    )
    found = _find_openvino_dir(pt_path.name)
    if not found:
        raise RuntimeError(
            f"OpenVINO 导出失败，未在以下目录找到 IR: "
            f"{', '.join(str(p) for p in _candidate_openvino_dirs(pt_path.name))}"
        )
    target = _canonicalize_openvino_dir(pt_path.name, found)
    print(f"✅ OpenVINO 就绪: {target}")
    return target


def resolve_openvino_path(pt_name: str) -> str:
    ready = _ensure_openvino_ready(
        pt_name, INFER_IMGSZ, allow_reexport=OPENVINO_AUTO_EXPORT
    )
    if ready:
        return str(ready)
    if not OPENVINO_AUTO_EXPORT:
        target = _openvino_dir_for(Path(pt_name).name)
        raise FileNotFoundError(
            f"未找到 OpenVINO 模型 {target}，请先运行: python -m retail export-openvino"
        )
    return str(export_openvino(pt_name))


def load_yolo_model(pt_name: str) -> YOLO:
    p = infer_profile()
    if p.backend == "openvino":
        path = resolve_openvino_path(pt_name)
        print(f"🧠 OpenVINO 加载: {path}")
        return YOLO(path)
    print(f"🧠 PyTorch 加载: {pt_name} ({p.label})")
    return YOLO(pt_name)


def export_all_openvino(seg_pt: str, pose_pt: str) -> None:
    export_openvino(seg_pt)
    export_openvino(pose_pt)
