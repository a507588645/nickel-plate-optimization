#!/usr/bin/env python3
"""Run the E3 native-module YOLO11m P2-head CPU smoke test.

This script intentionally uses synthetic tensors only. It does not read any
dataset or frozen validation list, does not start training, and forces all
model work onto CPU. The custom YAML may use only modules provided by the
installed Ultralytics runtime.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Must be set before importing torch: this process has no reason to initialize CUDA.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import ultralytics
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_YAML = ROOT / "yolo11m_p2.yaml"
OFFICIAL_YOLO11_YAML = (
    Path(ultralytics.__file__).resolve().parent / "cfg" / "models" / "11" / "yolo11.yaml"
)


def sha256_file(path: Path) -> str:
    """Return the immutable file digest used to identify the tested YAML."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def positive_int(value: str) -> int:
    """Parse an argument that must be strictly positive."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the zero-GPU E3 native-module P2-head smoke test on CPU."
    )
    parser.add_argument(
        "--model-yaml",
        type=Path,
        default=DEFAULT_MODEL_YAML,
        help="Custom YOLO11m P2 YAML to validate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT,
        help="Versioned evidence directory. Defaults to this script directory.",
    )
    parser.add_argument("--nc", type=positive_int, default=18, help="Detection class count.")
    parser.add_argument(
        "--microbatch-size",
        type=positive_int,
        default=1,
        help="Synthetic CPU batch size for each forward/backward check.",
    )
    parser.add_argument(
        "--microbatch-imgsz",
        type=positive_int,
        default=64,
        help="Square synthetic image size for the forward/backward checks.",
    )
    parser.add_argument(
        "--microbatches",
        type=positive_int,
        default=3,
        help="Number of independently sampled forward/backward checks.",
    )
    parser.add_argument(
        "--profile-imgsz",
        type=positive_int,
        default=640,
        help="Square image size used when requesting Ultralytics FLOPs information.",
    )
    parser.add_argument("--seed", type=int, default=0, help="CPU random seed.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and planned CPU inputs without building a model.",
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing same-mode report in output-dir.",
    )
    return parser.parse_args()


def require_p2_yaml(cfg: dict[str, Any], yaml_path: Path) -> None:
    """Reject a YAML that is not a four-scale P2 detection model."""
    for key in ("nc", "scales", "backbone", "head"):
        if key not in cfg:
            raise ValueError(f"{yaml_path} is missing required key: {key}")
    if cfg.get("scale") != "m":
        raise ValueError(f"{yaml_path} must explicitly set scale: m")
    if "m" not in cfg["scales"]:
        raise ValueError(f"{yaml_path} scales must define the m compound scale")
    detect_entries = [row for row in cfg["head"] if len(row) >= 3 and row[2] == "Detect"]
    if len(detect_entries) != 1:
        raise ValueError(f"{yaml_path} must contain exactly one Detect layer")
    detect_from = detect_entries[0][0]
    if not isinstance(detect_from, list) or len(detect_from) != 4:
        raise ValueError("E3 requires a four-input Detect(P2, P3, P4, P5) layer")


def load_cfg(yaml_path: Path, nc: int) -> dict[str, Any]:
    """Load and normalize a YOLO config without mutating the source YAML."""
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Model YAML does not exist: {yaml_path}")
    cfg = dict(YAML.load(yaml_path))
    require_p2_yaml(cfg, yaml_path)
    cfg["nc"] = nc
    cfg["scale"] = "m"
    return cfg


def load_baseline_cfg(nc: int) -> dict[str, Any]:
    """Load the runtime's unmodified YOLO11m configuration for cost comparison."""
    if not OFFICIAL_YOLO11_YAML.is_file():
        raise FileNotFoundError(f"Official runtime YAML does not exist: {OFFICIAL_YOLO11_YAML}")
    cfg = dict(YAML.load(OFFICIAL_YOLO11_YAML))
    cfg["nc"] = nc
    cfg["scale"] = "m"
    return cfg


def model_summary(model: DetectionModel, profile_imgsz: int) -> dict[str, Any]:
    """Return parameter and FLOPs data, preserving a profiling error as evidence."""
    summary: dict[str, Any] = {
        "layers": len(model.model),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "strides": [float(value) for value in model.stride.detach().cpu().tolist()],
    }
    try:
        summary["flops_g"] = float(get_flops(model, imgsz=profile_imgsz))
    except Exception as exc:  # Profiling is desirable, not a reason to hide a valid build.
        summary["flops_g"] = None
        summary["flops_error"] = f"{type(exc).__name__}: {exc}"
    return summary


def flatten_tensors(value: Any) -> Iterable[torch.Tensor]:
    """Yield tensors from the output structures used by installed Detect heads."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from flatten_tensors(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from flatten_tensors(child)


def run_microbatches(model: DetectionModel, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Exercise gradient flow through independent synthetic CPU microbatches."""
    records: list[dict[str, Any]] = []
    model.train()
    for index in range(args.microbatches):
        model.zero_grad(set_to_none=True)
        images = torch.randn(
            args.microbatch_size,
            3,
            args.microbatch_imgsz,
            args.microbatch_imgsz,
            device=torch.device("cpu"),
        )
        started = time.perf_counter()
        outputs = model(images)
        tensors = list(flatten_tensors(outputs))
        if not tensors:
            raise RuntimeError("DetectionModel forward returned no tensors")
        loss = sum(tensor.float().mean() for tensor in tensors)
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"Non-finite synthetic scalar loss: {loss.item()}")
        loss.backward()
        elapsed = time.perf_counter() - started
        gradient_tensors = sum(
            1 for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None
        )
        if gradient_tensors == 0:
            raise RuntimeError("Backward completed without any parameter gradients")
        records.append(
            {
                "index": index + 1,
                "input_shape": list(images.shape),
                "output_tensor_count": len(tensors),
                "output_shapes": [list(tensor.shape) for tensor in tensors],
                "synthetic_loss": float(loss.detach().cpu().item()),
                "parameter_tensors_with_gradients": gradient_tensors,
                "elapsed_seconds": round(elapsed, 6),
            }
        )
    return records


def report_markdown(report: dict[str, Any]) -> str:
    """Render a concise human-readable evidence report from the JSON record."""
    runtime = report["runtime"]
    lines = [
        "# L1 E3 P2 检测头本机零 GPU 冒烟",
        "",
        f"- 状态：`{report['status']}`",
        f"- 运行时间：`{report['created_at_utc']}`",
        "- 范围：仅合成 CPU 张量；未读取数据集、未读取 `release-validation`、未启动训练或 Kaggle。",
        f"- 运行时：Python `{runtime['python']}`；Torch `{runtime['torch']}`；Ultralytics `{runtime['ultralytics']}`。",
        f"- 强制设备：`{runtime['device']}`，进程 `CUDA_VISIBLE_DEVICES={runtime['cuda_visible_devices']}`。",
        f"- YAML：`{report['model_yaml']}`，SHA-256=`{report['model_yaml_sha256']}`。",
        "",
        "## 拓扑",
        "",
        "- 基于运行时自带 `yolo11.yaml` 的 YOLO11m 主干，新增 P3→P2 上采样、原生 P2 拼接和 P2→P3 回流；检测层为 P2/P3/P4/P5 四尺度。",
        "- 使用模块仅为 `Conv`、`C3k2`、`SPPF`、`C2PSA`、`nn.Upsample`、`Concat`、`Detect`；未修改或 vendor patch Ultralytics，未使用 SPD-Conv/NWD。",
        "",
        "## 结果",
        "",
    ]
    if report["status"] == "passed":
        p2 = report["p2_model"]
        baseline = report["baseline_yolo11m"]
        lines.extend(
            [
                f"- YAML 解析与模型构建：通过；检测步长 `{p2['strides']}`。",
                f"- 合成前向/反向：`{len(report['microbatches'])}` 个 microbatch 全部通过。",
                f"- P2 参数量：`{p2['parameters']:,}`；标准 YOLO11m：`{baseline['parameters']:,}`；增量：`{report['parameter_delta']:,}`。",
                f"- P2 FLOPs@{report['arguments']['profile_imgsz']}：`{p2['flops_g']}` GFLOPs；标准 YOLO11m：`{baseline['flops_g']}` GFLOPs；增量：`{report['flops_delta_g']}` GFLOPs。",
                "- 冒烟闸门：`pass_local_smoke`。这只证明固定运行时下的解析、构建和 CPU 梯度链路可用，不代表训练收益或可进入 Fold 2/3。",
            ]
        )
    elif report["status"] == "dry_run_passed":
        lines.extend(
            [
                "- 输入 YAML 与运行时路径检查通过；未构建模型、未请求 FLOPs、未运行前向/反向。",
                "- 后续须不带 `--dry-run` 执行实际 CPU 冒烟后，才能判定 E3 是否可进入 Kaggle Fold 1 训练准备。",
            ]
        )
    else:
        lines.extend(
            [
                f"- 错误：`{report['error']}`",
                "- 冒烟闸门：`fail_stop_e3`，不得准备或启动 E3 Kaggle 训练，除非以新版本 YAML 修复后重新立项。",
            ]
        )
    lines.extend(
        [
            "",
            "## 限制",
            "",
            "- 本验证未使用真实图像或标签，不评估 mAP、类别 AP/Recall、训练速度或显存；这些均需在 E3 的单独授权 Kaggle Fold 1 中按固定开发边界验证。",
            "- 结果属于 `auto_only` 工程可行性证据，不得作为人工真值或最终模型选择结论。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    args.model_yaml = args.model_yaml.resolve()
    args.output_dir = args.output_dir.resolve()
    report_name = "dry_run_report.json" if args.dry_run else "smoke_report.json"
    markdown_name = "dry_run_report.md" if args.dry_run else "smoke_report.md"
    report_path = args.output_dir / report_name
    markdown_path = args.output_dir / markdown_name
    if report_path.exists() and not args.allow_overwrite:
        raise FileExistsError(f"Refusing to overwrite existing evidence: {report_path}")

    torch.manual_seed(args.seed)
    runtime = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "device": "cpu",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available_after_cpu_mask": bool(torch.cuda.is_available()),
    }
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "failed",
        "gate": "fail_stop_e3",
        "scope": {
            "synthetic_tensors_only": True,
            "dataset_read": False,
            "release_validation_read": False,
            "kaggle_started": False,
            "training_started": False,
            "vendor_patch_used": False,
        },
        "arguments": {
            "nc": args.nc,
            "microbatch_size": args.microbatch_size,
            "microbatch_imgsz": args.microbatch_imgsz,
            "microbatches": args.microbatches,
            "profile_imgsz": args.profile_imgsz,
            "seed": args.seed,
            "dry_run": args.dry_run,
        },
        "runtime": runtime,
        "model_yaml": str(args.model_yaml),
        "model_yaml_sha256": sha256_file(args.model_yaml) if args.model_yaml.is_file() else None,
        "official_yolo11_yaml": str(OFFICIAL_YOLO11_YAML),
    }
    try:
        p2_cfg = load_cfg(args.model_yaml, args.nc)
        baseline_cfg = load_baseline_cfg(args.nc)
        report["yaml_validation"] = {
            "passed": True,
            "p2_detect_inputs": list(next(row[0] for row in p2_cfg["head"] if row[2] == "Detect")),
            "baseline_yaml_exists": OFFICIAL_YOLO11_YAML.is_file(),
        }
        if args.dry_run:
            report["status"] = "dry_run_passed"
            report["gate"] = "not_evaluated_dry_run"
        else:
            p2_model = DetectionModel(cfg=p2_cfg, ch=3, nc=args.nc, verbose=False).to("cpu")
            baseline_model = DetectionModel(cfg=baseline_cfg, ch=3, nc=args.nc, verbose=False).to("cpu")
            report["p2_model"] = model_summary(p2_model, args.profile_imgsz)
            report["baseline_yolo11m"] = model_summary(baseline_model, args.profile_imgsz)
            report["parameter_delta"] = (
                report["p2_model"]["parameters"] - report["baseline_yolo11m"]["parameters"]
            )
            p2_flops = report["p2_model"]["flops_g"]
            baseline_flops = report["baseline_yolo11m"]["flops_g"]
            report["flops_delta_g"] = (
                round(p2_flops - baseline_flops, 6)
                if p2_flops is not None and baseline_flops is not None
                else None
            )
            report["microbatches"] = run_microbatches(p2_model, args)
            report["status"] = "passed"
            report["gate"] = "pass_local_smoke"
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    write_json(report_path, report)
    write_text(markdown_path, report_markdown(report))
    return 0 if report["status"] in {"passed", "dry_run_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
