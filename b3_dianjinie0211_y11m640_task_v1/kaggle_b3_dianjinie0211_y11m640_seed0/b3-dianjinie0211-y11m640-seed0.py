#!/usr/bin/env python3
"""Route B3 Kaggle kernel: train YOLO11m@640 on Dianjinie0211 (5 classes).

Config-dataset pattern: attach the private image Dataset
``lly3323w/dianjinie`` (pre-existing mirror of ``datasets/Dianjinie0211``)
and the private config Dataset ``lly3323w/dianjinie0211-b3-config-v1``
(split lists + full SHA-256 tables + trainer).  Before training, this kernel
hash-verifies every mounted label and image against the config tables, so
local and remote content are proven byte-identical.

Protocol mirrors the L1 fair-training baseline:
seed0 / 300ep / batch8 / SGD / patience100 / imgsz640 / ultralytics 8.4.66.
Run on a GPU notebook with Internet enabled.  Switch MODE to "preflight" for
a validation-only run (hash checks still execute; no pip installs, no training).
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
MODE = "train"  # "train" or "preflight"
DATA_VERSION = "d0211v1"
RUN_NAME = "dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211v1"
EXPECTED_CLASSES = ["waterstain", "waterblock", "colorstain", "nodule", "bubble"]
VERIFY_IMAGE_HASHES = True  # 18 GiB read; disable only for a fast retry of a green run
REQUIRED_EVIDENCE = [
    "weights/best.pt",
    "weights/last.pt",
    "args.yaml",
    "requested_config.json",
    "environment.json",
    "results.csv",
    "README.md",
    "dataset_config/data.yaml",
    "dataset_config/train.txt",
    "dataset_config/val.txt",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_datasets() -> tuple[Path, Path]:
    """Uniquely locate (image_dataset_root, config_dataset_root).

    Kaggle mounts sources either flat at /kaggle/input/<slug> or scoped under
    /kaggle/input/datasets/<owner>/<slug>; scan both layouts.
    """
    if not INPUT_ROOT.is_dir():
        raise FileNotFoundError("未找到 /kaggle/input；请在 Kaggle Notebook 中运行")
    roots = [path for path in INPUT_ROOT.iterdir() if path.is_dir()]
    scoped_root = INPUT_ROOT / "datasets"
    if scoped_root.is_dir():
        roots.extend(path for path in scoped_root.rglob("*") if path.is_dir())
    image_roots = [
        path for path in roots
        if (path / "data.yaml").is_file() and (path / "images" / "train").is_dir()
    ]
    config_roots = [path for path in roots if (path / "b3_config_manifest.json").is_file()]
    if len(config_roots) != 1:
        inventory = {
            str(path.relative_to(INPUT_ROOT)): sorted(item.name for item in path.iterdir())[:6]
            for path in roots
        }
        raise RuntimeError(f"无法唯一定位 B3 config Dataset：matched={config_roots}，挂载清单={inventory}")
    if len(image_roots) != 1:
        inventory = {
            str(path.relative_to(INPUT_ROOT)): sorted(item.name for item in path.iterdir())[:6]
            for path in roots
        }
        raise RuntimeError(f"无法唯一定位图片 Dataset（需含 data.yaml 与 images/train）：matched={image_roots}，挂载清单={inventory}")
    return image_roots[0], config_roots[0]


def verify_config(config_root: Path) -> dict:
    manifest = json.loads((config_root / "b3_config_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "b3_config_v1" or manifest.get("classes") != EXPECTED_CLASSES:
        raise ValueError("B3 config manifest 版本或类别不符合预期")
    for name, expected_hash in manifest["files"].items():
        actual = sha256_file(config_root / name)
        if actual != expected_hash:
            raise ValueError(f"config 文件 sha256 不匹配：{name}")
    print("[config] manifest and payload hashes OK", flush=True)
    return manifest


def verify_semantics(image_root: Path, config_root: Path) -> None:
    remote_yaml = (image_root / "data.yaml").read_text(encoding="utf-8")
    if "nc: 5" not in remote_yaml:
        raise ValueError(f"图片 Dataset data.yaml 类别数不是 5：{image_root / 'data.yaml'}")
    for index, name in enumerate(EXPECTED_CLASSES):
        if f"{index}: {name}" not in remote_yaml:
            raise ValueError(f"图片 Dataset data.yaml 类别 {index} 不是 {name}")


def verify_payload_hashes(image_root: Path, config_root: Path, manifest: dict) -> None:
    label_table = json.loads((config_root / "label_hashes.json").read_text(encoding="utf-8"))
    train_lines = (config_root / "meta__train.txt").read_text(encoding="utf-8").split()
    val_lines = (config_root / "meta__val.txt").read_text(encoding="utf-8").split()
    listed = train_lines + val_lines
    if set(listed) != set(label_table):
        raise ValueError(f"清单与标签哈希表不一致：missing={set(label_table) ^ set(listed)}")
    print(f"[verify] {len(label_table)} labels...", flush=True)
    for relative, expected_hash in label_table.items():
        label_relative = relative.replace("images/", "labels/", 1).rsplit(".", 1)[0] + ".txt"
        actual = sha256_file(image_root / label_relative)
        if actual != expected_hash:
            raise ValueError(f"标签 sha256 不匹配（本地 vs 远端）：{relative}")
    if VERIFY_IMAGE_HASHES:
        image_table = json.loads((config_root / "image_hashes.json").read_text(encoding="utf-8"))
        if set(image_table) != set(listed):
            raise ValueError("清单与图片哈希表不一致")
        print(f"[verify] {len(image_table)} images (约 18 GiB 读取)...", flush=True)
        for index, (relative, expected_hash) in enumerate(sorted(image_table.items()), 1):
            actual = sha256_file(image_root / relative)
            if actual != expected_hash:
                raise ValueError(f"图片 sha256 不匹配（本地 vs 远端）：{relative}")
            if index % 200 == 0:
                print(f"[verify] {index} images OK", flush=True)
    else:
        print("[verify] image hash check skipped by flag; label checks already passed", flush=True)
    print("[verify] mounted dataset is byte-identical to the local split source", flush=True)


def install_dependencies() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--force-reinstall",
            "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "numpy==1.26.4", "ultralytics==8.4.66"],
        check=True,
    )


def check_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("未检测到 GPU；请在 Kaggle Notebook 设置中选择 GPU accelerator")
    print(f"[environment] torch={torch.__version__}, cuda={torch.version.cuda}, gpu={torch.cuda.get_device_name(0)}", flush=True)


def main() -> None:
    image_root, config_root = locate_datasets()
    print(f"[dataset] image={image_root}", flush=True)
    print(f"[dataset] config={config_root}", flush=True)
    manifest = verify_config(config_root)
    verify_semantics(image_root, config_root)
    verify_payload_hashes(image_root, config_root, manifest)

    trainer = config_root / "trainer__train_dianjinie0211_b3.py"
    command = [
        sys.executable, str(trainer),
        "--model", "yolo11m.pt",
        "--dataset-root", str(image_root),
        "--train-list", str(config_root / "meta__train.txt"),
        "--val-list", str(config_root / "meta__val.txt"),
        "--data-version", DATA_VERSION,
        "--platform", "kaggle",
        "--project", "/kaggle/working/runs",
        "--runtime-config-root", "/kaggle/working/runtime_configs",
        "--epochs", "300",
        "--imgsz", "640",
        "--batch", "8",
        "--workers", "8",
        "--device", "0",
        "--seed", "0",
        "--patience", "100",
    ]
    if MODE == "preflight":
        command.append("--dry-run")
    else:
        install_dependencies()
        check_gpu()
    print("[train]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    if MODE != "train":
        print("[preflight] all checks passed; no training executed", flush=True)
        return

    run_dir = Path("/kaggle/working/runs") / RUN_NAME
    if not run_dir.is_dir():
        raise FileNotFoundError(f"训练完成但未找到预期 run 目录：{run_dir}")
    missing = [relative for relative in REQUIRED_EVIDENCE if not (run_dir / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"run 缺少必要证据文件：{missing}")
    archive = Path(
        shutil.make_archive(str(Path("/kaggle/working") / f"{RUN_NAME}_results"), "zip", run_dir.parent, run_dir.name)
    )
    print(f"[archive] {archive}", flush=True)


if __name__ == "__main__":
    main()
