#!/usr/bin/env python3
"""Run one development-only L1 two-stage YOLO11m freeze task."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath


INPUT_ROOT = Path("/kaggle/input")
RUN_NAME = "dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_oofconsensusv1_freeze10_s1e020_s2e063_fold01"
VARIANT = "freeze10"
FREEZE = 10
MODE = "fold"
FOLD = 1
TRAINER_SOURCE = "#!/usr/bin/env python3\n\"\"\"Run a reproducible two-stage YOLO11m freeze ablation on the L1 dev view.\"\"\"\n\nfrom __future__ import annotations\n\nimport argparse\nimport hashlib\nimport json\nimport math\nimport platform\nimport shutil\nimport sys\nfrom datetime import datetime\nfrom pathlib import Path, PurePosixPath\nfrom typing import Any\n\n\nCLASS_NAMES = (\n    \"wuzi\",\n    \"waterstain\",\n    \"yellowstain\",\n    \"nodule\",\n    \"airhole\",\n    \"bubble\",\n    \"bump\",\n    \"fawu\",\n    \"overfire\",\n    \"NiOH\",\n    \"greenstain\",\n    \"suanwuran\",\n    \"quejiao\",\n    \"watermark\",\n    \"youpao\",\n    \"heibian\",\n    \"mijixiaokeng\",\n    \"fanxitong\",\n)\nEXPECTED_DATA_VERSION = \"oofconsensusv1\"\nEXPECTED_DEVELOPMENT_IMAGES = 1127\nEXPECTED_CANDIDATE_BOXES = 15671\nDEFAULT_ROOT = Path(__file__).resolve().parent\nDEFAULT_PROJECT = DEFAULT_ROOT / \"freeze_runs\"\nDEFAULT_RUNTIME_ROOT = DEFAULT_ROOT / \"freeze_runtime_configs\"\n\n\ndef parse_args(argv: list[str] | None = None) -> argparse.Namespace:\n    parser = argparse.ArgumentParser(\n        description=(\n            \"Run a development-only two-stage YOLO11m freeze experiment. \"\n            \"Stage 1 uses --freeze; stage 2 always unfreezes the full model.\"\n        ),\n        formatter_class=argparse.ArgumentDefaultsHelpFormatter,\n    )\n    parser.add_argument(\"--mode\", choices=(\"fold\", \"final\"), required=True)\n    parser.add_argument(\"--anchor-root\", type=Path, required=True)\n    parser.add_argument(\"--fold\", type=int, choices=(1, 2, 3))\n    parser.add_argument(\"--data-version\", default=EXPECTED_DATA_VERSION)\n    parser.add_argument(\"--model\", choices=(\"yolo11m.pt\",), default=\"yolo11m.pt\")\n    parser.add_argument(\"--variant\", choices=(\"control\", \"freeze10\"), required=True)\n    parser.add_argument(\"--freeze\", type=int, required=True)\n    parser.add_argument(\"--stage1-epochs\", type=int, default=20)\n    parser.add_argument(\"--stage2-epochs\", type=int, default=63)\n    parser.add_argument(\"--imgsz\", type=int, default=640)\n    parser.add_argument(\"--batch\", type=int, default=8)\n    parser.add_argument(\"--workers\", type=int, default=8)\n    parser.add_argument(\"--device\", default=\"0\")\n    parser.add_argument(\"--seed\", type=int, default=0)\n    parser.add_argument(\"--project\", type=Path, default=DEFAULT_PROJECT)\n    parser.add_argument(\"--runtime-config-root\", type=Path, default=DEFAULT_RUNTIME_ROOT)\n    parser.add_argument(\"--source-file\", type=Path, help=\"Copy this trainer source into the finished run evidence.\")\n    parser.add_argument(\"--dry-run\", action=\"store_true\", help=\"Validate the fixed view and print the request only.\")\n    return parser.parse_args(argv)\n\n\ndef sha256_file(path: Path) -> str:\n    digest = hashlib.sha256()\n    with path.open(\"rb\") as source:\n        for chunk in iter(lambda: source.read(1024 * 1024), b\"\"):\n            digest.update(chunk)\n    return digest.hexdigest()\n\n\ndef write_json(path: Path, payload: Any) -> None:\n    path.parent.mkdir(parents=True, exist_ok=True)\n    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + \"\\n\", encoding=\"utf-8\")\n\n\ndef validate_args(args: argparse.Namespace) -> None:\n    if args.data_version != EXPECTED_DATA_VERSION:\n        raise ValueError(f\"--data-version must be {EXPECTED_DATA_VERSION}\")\n    if args.mode == \"fold\" and args.fold is None:\n        raise ValueError(\"--mode fold requires --fold\")\n    if args.mode == \"final\" and args.fold is not None:\n        raise ValueError(\"--mode final must not receive --fold\")\n    expected_freeze = 0 if args.variant == \"control\" else 10\n    if args.freeze != expected_freeze:\n        raise ValueError(f\"Variant {args.variant!r} requires --freeze {expected_freeze}\")\n    if args.stage1_epochs <= 0 or args.stage2_epochs <= 0:\n        raise ValueError(\"Both training stages must have a positive epoch count\")\n    if args.stage1_epochs + args.stage2_epochs != 83:\n        raise ValueError(\"The paired experiment fixes the total budget at 83 epochs\")\n    if args.imgsz != 640 or args.batch != 8:\n        raise ValueError(\"The paired experiment fixes --imgsz 640 and --batch 8\")\n    if args.seed != 0:\n        raise ValueError(\"The paired experiment fixes --seed 0\")\n    if args.workers < 0:\n        raise ValueError(\"--workers must be non-negative\")\n\n\ndef read_classes(anchor_root: Path) -> None:\n    classes_path = anchor_root / \"classes.txt\"\n    if not classes_path.is_file():\n        raise FileNotFoundError(f\"Missing classes file: {classes_path}\")\n    actual = tuple(line.strip() for line in classes_path.read_text(encoding=\"utf-8\").splitlines() if line.strip())\n    if actual != CLASS_NAMES:\n        raise ValueError(\"The restored development view does not match the frozen RYW18 class order\")\n\n\ndef read_split(list_path: Path, anchor_root: Path) -> list[Path]:\n    if not list_path.is_file():\n        raise FileNotFoundError(f\"Missing fixed split list: {list_path}\")\n    images: list[Path] = []\n    for raw in list_path.read_text(encoding=\"utf-8\").splitlines():\n        value = raw.strip()\n        if not value:\n            continue\n        # Lists can carry paths from the machine where the view was first built.\n        image = anchor_root / \"images\" / PurePosixPath(value.replace(\"\\\\\", \"/\")).name\n        if not image.is_file():\n            raise FileNotFoundError(f\"Split image is missing from the restored view: {image}\")\n        # Keep the restored anchor pathname. Resolving a Kaggle image symlink\n        # changes its basename back to the source image name and breaks the\n        # required one-to-one label lookup under anchor_root/labels.\n        images.append(image.absolute())\n    if not images or len(images) != len(set(images)):\n        raise ValueError(f\"Split list is empty or contains duplicate images: {list_path}\")\n    return images\n\n\ndef audit_labels(images: list[Path], anchor_root: Path) -> int:\n    count = 0\n    for image in images:\n        label = anchor_root / \"labels\" / f\"{image.stem}.txt\"\n        if not label.is_file():\n            raise FileNotFoundError(f\"Missing YOLO label: {label}\")\n        for line_number, raw in enumerate(label.read_text(encoding=\"utf-8\").splitlines(), 1):\n            if not raw.strip():\n                continue\n            fields = raw.split()\n            if len(fields) != 5:\n                raise ValueError(f\"Label must contain five fields: {label}:{line_number}\")\n            try:\n                class_id = int(fields[0])\n                x, y, width, height = (float(value) for value in fields[1:])\n            except ValueError as error:\n                raise ValueError(f\"Non-numeric YOLO label: {label}:{line_number}\") from error\n            if not 0 <= class_id < len(CLASS_NAMES):\n                raise ValueError(f\"YOLO class index is out of range: {label}:{line_number}\")\n            if not all(math.isfinite(value) for value in (x, y, width, height)):\n                raise ValueError(f\"YOLO box is non-finite: {label}:{line_number}\")\n            if width <= 0 or height <= 0 or width > 1 or height > 1:\n                raise ValueError(f\"YOLO box dimensions are invalid: {label}:{line_number}\")\n            epsilon = 1e-9\n            if x - width / 2 < -epsilon or x + width / 2 > 1 + epsilon:\n                raise ValueError(f\"YOLO box exceeds horizontal bounds: {label}:{line_number}\")\n            if y - height / 2 < -epsilon or y + height / 2 > 1 + epsilon:\n                raise ValueError(f\"YOLO box exceeds vertical bounds: {label}:{line_number}\")\n            count += 1\n    return count\n\n\ndef collect_fold_images(anchor_root: Path, fold: int) -> tuple[list[Path], list[Path]]:\n    fold_root = anchor_root / \"splits\" / \"folds\" / f\"fold_{fold:02d}\"\n    train = read_split(fold_root / \"train.txt\", anchor_root)\n    val = read_split(fold_root / \"val.txt\", anchor_root)\n    if set(train) & set(val):\n        raise ValueError(f\"Fold {fold} train and val memberships overlap\")\n    expected = {1: (751, 376), 2: (752, 375), 3: (751, 376)}[fold]\n    if (len(train), len(val)) != expected:\n        raise ValueError(f\"Fold {fold} count mismatch: actual={(len(train), len(val))}, expected={expected}\")\n    return train, val\n\n\ndef collect_final_images(anchor_root: Path) -> list[Path]:\n    train_union: set[Path] = set()\n    val_union: set[Path] = set()\n    for fold in (1, 2, 3):\n        train, val = collect_fold_images(anchor_root, fold)\n        train_union.update(train)\n        val_union.update(val)\n    if train_union != val_union:\n        raise ValueError(\"The union of fixed OOF train and val lists is inconsistent\")\n    images = sorted(val_union, key=lambda item: item.name)\n    actual_images = {item.absolute() for item in (anchor_root / \"images\").glob(\"*.png\")}\n    if set(images) != actual_images or len(images) != EXPECTED_DEVELOPMENT_IMAGES:\n        raise ValueError(\"The fixed OOF lists do not cover exactly the development images\")\n    return images\n\n\ndef run_name(args: argparse.Namespace) -> str:\n    scope = f\"fold{args.fold:02d}\" if args.mode == \"fold\" else \"finaldev\"\n    return (\n        \"dianjienie_medium_yolo11m_640_kaggle_seed0_cm0_\"\n        f\"{args.data_version}_{args.variant}_s1e{args.stage1_epochs:03d}_s2e{args.stage2_epochs:03d}_{scope}\"\n    )\n\n\ndef write_path_list(path: Path, images: list[Path]) -> None:\n    path.write_text(\"\\n\".join(image.as_posix() for image in images) + \"\\n\", encoding=\"utf-8\")\n\n\ndef make_runtime_data(\n    anchor_root: Path,\n    train_images: list[Path],\n    val_images: list[Path],\n    name: str,\n    runtime_root: Path,\n) -> Path:\n    runtime = runtime_root / name\n    if runtime.exists():\n        raise FileExistsError(f\"Refusing to overwrite runtime configuration: {runtime}\")\n    runtime.mkdir(parents=True)\n    train_list = runtime / \"train.txt\"\n    val_list = runtime / \"val.txt\"\n    write_path_list(train_list, train_images)\n    write_path_list(val_list, val_images)\n    names = \"\\n\".join(f\"  {index}: {class_name}\" for index, class_name in enumerate(CLASS_NAMES))\n    data_yaml = runtime / \"data.yaml\"\n    data_yaml.write_text(\n        \"\\n\".join(\n            (\n                f'path: \"{anchor_root.as_posix()}\"',\n                f'train: \"{train_list.as_posix()}\"',\n                f'val: \"{val_list.as_posix()}\"',\n                \"\",\n                \"nc: 18\",\n                \"names:\",\n                names,\n                \"\",\n            )\n        ),\n        encoding=\"utf-8\",\n    )\n    return data_yaml\n\n\ndef training_kwargs(\n    data_yaml: Path,\n    name: str,\n    args: argparse.Namespace,\n    epochs: int,\n    freeze: int,\n    validation: bool,\n) -> dict[str, Any]:\n    return {\n        \"data\": str(data_yaml),\n        \"epochs\": epochs,\n        \"imgsz\": args.imgsz,\n        \"batch\": args.batch,\n        \"device\": args.device,\n        \"workers\": args.workers,\n        \"project\": str(args.project),\n        \"name\": name,\n        \"exist_ok\": False,\n        \"pretrained\": True,\n        \"optimizer\": \"SGD\",\n        \"seed\": args.seed,\n        \"deterministic\": True,\n        \"patience\": 0,\n        \"val\": validation,\n        \"cache\": False,\n        \"resume\": False,\n        \"close_mosaic\": 0,\n        \"amp\": True,\n        \"freeze\": freeze,\n        \"lr0\": 0.01,\n        \"lrf\": 0.01,\n        \"momentum\": 0.937,\n        \"weight_decay\": 0.0005,\n        \"warmup_epochs\": 3.0,\n        \"warmup_momentum\": 0.8,\n        \"warmup_bias_lr\": 0.1,\n        \"box\": 7.5,\n        \"cls\": 0.5,\n        \"dfl\": 1.5,\n        \"iou\": 0.7,\n        \"max_det\": 300,\n        \"hsv_h\": 0.015,\n        \"hsv_s\": 0.7,\n        \"hsv_v\": 0.4,\n        \"degrees\": 0.0,\n        \"translate\": 0.1,\n        \"scale\": 0.5,\n        \"shear\": 0.0,\n        \"perspective\": 0.0,\n        \"flipud\": 0.0,\n        \"fliplr\": 0.5,\n        \"bgr\": 0.0,\n        \"mosaic\": 1.0,\n        \"mixup\": 0.0,\n        \"cutmix\": 0.0,\n        \"copy_paste\": 0.0,\n        \"copy_paste_mode\": \"flip\",\n        \"auto_augment\": \"randaugment\",\n        \"erasing\": 0.4,\n        \"save_period\": -1,\n        # This only suppresses image rendering, avoiding an observed Kaggle callback crash.\n        \"plots\": False,\n    }\n\n\ndef environment_payload() -> dict[str, Any]:\n    import numpy\n    import torch\n    import torchvision\n    import ultralytics\n\n    return {\n        \"generated_at\": datetime.now().isoformat(timespec=\"seconds\"),\n        \"python\": sys.version,\n        \"platform\": platform.platform(),\n        \"torch\": torch.__version__,\n        \"torchvision\": torchvision.__version__,\n        \"cuda\": torch.version.cuda,\n        \"numpy\": numpy.__version__,\n        \"ultralytics\": ultralytics.__version__,\n        \"gpu\": torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"unavailable\",\n    }\n\n\ndef nonempty_file(path: Path, label: str) -> None:\n    if not path.is_file() or path.stat().st_size <= 0:\n        raise FileNotFoundError(f\"Missing or empty {label}: {path}\")\n\n\ndef write_readme(save_dir: Path, request: dict[str, Any]) -> None:\n    stage1 = request[\"training\"][\"stage1\"]\n    stage2 = request[\"training\"][\"stage2\"]\n    scope = f\"Fold {request['fold']}\" if request[\"fold\"] is not None else \"all 1,127 development images\"\n    save_dir.joinpath(\"README.md\").write_text(\n        \"\\n\".join(\n            (\n                \"# L1 YOLO11m two-stage freeze experiment\",\n                \"\",\n                f\"- Variant: `{request['variant']}`; Stage-1 freeze: `{stage1['freeze']}`.\",\n                f\"- Scope: `{scope}`; development-only: `{request['development_only']}`.\",\n                \"- Frozen historical release-validation is not mounted, read, or evaluated.\",\n                f\"- Stage 1: {stage1['epochs']} epochs from public `yolo11m.pt`.\",\n                f\"- Stage 2: {stage2['epochs']} epochs from Stage-1 `last.pt`, with `freeze=0` and `resume=false`.\",\n                \"- `weights/best.pt` is the Fold candidate when validation is enabled.\",\n                \"- For finaldev, `weights/final_fixed_epoch.pt` is copied from Stage-2 `last.pt`.\",\n                \"\",\n            )\n        ),\n        encoding=\"utf-8\",\n    )\n\n\ndef main(argv: list[str] | None = None) -> int:\n    args = parse_args(argv)\n    validate_args(args)\n    args.anchor_root = args.anchor_root.resolve()\n    args.project = args.project.resolve()\n    args.runtime_config_root = args.runtime_config_root.resolve()\n    if (args.anchor_root / \"splits\" / \"test.txt\").exists():\n        raise ValueError(\"This experiment accepts only the development-only OOF view and must not receive test.txt\")\n    read_classes(args.anchor_root)\n\n    if args.mode == \"fold\":\n        train_images, val_images = collect_fold_images(args.anchor_root, args.fold)\n        audit_images = sorted({*train_images, *val_images}, key=lambda item: item.name)\n    else:\n        train_images = collect_final_images(args.anchor_root)\n        val_images = train_images\n        audit_images = train_images\n    box_count = audit_labels(audit_images, args.anchor_root)\n    if len(audit_images) != EXPECTED_DEVELOPMENT_IMAGES or box_count != EXPECTED_CANDIDATE_BOXES:\n        raise ValueError(\n            \"The freeze experiment requires the exact 1,127-image / 15,671-box OOF development view; \"\n            f\"got images={len(audit_images)}, boxes={box_count}\"\n        )\n\n    name = run_name(args)\n    stage1_name = f\"{name}_stage1\"\n    runtime_yaml = args.runtime_config_root / name / \"data.yaml\"\n    stage1_kwargs = training_kwargs(\n        runtime_yaml, stage1_name, args, args.stage1_epochs, args.freeze, args.mode == \"fold\"\n    )\n    stage2_kwargs = training_kwargs(\n        runtime_yaml, name, args, args.stage2_epochs, 0, args.mode == \"fold\"\n    )\n    source = (args.source_file or Path(__file__)).resolve()\n    if not source.is_file():\n        raise FileNotFoundError(f\"Trainer source is missing: {source}\")\n    request: dict[str, Any] = {\n        \"experiment\": \"L1 YOLO11m paired two-stage freeze ablation\",\n        \"run_name\": name,\n        \"variant\": args.variant,\n        \"model\": args.model,\n        \"data_version\": args.data_version,\n        \"development_only\": True,\n        \"frozen_test_set\": None,\n        \"release_validation_access\": \"not read by this experiment\",\n        \"mode\": args.mode,\n        \"fold\": args.fold,\n        \"split_counts\": {\n            \"train\": len(train_images),\n            \"val\": len(val_images),\n            \"test_frozen\": None,\n            \"audited_development_images\": len(audit_images),\n            \"audited_candidate_boxes\": box_count,\n        },\n        \"anchor_root\": str(args.anchor_root),\n        \"runtime_data_yaml\": str(runtime_yaml),\n        \"expected_run_dir\": str(args.project / name),\n        \"trainer_source\": {\"path\": str(source), \"sha256\": sha256_file(source)},\n        \"training\": {\"stage1\": stage1_kwargs, \"stage2\": stage2_kwargs},\n        \"selection\": {\n            \"fold_candidate_weight\": \"weights/best.pt (stage 2 only)\",\n            \"final_weight\": \"weights/final_fixed_epoch.pt copied from stage-2 last.pt\",\n            \"total_epochs\": args.stage1_epochs + args.stage2_epochs,\n            \"notes\": \"The paired control uses the same two-stage optimizer reset and differs only by stage-1 freeze.\",\n        },\n    }\n    print(json.dumps(request, ensure_ascii=False, indent=2), flush=True)\n    if args.dry_run:\n        return 0\n\n    if (args.project / name).exists() or (args.project / stage1_name).exists():\n        raise FileExistsError(\"A stage output already exists; refusing to overwrite a prior experiment\")\n    try:\n        from ultralytics import YOLO\n    except ModuleNotFoundError as error:\n        raise RuntimeError(\"Ultralytics is unavailable; use the pinned Kaggle GPU runtime\") from error\n\n    data_yaml = make_runtime_data(args.anchor_root, train_images, val_images, name, args.runtime_config_root)\n    if data_yaml != runtime_yaml:\n        raise RuntimeError(\"Runtime data path unexpectedly changed\")\n    write_json(data_yaml.parent / \"requested_config.json\", request)\n\n    stage1_model = YOLO(args.model)\n    stage1_model.train(**stage1_kwargs)\n    stage1_dir = Path(stage1_model.trainer.save_dir)\n    stage1_last = stage1_dir / \"weights\" / \"last.pt\"\n    stage1_best = stage1_dir / \"weights\" / \"best.pt\"\n    nonempty_file(stage1_last, \"stage-1 last.pt\")\n    nonempty_file(stage1_best, \"stage-1 best.pt\")\n\n    stage2_model = YOLO(str(stage1_last))\n    stage2_model.train(**stage2_kwargs)\n    save_dir = Path(stage2_model.trainer.save_dir)\n    best_weight = save_dir / \"weights\" / \"best.pt\"\n    last_weight = save_dir / \"weights\" / \"last.pt\"\n    for artifact, label in (\n        (best_weight, \"stage-2 best.pt\"),\n        (last_weight, \"stage-2 last.pt\"),\n        (save_dir / \"args.yaml\", \"stage-2 args.yaml\"),\n        (save_dir / \"results.csv\", \"stage-2 results.csv\"),\n    ):\n        nonempty_file(artifact, label)\n\n    request[\"artifacts\"] = {\n        \"stage1_last_pt_sha256\": sha256_file(stage1_last),\n        \"stage1_best_pt_sha256\": sha256_file(stage1_best),\n        \"stage2_last_pt_sha256\": sha256_file(last_weight),\n        \"stage2_best_pt_sha256\": sha256_file(best_weight),\n    }\n    write_json(save_dir / \"requested_config.json\", request)\n    shutil.copytree(data_yaml.parent, save_dir / \"dataset_config\", dirs_exist_ok=False)\n    shutil.copytree(stage1_dir, save_dir / \"freeze_stage1\", dirs_exist_ok=False)\n    source_dir = save_dir / \"source\"\n    source_dir.mkdir()\n    shutil.copy2(source, source_dir / source.name)\n    write_json(save_dir / \"environment.json\", environment_payload())\n    write_readme(save_dir, request)\n\n    if args.mode == \"final\":\n        final_weight = save_dir / \"weights\" / \"final_fixed_epoch.pt\"\n        shutil.copy2(last_weight, final_weight)\n        write_json(\n            save_dir / \"release_weight.json\",\n            {\n                \"source\": \"weights/last.pt\",\n                \"sha256\": sha256_file(final_weight),\n                \"fixed_total_epochs\": args.stage1_epochs + args.stage2_epochs,\n                \"release_validation_access\": \"not read\",\n            },\n        )\n        write_json(\n            save_dir / \"dataset_config\" / \"epoch_selection_summary.json\",\n            {\n                \"purpose\": \"pre-registered fixed total epoch budget for the freeze10 finaldev candidate\",\n                \"total_epochs\": args.stage1_epochs + args.stage2_epochs,\n                \"stage1_epochs\": args.stage1_epochs,\n                \"stage2_epochs\": args.stage2_epochs,\n                \"selection_source\": \"paired Fold 1 freeze ablation gate\",\n            },\n        )\n    print(json.dumps({\"run_dir\": str(save_dir), \"run_name\": name}, ensure_ascii=False), flush=True)\n    return 0\n\n\nif __name__ == \"__main__\":\n    raise SystemExit(main())\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def mounted_roots() -> list[Path]:
    roots = [path for path in INPUT_ROOT.iterdir() if path.is_dir()]
    scoped = INPUT_ROOT / "datasets"
    if scoped.is_dir():
        roots.extend(path for path in scoped.rglob("*") if path.is_dir())
    return roots


def inventory() -> dict[str, list[str]]:
    return {
        str(root.relative_to(INPUT_ROOT)): sorted(path.name for path in root.iterdir())[:12]
        for root in mounted_roots()
    }


def resolve_config_payload() -> tuple[Path, Path]:
    candidates = [root for root in mounted_roots() if (root / "config_archive_manifest.json").is_file()]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate OOF config Dataset: {inventory()}")
    package_root = candidates[0]
    archive_manifest = json.loads((package_root / "config_archive_manifest.json").read_text(encoding="utf-8"))
    archive_name = archive_manifest.get("config_archive")
    payload_files = archive_manifest.get("payload_files")
    manifest_hash = archive_manifest.get("source_manifest_sha256")
    if not isinstance(archive_name, str) or not isinstance(payload_files, list) or not isinstance(manifest_hash, str):
        raise ValueError("Invalid OOF config archive manifest")
    payload_root = package_root / Path(archive_name).stem
    actual = {path.name for path in payload_root.iterdir() if path.is_file()} if payload_root.is_dir() else set()
    if actual != set(payload_files):
        raise ValueError(f"Expanded OOF config payload mismatch: expected={len(payload_files)}, actual={len(actual)}")
    config_manifest = payload_root / "config_manifest.json"
    if not config_manifest.is_file() or sha256_file(config_manifest) != manifest_hash:
        raise ValueError("OOF config manifest hash mismatch")
    runner = payload_root / "runner__kaggle_dianjienie_ryw18_existing_data_runner.py"
    if not runner.is_file():
        raise FileNotFoundError(f"Missing OOF runtime runner: {runner}")
    return payload_root, runner


def resolve_image_root(payload_root: Path) -> Path:
    manifest = json.loads((payload_root / "config_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("data_version") != "oofconsensusv1" or manifest.get("development_only") is not True:
        raise ValueError("This Kernel requires the development-only OOF-consensus configuration")
    images = manifest.get("images")
    if not isinstance(images, list) or not images or not isinstance(images[0], dict):
        raise ValueError("OOF config has no valid source image probe")
    source_relative = images[0].get("source_relative")
    if not isinstance(source_relative, str):
        raise ValueError("OOF source image probe is invalid")
    parts = PurePosixPath(source_relative).parts
    candidates = [
        root for root in mounted_roots()
        if root.is_dir() and root != payload_root.parent and root.joinpath(*parts).is_file()
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot uniquely locate image Dataset: matches={candidates}, inputs={inventory()}")
    return candidates[0]


def install_dependencies() -> None:
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "--force-reinstall",
            "torch==2.6.0", "torchvision==0.21.0", "--index-url", "https://download.pytorch.org/whl/cu124",
        ],
        check=True,
    )
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "--yes", "torchaudio"], check=False)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "numpy==1.26.4", "ultralytics==8.4.66"],
        check=True,
    )
    import numpy
    import torch
    import torchvision
    import ultralytics

    if (
        torch.__version__ != "2.6.0+cu124"
        or torchvision.__version__.split("+", 1)[0] != "0.21.0"
        or numpy.__version__ != "1.26.4"
        or ultralytics.__version__ != "8.4.66"
        or not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "Pinned runtime verification failed: "
            f"torch={torch.__version__}, torchvision={torchvision.__version__}, "
            f"numpy={numpy.__version__}, ultralytics={ultralytics.__version__}, cuda={torch.cuda.is_available()}"
        )


def main() -> None:
    payload_root, runner = resolve_config_payload()
    image_root = resolve_image_root(payload_root)
    anchor_root = Path("/kaggle/working") / f"{RUN_NAME}_anchor"
    prepare = [
        sys.executable,
        str(runner),
        "--config-root", str(payload_root),
        "--data-version", "oofconsensusv1",
        "--development-only",
        "--image-dataset-root", str(image_root),
        "--model", "yolo11m.pt",
        "--fold", str(FOLD),
        "--work-root", str(anchor_root),
        "--prepare-only",
    ]
    print("[prepare]", " ".join(prepare), flush=True)
    subprocess.run(prepare, check=True)
    install_dependencies()
    trainer = Path("/kaggle/working/train_l1_freeze_two_stage.py")
    trainer.write_text(TRAINER_SOURCE, encoding="utf-8")
    command = [
        sys.executable,
        str(trainer),
        "--mode", MODE,
        "--anchor-root", str(anchor_root),
        "--data-version", "oofconsensusv1",
        "--variant", VARIANT,
        "--freeze", str(FREEZE),
        "--project", "/kaggle/working/runs",
        "--runtime-config-root", "/kaggle/working/freeze_runtime_configs",
        "--source-file", str(trainer),
    ]
    if MODE == "fold":
        command.extend(["--fold", str(FOLD)])
    print("[train]", " ".join(command), flush=True)
    subprocess.run(command, check=True)
    run_dir = Path("/kaggle/working/runs") / RUN_NAME
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Expected run directory is missing: {run_dir}")
    archive = Path(shutil.make_archive(str(Path("/kaggle/working") / f"{RUN_NAME}_results"), "zip", run_dir.parent, run_dir.name))
    print(json.dumps({"run_dir": str(run_dir), "archive": str(archive), "archive_sha256": sha256_file(archive)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
