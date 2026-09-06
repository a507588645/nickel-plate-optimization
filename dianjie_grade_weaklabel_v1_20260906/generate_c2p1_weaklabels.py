# -*- coding: utf-8 -*-
"""C2.1 电解镍判级弱标签数据集生成（路线文档 §4.2 任务书，弱标签蒸馏第一步）。

用企业认可的旧方案完整管线（c5055bc：dianjie = yolov5/ryw.pt + ryw.pth ResNet + findcent + 决策树）
对 anchor_v1 dev 1,127 图（三个 fold 的 val 并集；冻结 198 test 不读取）逐图产出
type(0-6) 与 nodule(0-3) 弱标签。

划分：按 anchor_v1 group_assignments.csv 的板级 group 分组防泄漏，fold_01/02/03 分区
分别映射为 train/val/test（组与 fold 一一对应，组间无跨划分图片）。

manifest 记录：管线代码来源（commit c5055bc + scratch 解包）、ryw.pt/ryw.pth SHA-256、
生成日期、图片清单、分组与划分。产物只写入本目录，不碰数据集与 WebAPI 仓库。

用法：
  python generate_c2p1_weaklabels.py --dry-run     # 只校验清单/映射
  python generate_c2p1_weaklabels.py --limit 4     # 冒烟
  python generate_c2p1_weaklabels.py               # 全量 1,127 图（CPU 约 30-40 分钟）
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("YOLOV5_AUTOINSTALL", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
SCRATCH = Path("C:/Users/23916/AppData/Local/Temp/jcw_c21_weaklabel_scratch")
PIPELINE_COMMIT = "c5055bc"
ANCHOR_ROOT = Path("E:/PythonProject/datasets/dianjienie_ryw18_anchor_v1")
FOLDS = ("fold_01", "fold_02", "fold_03")
SPLIT_BY_FOLD = {"fold_01": "train", "fold_02": "val", "fold_03": "test"}
PIPELINE_FILES = (
    "inference/detectors.py",
    "inference/grade_features.py",
    "inference/models.py",
    "models.yaml",
    "preprocess.py",
    "model_decryptor.py",
    "hubconf.py",
    "adapters/yolov5.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_list(path: Path) -> list[str]:
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def stem_to_sample_id(image_filename: str) -> str:
    # dianjienie__train__1.png -> train__1
    stem = Path(image_filename).stem
    prefix = "dianjienie__"
    if not stem.startswith(prefix):
        raise ValueError(f"anchor 图文件名缺少 dianjienie__ 前缀: {image_filename}")
    return stem[len(prefix):]


def load_mappings() -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    with (ANCHOR_ROOT / "manifests" / "source_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        manifest = {row["sample_id"]: row for row in csv.DictReader(handle)}
    with (ANCHOR_ROOT / "manifests" / "group_assignments.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        group_partition = {row["group_id"]: row["partition"] for row in csv.DictReader(handle)}
    return manifest, group_partition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="只校验清单与映射，不加载模型")
    parser.add_argument("--limit", type=int, default=0, help="只处理前 N 张（0=全量）")
    args = parser.parse_args()

    manifest, group_partition = load_mappings()
    entries: list[dict[str, str]] = []
    seen: set[str] = set()
    for fold in FOLDS:
        for line in read_list(ANCHOR_ROOT / "splits" / "folds" / fold / "val.txt"):
            image_path = Path(line)
            sample_id = stem_to_sample_id(image_path.name)
            if sample_id in seen:
                raise ValueError(f"图片跨 fold 重复: {image_path.name}")
            seen.add(sample_id)
            row = manifest.get(sample_id)
            if row is None:
                raise ValueError(f"source_manifest 缺少 sample_id: {sample_id}")
            group_id = row["group_id"]
            partition = group_partition.get(group_id)
            if partition != fold:
                raise ValueError(f"group 分区与 fold 不一致: {sample_id} group={group_id} partition={partition} fold={fold}")
            entries.append({
                "image_path": str(image_path),
                "sample_id": sample_id,
                "group_id": group_id,
                "fold": fold,
                "split": SPLIT_BY_FOLD[fold],
                "source_split": row["source_split"],
            })
    if len(entries) != 1127:
        raise ValueError(f"dev 图数 {len(entries)} != 1127")
    if not all(Path(e["image_path"]).is_file() for e in entries):
        missing = [e["image_path"] for e in entries if not Path(e["image_path"]).is_file()]
        raise FileNotFoundError(f"缺少 {len(missing)} 张图片，首例: {missing[0]}")

    split_counts = {s: sum(1 for e in entries if e["split"] == s) for s in ("train", "val", "test")}
    group_counts = {s: len({e["group_id"] for e in entries if e["split"] == s}) for s in ("train", "val", "test")}
    print(json.dumps({"images": len(entries), "split_images": split_counts, "split_groups": group_counts},
                     ensure_ascii=False))
    # 组跨划分校验：任一 group 不得出现在多个 split
    group_to_splits: dict[str, set[str]] = {}
    for e in entries:
        group_to_splits.setdefault(e["group_id"], set()).add(e["split"])
    cross = {g: ss for g, ss in group_to_splits.items() if len(ss) > 1}
    if cross:
        raise ValueError(f"分组跨划分（泄漏）: {list(cross)[:3]}")

    if args.dry_run:
        (HERE / "dry_run_check.json").write_text(json.dumps(
            {"images": len(entries), "split_images": split_counts, "split_groups": group_counts,
             "cross_groups": 0}, ensure_ascii=False, indent=2), encoding="utf-8")
        print("[dry-run] 清单与映射校验通过")
        return 0

    if args.limit:
        entries = entries[: args.limit]

    if not SCRATCH.is_dir():
        raise FileNotFoundError(f"未找到管线解包目录: {SCRATCH}")
    os.chdir(SCRATCH)
    if str(SCRATCH) not in sys.path:
        sys.path.insert(0, str(SCRATCH))

    from inference import models as model_state
    from inference.detectors import dianjie_detect

    ryw_sha = file_sha256(SCRATCH / "weights" / "ryw.pt")
    rywpth_sha = file_sha256(SCRATCH / "weights" / "ryw.pth")
    model_state.load_resnet()
    model_state.load_dianjie_detector()
    assert model_state.yolo is not None and model_state.resnet is not None
    detector_kind = model_state.MODELS_CONFIG["detectors"]["dianjie"]["kind"]
    detector_weights = model_state.MODELS_CONFIG["detectors"]["dianjie"]["weights"]
    assert detector_kind == "yolov5" and "ryw.pt" in detector_weights, "scratch models.yaml 不是旧方案 ryw.pt 配置"
    print(f"[1/3] 管线加载完成: kind={detector_kind} weights={detector_weights} ryw.pt={ryw_sha[:12]}…", flush=True)

    rows: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    t0 = time.perf_counter()
    for index, entry in enumerate(entries, 1):
        result = dianjie_detect(entry["image_path"])
        data = result.get("data") or {}
        if result.get("code") != 1 or "type" not in data:
            failures.append({"image_path": entry["image_path"], "result": json.dumps(result, ensure_ascii=False)})
            continue
        rows.append({
            **entry,
            "type": int(data["type"]),
            "nodule": int(data["nodule"]),
            "n_defects": len(data.get("defects") or []),
            "defects": json.dumps(data.get("defects") or [], ensure_ascii=False),
        })
        if index == 1 or index % 50 == 0 or index == len(entries):
            rate = index / (time.perf_counter() - t0)
            eta_min = (len(entries) - index) / rate / 60 if rate else 0
            print(f"  [{index}/{len(entries)}] {rate:.2f} img/s, ETA {eta_min:.0f} min, fail={len(failures)}", flush=True)

    if failures:
        raise RuntimeError(f"{len(failures)} 张图管线失败，首例: {failures[0]}")

    with (HERE / "weaklabels.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (HERE / "weaklabels_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_path", "sample_id", "group_id", "fold", "split", "source_split",
            "type", "nodule", "n_defects"])
        writer.writeheader()
        writer.writerows([{k: row[k] for k in writer.fieldnames} for row in rows])  # type: ignore[arg-type]

    type_dist: dict[str, int] = {}
    nodule_dist: dict[str, int] = {}
    for row in rows:
        type_dist[str(row["type"])] = type_dist.get(str(row["type"]), 0) + 1
        nodule_dist[str(row["nodule"])] = nodule_dist.get(str(row["nodule"]), 0) + 1

    manifest_out = {
        "task": "C2.1 电解镍判级弱标签数据集（路线文档 §4.2）",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pipeline": {
            "description": "旧方案完整管线（企业认可配置）：dianjie=yolov5/ryw.pt + ryw.pth ResNet + findcent + 决策树",
            "commit": PIPELINE_COMMIT,
            "scratch_dir": str(SCRATCH),
            "files": {name: file_sha256(SCRATCH / name) for name in PIPELINE_FILES},
            "detector": {"kind": detector_kind, "weights": detector_weights, "threshold": 0.25,
                         "adapter_defaults": "AutoShape 640 / iou 0.45 / max_det 1000",
                         "ryw_pt_sha256": ryw_sha},
            "classifier": {"weights": "weights/ryw.pth", "ryw_pth_sha256": rywpth_sha},
        },
        "images": {
            "total": len(rows),
            "source": "anchor_v1 dev = fold_01/02/03 val 并集（冻结 198 test 未读取）",
            "split_images": {s: sum(1 for r in rows if r["split"] == s) for s in ("train", "val", "test")},
            "split_groups": {s: len({r["group_id"] for r in rows if r["split"] == s}) for s in ("train", "val", "test")},
            "grouping": "anchor_v1 group_assignments.csv 板级分组；fold 分区一一对应，组无跨划分",
        },
        "labels": {"type_range": "决策树输出 {1,2,3,4,6}（0/5 未使用）", "nodule_range": "{0,1,2,3}",
                   "type_distribution": type_dist, "nodule_distribution": nodule_dist},
        "outputs": {"jsonl": str(HERE / "weaklabels.jsonl"), "summary_csv": str(HERE / "weaklabels_summary.csv")},
        "provenance": {"generator": str(Path(__file__).resolve()), "python_env": "jcw_b3clean_wt/.venv (torch 2.12.1+cpu)"},
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "type_dist": type_dist, "nodule_dist": nodule_dist,
                      "split_images": manifest_out["images"]["split_images"]}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
