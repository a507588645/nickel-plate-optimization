#!/usr/bin/env python3
"""
对比电积镍 4 类分类方案：
1. 基于检测特征的 LightGBM（classify_from_detection_v1/models/dianji_lgbm*.pkl）
2. 基于 folder 图像的 ResNet18（resnet_dianji_weighted_os/best_model.pth）

输出：表格 + 各类别 F1/Recall/Precision，便于选择最终方案。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

warnings_ignored = True


class SimpleImageDataset(Dataset):
    def __init__(self, root: Path, transform: T.Compose):
        self.dataset = ImageFolder(root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img_path, label = self.dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


def load_resnet_result(model_dir: Path, data_root: Path, device: torch.device) -> tuple[list[int], list[int], list[str]]:
    artifact = torch.load(model_dir / "best_model.pth", map_location=device, weights_only=False)
    class_names = artifact["class_names"]
    num_classes = artifact["num_classes"]
    img_size = artifact.get("img_size", 224)

    from torchvision.models import resnet18

    model = resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(artifact["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_dataset = SimpleImageDataset(data_root / "valid", transform)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    return all_labels, all_preds, class_names


def load_lgbm_result(model_path: Path, feature_csv: Path) -> tuple[list[int], list[int], dict[int, str]]:
    artifact = joblib.load(model_path)
    clf = artifact["model"]
    label_map = artifact["label_map"]
    feature_names = artifact["feature_names"]

    df = pd.read_csv(feature_csv)
    val_df = df[df["split"] == "valid"]
    X = val_df[feature_names]
    y_true = val_df["label"].astype(int).tolist()
    y_pred = clf.predict(X).astype(int).tolist()

    id_to_name = {v: k for k, v in label_map.items()}
    return y_true, y_pred, id_to_name


def summarize(y_true: list[int], y_pred: list[int], names: dict[int, str] | list[str]) -> dict[str, Any]:
    if isinstance(names, list):
        id_to_name = {i: n for i, n in enumerate(names)}
        class_names = names
    else:
        id_to_name = names
        class_names = [id_to_name[i] for i in sorted(id_to_name)]

    report = classification_report(
        y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
    )
    return {
        "accuracy": report["accuracy"],
        "f1_macro": report["macro avg"]["f1-score"],
        "f1_weighted": report["weighted avg"]["f1-score"],
        "per_class": {
            name: {
                "precision": report[name]["precision"],
                "recall": report[name]["recall"],
                "f1-score": report[name]["f1-score"],
                "support": report[name]["support"],
            }
            for name in class_names
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="对比电积镍分类方案")
    parser.add_argument("--data-root", type=Path, default=Path(r"E:/PythonProject/datasets/classify_4types_dianji"))
    parser.add_argument("--resnet-dir", type=Path, default=Path(r"E:/PythonProject/优化/classify_from_detection_v1/models/resnet_dianji_weighted_os"))
    parser.add_argument("--lgbm-model", type=Path, default=Path(r"E:/PythonProject/优化/classify_from_detection_v1/models/dianji_lgbm_weighted.pkl"))
    parser.add_argument("--features-csv", type=Path, default=Path(r"E:/PythonProject/优化/classify_from_detection_v1/dianji/features.csv"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=Path, default=Path(r"E:/PythonProject/优化/classify_from_detection_v1/models/dianji_comparison.json"))
    args = parser.parse_args()

    device = torch.device(args.device)

    lgbm_true, lgbm_pred, lgbm_names = load_lgbm_result(args.lgbm_model, args.features_csv)
    res_true, res_pred, res_names = load_resnet_result(args.resnet_dir, args.data_root, device)

    lgbm_summary = summarize(lgbm_true, lgbm_pred, lgbm_names)
    res_summary = summarize(res_true, res_pred, res_names)

    comparison = {
        "LightGBM_weighted": lgbm_summary,
        "ResNet_weighted_os": res_summary,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)

    print("=== 电积镍 4 类分类方案对比 ===\n")
    print(f"{'方案':<20} {'Acc':<8} {'F1-macro':<10} {'F1-weighted':<12}")
    for name, s in comparison.items():
        print(f"{name:<20} {s['accuracy']:<8.4f} {s['f1_macro']:<10.4f} {s['f1_weighted']:<12.4f}")

    print("\n各类别 F1:")
    classes = list(comparison["ResNet_weighted_os"]["per_class"].keys())
    print(f"{'类别':<10} {'LGBM F1':<10} {'ResNet F1':<10}")
    for c in classes:
        l = comparison["LightGBM_weighted"]["per_class"].get(c, {}).get("f1-score", 0.0)
        r = comparison["ResNet_weighted_os"]["per_class"].get(c, {}).get("f1-score", 0.0)
        print(f"{c:<10} {l:<10.4f} {r:<10.4f}")

    print(f"\n详细结果已保存: {args.output}")


if __name__ == "__main__":
    main()
