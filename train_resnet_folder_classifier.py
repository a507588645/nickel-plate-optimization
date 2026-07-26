#!/usr/bin/env python3
"""
基于文件夹分类数据训练 ResNet 等级分类器。

输入：ImageFolder 格式数据集（如 classify_4types_dianji）
输出：模型、类别映射、训练曲线、评估指标

改进：
- 自动类别权重处理类别不平衡
- WeightedRandomSampler 训练集过采样
- 数据增强 + ImageNet 预训练微调
- 早停基于验证 F1-macro
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.models import resnet18, ResNet18_Weights

warnings.filterwarnings("ignore", category=FutureWarning)


CLASS_NAMES_DIANJI = ["3#", "9950", "9996", "bad"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于 ImageFolder 训练 ResNet 等级分类器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"E:/PythonProject/datasets/classify_4types_dianji"),
        help="ImageFolder 根目录（含 train/valid）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(r"E:/PythonProject/优化/classify_from_detection_v1/models/resnet_dianji"),
        help="输出目录",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        nargs="+",
        default=None,
        help="按文件夹顺序指定类别名；默认自动从目录读取",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="输入尺寸",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="批次大小",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=60,
        help="最大 epoch",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=12,
        help="验证 F1-macro 不提升则早停",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="学习率",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="L2 正则",
    )
    parser.add_argument(
        "--unfreeze-epoch",
        type=int,
        default=5,
        help="前几 epoch 冻结 backbone，之后解冻",
    )
    parser.add_argument(
        "--oversample",
        action="store_true",
        help="训练集使用 WeightedRandomSampler 过采样",
    )
    parser.add_argument(
        "--class-weight",
        action="store_true",
        help="交叉熵损失使用类别权重",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="计算设备：auto/cpu/cuda",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers；Windows 建议 0",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(img_size: int, is_train: bool) -> T.Compose:
    if is_train:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.RandomVerticalFlip(p=0.3),
                T.RandomRotation(degrees=15),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return T.Compose(
        [
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def compute_class_weights(dataset: ImageFolder) -> torch.Tensor:
    counts = np.bincount([s[1] for s in dataset.samples], minlength=len(dataset.classes))
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(dataset.classes)
    return torch.tensor(weights, dtype=torch.float32)


def build_sampler(dataset: ImageFolder) -> WeightedRandomSampler:
    targets = np.array([s[1] for s in dataset.samples])
    counts = np.bincount(targets, minlength=len(dataset.classes))
    sample_weights = 1.0 / (counts[targets] + 1e-6)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float32),
        num_samples=len(dataset),
        replacement=True,
    )


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    class_names: list[str],
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True
    )
    cm = confusion_matrix(all_labels, all_preds)
    return {
        "loss": total_loss / len(all_labels),
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "confusion_matrix": cm.tolist(),
        "preds": all_preds,
        "labels": all_labels,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available()
        else "cpu"
    )
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dir = args.data_root / "train"
    val_dir = args.data_root / "valid"

    train_dataset = ImageFolder(train_dir, transform=build_transforms(args.img_size, True))
    val_dataset = ImageFolder(val_dir, transform=build_transforms(args.img_size, False))

    # 保证 train/valid 类别顺序一致
    assert train_dataset.classes == val_dataset.classes, "train/valid 类别不一致"
    class_names = args.class_names or train_dataset.classes
    print(f"类别顺序: {train_dataset.classes} -> {class_names}")

    sampler = build_sampler(train_dataset) if args.oversample else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # 模型：预训练 ResNet18，替换最后一层
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    set_requires_grad(model, False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model = model.to(device)

    class_weights = None
    if args.class_weight:
        class_weights = compute_class_weights(train_dataset).to(device)
        print(f"类别权重: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.fc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=4
    )

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history: list[dict[str, float]] = []

    for epoch in range(args.epochs):
        if epoch == args.unfreeze_epoch:
            set_requires_grad(model, True)
            optimizer = optim.AdamW(model.parameters(), lr=args.lr / 10, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=4
            )
            print(f"Epoch {epoch}: 解冻 backbone，优化器重新初始化")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device, class_names)

        scheduler.step(val_metrics["f1_macro"])

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "val_f1_macro": val_metrics["f1_macro"],
                "val_f1_weighted": val_metrics["f1_weighted"],
            }
        )

        print(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        if val_metrics["f1_macro"] > best_f1:
            best_f1 = val_metrics["f1_macro"]
            best_epoch = epoch + 1
            patience_counter = 0
            artifact = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "folder_classes": train_dataset.classes,
                "label_map": {name: idx for idx, name in enumerate(class_names)},
                "img_size": args.img_size,
                "num_classes": len(class_names),
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_f1,
                "args": vars(args),
            }
            torch.save(artifact, args.output_dir / "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"早停于 epoch {epoch+1}，最佳 epoch {best_epoch}，val_f1_macro={best_f1:.4f}")
            break

    # 最终评估最佳模型
    artifact = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(artifact["model_state_dict"])
    final_metrics = evaluate(model, val_loader, criterion, device, class_names)

    summary = {
        "seed": args.seed,
        "class_names": class_names,
        "folder_classes": train_dataset.classes,
        "class_distribution": {
            "train": {class_names[i]: int(c) for i, c in enumerate(np.bincount([s[1] for s in train_dataset.samples], minlength=len(class_names)))},
            "valid": {class_names[i]: int(c) for i, c in enumerate(np.bincount([s[1] for s in val_dataset.samples], minlength=len(class_names)))},
        },
        "best_epoch": best_epoch,
        "val_accuracy": final_metrics["accuracy"],
        "val_f1_macro": final_metrics["f1_macro"],
        "val_f1_weighted": final_metrics["f1_weighted"],
        "classification_report": final_metrics["report"],
        "confusion_matrix": final_metrics["confusion_matrix"],
        "history": history,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 保存特征重要性：这里用最后一层权重的类间范数作为参考
    fc_weight = model.fc.weight.detach().cpu().numpy()
    importance = np.linalg.norm(fc_weight, axis=1)
    pd.DataFrame({"class": class_names, "fc_weight_norm": importance}).to_csv(
        args.output_dir / "fc_weight_norm.csv", index=False, encoding="utf-8-sig"
    )

    print("\n=== 最终验证结果 ===")
    print(f"Accuracy: {final_metrics['accuracy']:.4f}")
    print(f"F1-macro: {final_metrics['f1_macro']:.4f}")
    print(f"F1-weighted: {final_metrics['f1_weighted']:.4f}")
    print(f"混淆矩阵:\n{np.array(final_metrics['confusion_matrix'])}")
    print(f"结果已保存到: {args.output_dir}")


if __name__ == "__main__":
    main()
