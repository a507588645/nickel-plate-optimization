#!/usr/bin/env python3
"""
轻量 CNN 基线，用于电积镍 4 类 folder 分类。

相比 ResNet18：
- 模型更小，CPU 训练/推理更快；
- 从头训练，不依赖 ImageNet 预训练权重；
- 可作为 ResNet 方案的快速对照。
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
import torch.nn.functional as F
import torchvision.transforms as T
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets import ImageFolder

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="轻量 CNN folder 分类基线",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=Path(r"E:/PythonProject/datasets/classify_4types_dianji"))
    parser.add_argument("--output-dir", type=Path, default=Path(r"E:/PythonProject/优化/classify_from_detection_v1/models/cnn_dianji"))
    parser.add_argument("--class-names", type=str, nargs="+", default=None)
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--oversample", action="store_true")
    parser.add_argument("--class-weight", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_transforms(img_size: int, is_train: bool) -> T.Compose:
    if is_train:
        return T.Compose(
            [
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.RandomRotation(10),
                T.ColorJitter(0.1, 0.1),
                T.ToTensor(),
            ]
        )
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor()])


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return self.fc(x)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, class_names: list[str]) -> dict[str, Any]:
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "report": report,
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = ImageFolder(args.data_root / "train", transform=build_transforms(args.img_size, True))
    val_ds = ImageFolder(args.data_root / "valid", transform=build_transforms(args.img_size, False))
    class_names = args.class_names or train_ds.classes
    print(f"类别: {class_names}")

    sampler = None
    if args.oversample:
        targets = np.array([s[1] for s in train_ds.samples])
        counts = np.bincount(targets, minlength=len(class_names))
        weights = 1.0 / (counts[targets] + 1e-6)
        sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.float32), len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = SmallCNN(len(class_names)).to(device)

    class_weights = None
    if args.class_weight:
        counts = np.bincount([s[1] for s in train_ds.samples], minlength=len(class_names))
        class_weights = torch.tensor(1.0 / (counts + 1e-6), dtype=torch.float32).to(device)
        class_weights = class_weights / class_weights.sum() * len(class_names)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_f1 = -1.0
    best_epoch = 0
    patience_counter = 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)

        metrics = evaluate(model, val_loader, device, class_names)
        scheduler.step(metrics["f1_macro"])

        train_loss = total_loss / len(train_ds)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_acc": metrics["accuracy"],
            "val_f1_macro": metrics["f1_macro"],
            "val_f1_weighted": metrics["f1_weighted"],
        })
        print(f"Epoch {epoch+1}/{args.epochs} loss={train_loss:.4f} acc={metrics['accuracy']:.4f} f1_macro={metrics['f1_macro']:.4f}")

        if metrics["f1_macro"] > best_f1:
            best_f1 = metrics["f1_macro"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
                "label_map": {name: i for i, name in enumerate(class_names)},
                "img_size": args.img_size,
                "num_classes": len(class_names),
                "best_epoch": best_epoch,
                "best_val_f1_macro": best_f1,
            }, args.output_dir / "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"早停，最佳 epoch={best_epoch}, f1_macro={best_f1:.4f}")
                break

    # 加载最佳模型保存 summary
    artifact = torch.load(args.output_dir / "best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(artifact["model_state_dict"])
    final = evaluate(model, val_loader, device, class_names)
    summary = {
        "seed": args.seed,
        "class_names": class_names,
        "best_epoch": best_epoch,
        "val_accuracy": final["accuracy"],
        "val_f1_macro": final["f1_macro"],
        "val_f1_weighted": final["f1_weighted"],
        "classification_report": final["report"],
        "confusion_matrix": final["confusion_matrix"],
        "history": history,
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n最终: acc={final['accuracy']:.4f} f1_macro={final['f1_macro']:.4f}")
    print(f"保存: {args.output_dir}")


if __name__ == "__main__":
    main()
