#!/usr/bin/env python3
"""
基于检测特征训练等级分类器（LightGBM）。

输入：build_grade_dataset.py 生成的 features.csv
输出：模型、特征列表、类别映射、评估指标

改进项：
- 支持类别权重处理类别不均衡
- 支持训练集过采样（random / smote）
- 可选拼接 ResNet18 整板图像特征
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold


METADATA_COLS = {
    "image_path",
    "split",
    "class_name",
    "label",
    "task",
    "image_name",
    "img_w",
    "img_h",
    "plate_area",
    "total_defect_count",
    "total_defect_area",
    "total_defect_area_ratio",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="基于检测特征训练 LightGBM 等级分类器",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:/PythonProject/优化/classify_from_detection_v1"),
        help="build_grade_dataset.py 输出根目录",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        choices=["dianjie", "dianji"],
        default=["dianjie", "dianji"],
        help="要训练的任务",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="分层交叉验证折数",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
        help="LightGBM 学习率",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=200,
        help="LightGBM 树数量",
    )
    parser.add_argument(
        "--num-leaves",
        type=int,
        default=31,
        help="LightGBM 叶子数",
    )
    parser.add_argument(
        "--use-class-weights",
        action="store_true",
        help="使用 LightGBM 类别权重处理类别不均衡",
    )
    parser.add_argument(
        "--oversample",
        type=str,
        choices=["", "smote", "random"],
        default="",
        help="对训练集少数类过采样（smote/random），验证集保持原分布",
    )
    parser.add_argument(
        "--add-image-features",
        action="store_true",
        help="额外提取 ResNet18 整板图像特征（512 维）与检测特征拼接",
    )
    parser.add_argument(
        "--image-resize",
        type=int,
        nargs=2,
        default=[224, 224],
        help="图像特征提取输入尺寸 (宽 高)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="图像特征提取或推理设备，如 cpu / cuda:0",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只加载数据并打印信息，不训练",
    )
    parser.add_argument(
        "--output-suffix",
        type=str,
        default="",
        help="模型文件名后缀，用于区分不同实验，例如 _weighted",
    )
    return parser.parse_args()


def load_data(dataset_root: Path, task: str) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """加载特征 CSV，返回 (df, feature_names, label_map)。"""
    feat_path = dataset_root / task / "features.csv"
    if not feat_path.exists():
        raise FileNotFoundError(f"特征文件不存在: {feat_path}")

    df = pd.read_csv(feat_path)

    # 自动识别特征列：排除已知元数据列
    feature_cols = [c for c in df.columns if c not in METADATA_COLS]

    # 过滤掉特征全为 NaN 的列
    valid_cols = [c for c in feature_cols if df[c].notna().any()]
    if len(valid_cols) != len(feature_cols):
        dropped = set(feature_cols) - set(valid_cols)
        warnings.warn(f"{task} 丢弃全空特征列: {dropped}", stacklevel=2)
        feature_cols = valid_cols

    # 从 metadata.json 读取标签映射
    meta_path = dataset_root / "metadata.json"
    label_map: dict[str, int] = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        label_map = meta.get("tasks", {}).get(task, {}).get("label_map", {})

    return df, feature_cols, label_map


def split_train_val(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    按 features.csv 中的 'split' 列划分为训练集与验证集。
    若不存在有效划分，则返回 X_all/y_all 作为训练集，验证集为空。
    """
    df = df.copy()
    df["label"] = df["label"].astype(int)

    if "split" in df.columns and df["split"].nunique() > 1:
        df["split_norm"] = df["split"].astype(str).str.lower().replace({"val": "valid"})
        train_df = df[df["split_norm"] == "train"]
        valid_df = df[df["split_norm"].isin(["valid", "test"])]
        if len(train_df) > 0 and len(valid_df) > 0:
            X_train = train_df[feature_cols].fillna(0.0)
            y_train = train_df["label"]
            X_val = valid_df[feature_cols].fillna(0.0)
            y_val = valid_df["label"]
            return X_train, X_val, y_train, y_val

    X = df[feature_cols].fillna(0.0)
    y = df["label"]
    return X, pd.DataFrame(), y, pd.Series(dtype=int)


def compute_class_weight(y: pd.Series) -> dict[int, float]:
    """按样本数倒数计算类别权重。"""
    counts = y.value_counts().sort_index()
    total = len(y)
    return {int(cls): total / (len(counts) * cnt) for cls, cnt in counts.items()}


def random_oversample(
    X: pd.DataFrame, y: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """随机过采样：将少数类复制到与多数类数量相同。"""
    rng = np.random.default_rng(seed)
    df = pd.concat([X, y.rename("_y_")], axis=1)
    max_count = y.value_counts().max()
    chunks = []
    for cls in sorted(y.unique()):
        sub = df[df["_y_"] == cls]
        if len(sub) < max_count:
            n_extra = max_count - len(sub)
            extra = sub.iloc[rng.integers(0, len(sub), size=n_extra)]
            sub = pd.concat([sub, extra], ignore_index=True)
        chunks.append(sub)
    balanced = pd.concat(chunks, ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)
    return balanced.drop(columns=["_y_"]), balanced["_y_"]


def smote_oversample(
    X: pd.DataFrame, y: pd.Series, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """SMOTE 过采样（需安装 imblearn）。"""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise ImportError("SMOTE 需要 imblearn，请运行: pip install imbalanced-learn") from e
    smote = SMOTE(random_state=seed, k_neighbors=min(5, y.value_counts().min() - 1))
    X_res, y_res = smote.fit_resample(X, y)
    return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res, name=y.name)


def apply_oversample(
    X: pd.DataFrame, y: pd.Series, method: str, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    if method == "random":
        return random_oversample(X, y, seed)
    if method == "smote":
        return smote_oversample(X, y, seed)
    return X, y


def extract_image_features(
    df: pd.DataFrame,
    resize_wh: tuple[int, int],
    device: str,
) -> np.ndarray:
    """用 ResNet18 提取整板图像特征（avgpool 后 512 维）。"""
    from torchvision.models import resnet18, ResNet18_Weights
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)

    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(resize_wh),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    feats = []
    for path in df["image_path"]:
        img = cv2.imread(str(path))
        if img is None:
            feats.append(np.zeros(512, dtype=np.float32))
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            feat = model(tensor).squeeze().cpu().numpy()
        feats.append(feat)
    return np.vstack(feats)


def create_classifier(
    n_classes: int,
    args: argparse.Namespace,
    seed: int,
    class_weight: dict[int, float] | None,
) -> LGBMClassifier:
    """创建 LightGBM 分类器。"""
    params: dict[str, Any] = {
        "boosting_type": "gbdt",
        "num_leaves": args.num_leaves,
        "learning_rate": args.learning_rate,
        "n_estimators": args.n_estimators,
        "random_state": seed,
        "verbose": -1,
        "n_jobs": -1,
    }
    if class_weight is not None and len(class_weight) > 0:
        params["class_weight"] = class_weight
    if n_classes == 2:
        params["objective"] = "binary"
        params["metric"] = "binary_logloss"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = n_classes
        params["metric"] = "multi_logloss"
    return LGBMClassifier(**params)


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    n_classes: int,
    args: argparse.Namespace,
    class_weight: dict[int, float] | None,
) -> dict[str, Any]:
    """分层 K 折交叉验证；只在训练折内做过采样。"""
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_scores = []
    all_y_true = []
    all_y_pred = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if args.oversample:
            X_train, y_train = apply_oversample(X_train, y_train, args.oversample, args.seed + fold)

        clf = create_classifier(n_classes, args, args.seed, class_weight)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
        f1_weighted = f1_score(y_val, y_pred, average="weighted", zero_division=0)

        fold_scores.append({
            "fold": fold,
            "accuracy": acc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "n_train": len(X_train),
            "n_val": len(X_val),
        })
        all_y_true.extend(y_val.tolist())
        all_y_pred.extend(y_pred.tolist())

    overall = {
        "accuracy": accuracy_score(all_y_true, all_y_pred),
        "f1_macro": f1_score(all_y_true, all_y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(all_y_true, all_y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(all_y_true, all_y_pred).tolist(),
        "classification_report": classification_report(
            all_y_true, all_y_pred, output_dict=True, zero_division=0
        ),
    }
    return {"folds": fold_scores, "overall": overall}


def train_final(
    X: pd.DataFrame,
    y: pd.Series,
    n_classes: int,
    args: argparse.Namespace,
    class_weight: dict[int, float] | None,
) -> LGBMClassifier:
    """在全部训练数据上训练最终模型。"""
    if args.oversample:
        X, y = apply_oversample(X, y, args.oversample, args.seed)
    clf = create_classifier(n_classes, args, args.seed, class_weight)
    clf.fit(X, y)
    return clf


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_root = args.dataset_root / "models"
    output_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {"seed": args.seed, "tasks": {}}

    for task in args.tasks:
        print(f"\n=== 训练任务: {task} ===")
        df, feature_cols, label_map = load_data(args.dataset_root, task)

        # 可选：追加整板图像特征
        image_feature_cols: list[str] = []
        if args.add_image_features:
            print("提取 ResNet18 整板图像特征...")
            img_feats = extract_image_features(df, tuple(args.image_resize), args.device)
            n_img = img_feats.shape[1]
            image_feature_cols = [f"img_feat_{i}" for i in range(n_img)]
            img_df = pd.DataFrame(img_feats, columns=image_feature_cols, index=df.index)
            df = pd.concat([df, img_df], axis=1)
            feature_cols = feature_cols + image_feature_cols

        X_train, X_val, y_train, y_val = split_train_val(df, feature_cols)
        y_all = df["label"].astype(int)
        n_classes = y_all.nunique()
        class_distribution = y_all.value_counts().sort_index().to_dict()

        class_weight = compute_class_weight(y_train) if args.use_class_weights else None

        print(f"总样本数: {len(df)}")
        print(f"训练集: {len(X_train)}")
        print(f"验证集: {len(X_val)}")
        print(f"检测特征数: {len(feature_cols) - len(image_feature_cols)}")
        print(f"图像特征数: {len(image_feature_cols)}")
        print(f"总特征数: {len(feature_cols)}")
        print(f"类别数: {n_classes}")
        print(f"类别分布: {class_distribution}")
        print(f"标签映射: {label_map}")
        if class_weight:
            print(f"类别权重: {class_weight}")

        if args.dry_run:
            print("--dry-run 模式：跳过训练")
            continue

        cv_result = cross_validate(X_train, y_train, n_classes, args, class_weight)
        print(f"\n训练集 {args.n_folds} 折交叉验证:")
        for f in cv_result["folds"]:
            print(
                f"  Fold {f['fold']}: acc={f['accuracy']:.4f}, "
                f"f1_macro={f['f1_macro']:.4f}, f1_weighted={f['f1_weighted']:.4f}, "
                f"n_train={f['n_train']}"
            )
        print(
            f"\nCV 整体: acc={cv_result['overall']['accuracy']:.4f}, "
            f"f1_macro={cv_result['overall']['f1_macro']:.4f}, "
            f"f1_weighted={cv_result['overall']['f1_weighted']:.4f}"
        )

        print("\n训练最终模型...")
        clf = train_final(X_train, y_train, n_classes, args, class_weight)

        val_result = None
        if len(X_val) > 0:
            y_val_pred = clf.predict(X_val)
            val_result = {
                "accuracy": accuracy_score(y_val, y_val_pred),
                "f1_macro": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
                "f1_weighted": f1_score(y_val, y_val_pred, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(y_val, y_val_pred).tolist(),
                "classification_report": classification_report(
                    y_val, y_val_pred, output_dict=True, zero_division=0
                ),
            }
            print(
                f"验证集: acc={val_result['accuracy']:.4f}, "
                f"f1_macro={val_result['f1_macro']:.4f}, "
                f"f1_weighted={val_result['f1_weighted']:.4f}"
            )

        importance = pd.DataFrame({
            "feature": feature_cols,
            "importance": clf.feature_importances_,
        }).sort_values("importance", ascending=False)

        suffix = args.output_suffix or ""
        model_path = output_root / f"{task}_lgbm{suffix}.pkl"
        joblib.dump(
            {
                "model": clf,
                "feature_names": feature_cols,
                "label_map": label_map,
                "n_classes": n_classes,
                "class_distribution": class_distribution,
                "cv_result": cv_result,
                "val_result": val_result,
                "importance": importance.to_dict("records"),
                "config": {
                    "learning_rate": args.learning_rate,
                    "n_estimators": args.n_estimators,
                    "num_leaves": args.num_leaves,
                    "n_folds": args.n_folds,
                    "seed": args.seed,
                    "use_class_weights": args.use_class_weights,
                    "oversample": args.oversample,
                    "add_image_features": args.add_image_features,
                },
            },
            model_path,
        )
        print(f"模型已保存: {model_path}")

        imp_path = output_root / f"{task}_feature_importance{suffix}.csv"
        importance.to_csv(imp_path, index=False, encoding="utf-8-sig")
        print(f"特征重要性已保存: {imp_path}")

        summary["tasks"][task] = {
            "n_samples": len(df),
            "n_train": len(X_train),
            "n_val": len(X_val),
            "n_features": len(feature_cols),
            "n_image_features": len(image_feature_cols),
            "n_classes": n_classes,
            "class_distribution": class_distribution,
            "cv_accuracy": cv_result["overall"]["accuracy"],
            "cv_f1_macro": cv_result["overall"]["f1_macro"],
            "cv_f1_weighted": cv_result["overall"]["f1_weighted"],
            "val_accuracy": val_result["accuracy"] if val_result else None,
            "val_f1_macro": val_result["f1_macro"] if val_result else None,
            "val_f1_weighted": val_result["f1_weighted"] if val_result else None,
            "model_path": str(model_path),
        }

    summary_path = output_root / f"training_summary{args.output_suffix or ''}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n训练汇总已保存: {summary_path}")


if __name__ == "__main__":
    main()
