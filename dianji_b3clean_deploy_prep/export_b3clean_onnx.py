# -*- coding: utf-8 -*-
"""A 方案关卡②：B3-clean 部署 ONNX 导出 + PT↔ONNX 迷你一致性校验。

- 从 B3-clean best.pt（SHA 见输出）导出 640 固定输入 ONNX，产物与部署契约候选放本目录；
- 不替换、不触碰现役部署 ONNX（901ccf60…）；
- 在电积镍样本图上做 PT vs ONNX 逐框 IoU 匹配（同类 IoU≥0.5 匹配率 + 框数差），
  作为导出忠实性冒烟；完整零容差复核在 WebAPI 集成后进行。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PT_SRC = Path("E:/PythonProject/models/runs/dianjinie0211_medium_yolo11m_640_kaggle_seed0_cm0_d0211clean1/weights/best.pt")
SAMPLE_CANDIDATES = [
    Path("E:/PythonProject/datasets/Dianjinie0211/valid/images"),
    Path("E:/PythonProject/datasets/Dianjinie0211/images/val"),
]
N_SAMPLES = 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    from ultralytics import YOLO

    HERE.mkdir(parents=True, exist_ok=True)
    pt_local = HERE / "b3clean_best.pt"
    if not pt_local.exists():
        shutil.copy2(PT_SRC, pt_local)
    pt_sha = sha256(pt_local)
    print(f"best.pt sha256={pt_sha}", flush=True)

    print("[1/3] 导出 ONNX（imgsz=640）...", flush=True)
    model = YOLO(str(pt_local))
    onnx_path = Path(model.export(format="onnx", imgsz=640, simplify=True))
    onnx_final = HERE / "dianji0211_y11m640_b3clean.onnx"
    if onnx_path.resolve() != onnx_final.resolve():
        shutil.copy2(onnx_path, onnx_final)
    onnx_sha = sha256(onnx_final)
    print(f"ONNX sha256={onnx_sha}", flush=True)

    print(f"[2/3] 采样 {N_SAMPLES} 张电积镍图...", flush=True)
    sample_dir = next((p for p in SAMPLE_CANDIDATES if p.is_dir()), None)
    if sample_dir is None:
        raise FileNotFoundError(f"未找到样本目录，尝试过: {SAMPLE_CANDIDATES}")
    images = sorted(p for p in sample_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})[:N_SAMPLES]
    print(f"样本目录: {sample_dir}，实际 {len(images)} 张", flush=True)

    print("[3/3] PT vs ONNX 逐框匹配...", flush=True)
    pt_model = YOLO(str(pt_local))
    onnx_model = YOLO(str(onnx_final), task="detect")
    report = {"generated": datetime.now().isoformat(timespec="seconds"), "pt_sha256": pt_sha,
              "onnx_sha256": onnx_sha, "imgsz": 640, "images": []}
    rates = []
    for img in images:
        kw = dict(imgsz=640, conf=0.25, iou=0.45, device="cpu", verbose=False)
        a = pt_model.predict(str(img), **kw)[0]
        b = onnx_model.predict(str(img), **kw)[0]
        ta = a.boxes.xyxy.cpu().numpy() if a.boxes is not None else np.zeros((0, 4))
        ca = a.boxes.cls.cpu().numpy() if a.boxes is not None else np.zeros((0,))
        tb = b.boxes.xyxy.cpu().numpy() if b.boxes is not None else np.zeros((0, 4))
        cb = b.boxes.cls.cpu().numpy() if b.boxes is not None else np.zeros((0,))
        matched = 0
        used = set()
        for i in range(len(ta)):
            best, bj = 0.0, -1
            for j in range(len(tb)):
                if j in used or ca[i] != cb[j]:
                    continue
                x = max(0.0, min(ta[i][2], tb[j][2]) - max(ta[i][0], tb[j][0]))
                y = max(0.0, min(ta[i][3], tb[j][3]) - max(ta[i][1], tb[j][1]))
                inter = x * y
                union = (ta[i][2]-ta[i][0])*(ta[i][3]-ta[i][1]) + (tb[j][2]-tb[j][0])*(tb[j][3]-tb[j][1]) - inter
                v = inter/union if union > 0 else 0.0
                if v > best:
                    best, bj = v, j
            if best >= 0.5:
                matched += 1
                used.add(bj)
        rate = matched / len(ta) if len(ta) else 1.0
        rates.append(rate)
        report["images"].append({"image": img.name, "pt_boxes": int(len(ta)), "onnx_boxes": int(len(tb)),
                                 "matched_iou50": matched, "match_rate": round(rate, 5)})
        print(f"  {img.name}: pt={len(ta)} onnx={len(tb)} matched={matched} rate={rate:.3f}", flush=True)

    report["mean_match_rate"] = round(float(np.mean(rates)), 5)
    (HERE / "export_parity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print(f"DONE mean_match_rate={report['mean_match_rate']} -> {HERE/'export_parity_report.json'}", flush=True)


if __name__ == "__main__":
    main()
