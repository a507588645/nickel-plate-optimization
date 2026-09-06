"""通过 FastAPI /dianjiclass 验证电积镍 LightGBM 等级模型。"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}
GRADE_NAMES = ("3#", "9950", "9996", "bad")
DEFAULT_SAMPLE_ROOT = Path(r"E:\PythonProject\archives\dianjinie_grade_classification_v1\images")


def _decode_bgr(image_path: Path) -> np.ndarray:
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取样本图像: {image_path}")
    return image


def _sample_paths(sample_root: Path, limit: int) -> list[Path]:
    grouped: dict[str, list[Path]] = {grade: [] for grade in GRADE_NAMES}
    for path in sorted(sample_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and path.parent.name in grouped:
            grouped[path.parent.name].append(path)

    selected = [paths[0] for paths in grouped.values() if paths]
    if len(selected) < limit:
        selected.extend(
            path
            for paths in grouped.values()
            for path in paths[1:]
            if path not in selected
        )
    if len(selected) < limit:
        raise ValueError(f"样本图像不足 {limit} 张: {sample_root}")
    return selected[:limit]


def _validate_api_result(payload: dict, threshold: float) -> tuple[int, str, float]:
    if payload.get("code") != 1:
        raise AssertionError(f"API 返回失败: {payload}")
    data = payload.get("data", {})
    if type(data.get("type")) is not int or data["type"] not in {1, 2, 3, 4}:
        raise AssertionError(f"API type 无效: {payload}")
    if not isinstance(data.get("defects"), list):
        raise AssertionError(f"API defects 不是列表: {payload}")
    for defect in data["defects"]:
        label = defect.get("label") if isinstance(defect, dict) else None
        if type(label) is not int or not 0 <= label < 18:
            raise AssertionError(f"API defect label 无效，必须是 0-17 的 int: {payload}")
    if not isinstance(data.get("grade"), str):
        raise AssertionError(f"API grade 无效: {payload}")
    if not isinstance(data.get("confidence"), float) or not np.isfinite(data["confidence"]):
        raise AssertionError(f"API confidence 无效: {payload}")
    if not isinstance(data.get("manual_review"), bool):
        raise AssertionError(f"API manual_review 无效: {payload}")
    if data["manual_review"] != (data["confidence"] < threshold):
        raise AssertionError(f"API manual_review 与阈值不一致: {payload}")
    if data.get("grade_status") != "unconfirmed_pending_enterprise":
        raise AssertionError(f"API grade_status 无效: {payload}")
    if data.get("grade_source") != "lgbm_y11m640_v1":
        raise AssertionError(f"API 未使用 LightGBM 等级模型: {payload}")
    return len(data["defects"]), data["grade"], data["confidence"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_root", type=Path, nargs="?", default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--limit", type=int, default=4, choices=range(3, 6), help="抽取 3-5 张图像")
    args = parser.parse_args()

    sample_root = args.sample_root.expanduser().resolve()
    if not sample_root.is_dir():
        raise ValueError(f"样本目录不存在: {sample_root}")
    sample_paths = _sample_paths(sample_root, args.limit)

    os.environ["JINCHUAN_PRELOAD_MODELS"] = "false"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from fastapi_app import app
    from inference import models as model_state
    import requests
    import uvicorn

    model_state.load_dianjie_detector()
    model_state.load_dianji_grade_model()
    model_state.load_resnet()
    model_state.load_dianji_detector()
    if model_state.dianji_grade_model is None:
        raise RuntimeError("LightGBM 等级模型未加载，无法执行冒烟测试")
    threshold = float(model_state.dianji_grade_threshold)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
    )
    server_thread = threading.Thread(target=server.run, daemon=True)
    server_thread.start()
    deadline = time.monotonic() + 30
    while not server.started:
        if not server_thread.is_alive():
            raise RuntimeError("Uvicorn 启动失败")
        if time.monotonic() >= deadline:
            server.should_exit = True
            server_thread.join(timeout=30)
            raise TimeoutError("Uvicorn 启动超时")
        time.sleep(0.1)

    try:
        for image_path in sample_paths:
            response = requests.post(
                f"http://127.0.0.1:{port}/dianjiclass",
                json={"image": str(image_path)},
                timeout=300,
            )
            if response.status_code != 200:
                raise AssertionError(f"HTTP 状态异常: {response.status_code}, body={response.text}")
            payload = response.json()
            defect_count, grade, confidence = _validate_api_result(payload, threshold)
            print(
                json.dumps(
                    {
                        "image": str(image_path),
                        "code": payload["code"],
                        "type": payload["data"]["type"],
                        "grade": grade,
                        "confidence": confidence,
                        "defect_count": defect_count,
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        server.should_exit = True
        server_thread.join(timeout=30)
        if server_thread.is_alive():
            raise RuntimeError("Uvicorn 未能正常退出")

    print(json.dumps({"sample_count": len(sample_paths), "status": "PASS"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
