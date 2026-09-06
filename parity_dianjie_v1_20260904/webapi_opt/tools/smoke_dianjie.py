"""通过 FastAPI /dianjiEclass 验证 models.yaml 配置的电解镍检测器。"""
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


CLASS_COUNT = 18
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def _decode_bgr(image_path: Path) -> np.ndarray:
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取样本图像: {image_path}")
    return image


def _validate_api_result(payload: dict, width: int, height: int) -> tuple[int, list[int]]:
    if payload.get("code") != 1:
        raise AssertionError(f"API 返回失败: {payload}")

    defects = payload.get("data", {}).get("defects", [])
    if not isinstance(defects, list):
        raise AssertionError(f"API defects 不是列表: {payload}")

    labels: list[int] = []
    for defect in defects:
        label = int(defect["label"])
        if not 0 <= label < CLASS_COUNT:
            raise AssertionError(f"缺陷标签超出 18 类范围: {label}")

        cx = float(defect["cx"])
        cy = float(defect["cy"])
        box_width = float(defect["w"])
        box_height = float(defect["h"])
        values = np.asarray([cx, cy, box_width, box_height], dtype=np.float64)
        if not np.isfinite(values).all() or box_width <= 0 or box_height <= 0:
            raise AssertionError(f"缺陷框数值异常: {defect}")

        x1 = cx - box_width / 2
        y1 = cy - box_height / 2
        x2 = cx + box_width / 2
        y2 = cy + box_height / 2
        if not (0 <= x1 <= x2 <= width and 0 <= y1 <= y2 <= height):
            raise AssertionError(f"缺陷框坐标超出图像范围: {defect}, image=({width}, {height})")
        labels.append(label)

    return len(defects), labels


def _sample_paths(sample_dir: Path, limit: int) -> list[Path]:
    paths = sorted(
        path for path in sample_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) < limit:
        raise ValueError(f"样本图像不足 {limit} 张: {sample_dir}")
    return paths[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_dir", type=Path, help="开发集 images 目录")
    parser.add_argument("--limit", type=int, default=5, choices=range(3, 6), help="抽取 3-5 张图像")
    args = parser.parse_args()

    sample_dir = args.sample_dir.expanduser().resolve()
    if not sample_dir.is_dir():
        raise ValueError(f"样本目录不存在: {sample_dir}")

    # 跳过未替换的电积镍和分割模型，只加载本次冒烟所需的真实模型。
    os.environ["JINCHUAN_PRELOAD_MODELS"] = "false"
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from adapters import create_detector
    from fastapi_app import app
    from inference import models as model_state
    import requests
    import uvicorn

    sample_paths = _sample_paths(sample_dir, args.limit)

    model_state.load_dianjie_detector()
    model_state.load_resnet()
    detector_config = model_state.MODELS_CONFIG["detectors"]["dianjie"]
    direct_detector = create_detector(
        detector_config["kind"],
        weights=model_state._resolved_path(detector_config["weights"]),
        device="cuda" if model_state.use_gpu else "cpu",
        threshold=detector_config.get("threshold", 0.25),
        **model_state._detector_kwargs(detector_config),
    )

    direct_rows = direct_detector(str(sample_paths[0])).xyxy[0].numpy()
    direct_count = int(direct_rows.shape[0])
    print(json.dumps({"direct_image": str(sample_paths[0]), "direct_count": direct_count}, ensure_ascii=False))

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

    api_counts: list[int] = []
    try:
        for image_path in sample_paths:
            image = _decode_bgr(image_path)
            response = requests.post(
                f"http://127.0.0.1:{port}/dianjiEclass",
                json={"image": str(image_path)},
                timeout=300,
            )
            if response.status_code != 200:
                raise AssertionError(f"HTTP 状态异常: {response.status_code}, body={response.text}")

            payload = response.json()
            count, labels = _validate_api_result(payload, image.shape[1], image.shape[0])
            api_counts.append(count)
            print(
                json.dumps(
                    {
                        "image": str(image_path),
                        "http_status": response.status_code,
                        "code": payload["code"],
                        "defect_count": count,
                        "labels": sorted(set(labels)),
                    },
                    ensure_ascii=False,
                )
            )
    finally:
        server.should_exit = True
        server_thread.join(timeout=30)
        if server_thread.is_alive():
            raise RuntimeError("Uvicorn 未能正常退出")

    if abs(api_counts[0] - direct_count) > 2:
        raise AssertionError(
            f"API 与直接推理框数差异超过 2: api={api_counts[0]}, direct={direct_count}"
        )
    print(
        json.dumps(
            {
                "sample_count": len(sample_paths),
                "api_counts": api_counts,
                "direct_count_first_sample": direct_count,
                "first_sample_count_delta": api_counts[0] - direct_count,
                "status": "PASS",
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
