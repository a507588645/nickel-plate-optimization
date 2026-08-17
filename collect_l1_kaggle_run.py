#!/usr/bin/env python3
"""Verify, download, and safely archive one completed L1 Kaggle training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.services.kernels_api_service import (
    ApiGetKernelSessionStatusRequest,
    ApiListKernelSessionOutputRequest,
)


DEFAULT_RAW_OUTPUT_ROOT = Path(r"E:\PythonProject\models\kaggle_outputs")
DEFAULT_RUNS_ROOT = Path(r"E:\PythonProject\models\runs")
STANDARD_REQUIRED_RUN_FILES = (
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
)
FINAL_RELEASE_REQUIRED_RUN_FILES = (
    "weights/final_fixed_epoch.pt",
    "release_weight.json",
    "dataset_config/epoch_selection_summary.json",
)
FROZEN_TEST_REQUIRED_FILE = "dataset_config/test.txt"
CHUNK_SIZE = 1024 * 1024
API_TIMEOUT_SECONDS = (30, 90)
# The Kaggle content endpoint intermittently stalls on this network. Smaller
# verified ranges trade a little overhead for a resumable, reliable transfer.
RESULT_DOWNLOAD_CHUNK_BYTES = 512 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="校验、下载并归档一个已完成的 L1 Kaggle 训练结果，不覆盖已有 run。"
    )
    parser.add_argument("--kernel", required=True, help="Kaggle kernel，格式为 owner/slug。")
    parser.add_argument("--run-name", required=True, help="ZIP 内和 models/runs 下的预期 run 目录名。")
    parser.add_argument("--raw-output-root", type=Path, default=DEFAULT_RAW_OUTPUT_ROOT)
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--retries", type=int, default=10)
    parser.add_argument(
        "--development-only",
        action="store_true",
        help="归档仅含开发集 train/val 的版本；拒绝要求或读取 dataset_config/test.txt。",
    )
    parser.add_argument(
        "--final-release",
        action="store_true",
        help="要求最终全开发集发布权重、固定 epoch 选择和发布元数据；普通 Fold 实验不得传此参数。",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="只查询 Kaggle 状态和输出清单，不写文件。")
    mode.add_argument("--download", action="store_true", help="下载并校验 ZIP，但不归档到 models/runs。")
    mode.add_argument("--install", action="store_true", help="下载、校验并归档到新的 models/runs 目录。")
    return parser.parse_args()


def required_run_files(development_only: bool, final_release: bool = False) -> tuple[str, ...]:
    required = STANDARD_REQUIRED_RUN_FILES
    if final_release:
        required = (*required, *FINAL_RELEASE_REQUIRED_RUN_FILES)
    if development_only:
        return required
    return (*required, FROZEN_TEST_REQUIRED_FILE)


def split_kernel(kernel: str) -> tuple[str, str]:
    try:
        owner, slug = kernel.split("/", 1)
    except ValueError as error:
        raise ValueError("--kernel 必须是 owner/slug") from error
    if not owner or not slug or "/" in slug:
        raise ValueError("--kernel 必须是 owner/slug")
    return owner, slug


def bounded_sdk_session(api: KaggleApi) -> Any:
    """Yield a Kaggle client whose HTTP calls cannot wait indefinitely."""
    return api.build_kaggle_client()


def configure_sdk_timeout(client: Any) -> None:
    service = client.kernels.kernels_api_client._client
    service._init_session()
    original_send = service._session.send

    def send_with_timeout(request: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("timeout", API_TIMEOUT_SECONDS)
        return original_send(request, **kwargs)

    service._session.send = send_with_timeout


def kernel_status(api: KaggleApi, kernel: str) -> str:
    owner, slug = split_kernel(kernel)
    with bounded_sdk_session(api) as client:
        configure_sdk_timeout(client)
        status_request = ApiGetKernelSessionStatusRequest()
        status_request.user_name = owner
        status_request.kernel_slug = slug
        status_response = client.kernels.kernels_api_client.get_kernel_session_status(status_request)
    # Kaggle SDK exposes an enum string such as ``KernelWorkerStatus.COMPLETE``.
    return str(getattr(status_response, "status", "UNKNOWN")).rsplit(".", 1)[-1]


def query_kernel_output(api: KaggleApi, kernel: str) -> Any:
    """List full output metadata only for explicit diagnostic inspection."""
    owner, slug = split_kernel(kernel)
    with bounded_sdk_session(api) as client:
        configure_sdk_timeout(client)
        output_request = ApiListKernelSessionOutputRequest()
        output_request.user_name = owner
        output_request.kernel_slug = slug
        return client.kernels.kernels_api_client.list_kernel_session_output(output_request)


def direct_archive_download_url(kernel: str, archive_name: str) -> str:
    """Use Kaggle's small redirect endpoint instead of its large log payload."""
    owner, slug = split_kernel(kernel)
    return (
        "https://www.kaggle.com/api/v1/kernels/output/download/"
        f"{quote(owner, safe='')}/{quote(slug, safe='')}/{quote(archive_name, safe='')}"
    )


def curl_config_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\r", "").replace("\n", "")


def kaggle_credentials(api: KaggleApi) -> tuple[str, str]:
    username = str(api.config_values.get("username", ""))
    key = str(api.config_values.get("key", ""))
    if not username or not key:
        raise RuntimeError("Kaggle 凭据不可用，无法下载结果 ZIP")
    return username, key


def output_summary(status: str, response: Any) -> dict[str, Any]:
    files = []
    for item in getattr(response, "files", []) or []:
        files.append(
            {
                "file_name": str(getattr(item, "file_name", "")),
                "has_url": bool(getattr(item, "url", "")),
                "total_bytes": getattr(item, "total_bytes", None),
            }
        )
    return {
        "status": status,
        "has_log": bool(getattr(response, "log", "")),
        "files": files,
    }


def find_result_archive(response: Any, run_name: str) -> Any:
    expected_name = f"{run_name}_results.zip"
    candidates = [
        item
        for item in getattr(response, "files", []) or []
        if str(getattr(item, "file_name", "")) == expected_name and getattr(item, "url", "")
    ]
    if len(candidates) != 1:
        available = [str(getattr(item, "file_name", "")) for item in getattr(response, "files", []) or []]
        raise FileNotFoundError(f"未找到唯一结果压缩包 {expected_name}，当前输出：{available}")
    return candidates[0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def curl_config(username: str, key: str) -> str:
    return f'user = "{curl_config_value(f"{username}:{key}")}"\n'


def run_curl(command: list[str], config: str, timeout: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            input=config,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("curl 请求超时") from error


def remote_archive_size(curl: str, url: str, config: str, work_dir: Path) -> int:
    """Probe the content range through Kaggle's redirect endpoint."""
    headers_path = work_dir / f".archive-headers-{uuid.uuid4().hex}.txt"
    command = [
        curl,
        "--config",
        "-",
        "--fail",
        "--location",
        "--http1.1",
        "--range",
        "0-0",
        "--connect-timeout",
        "15",
        "--max-time",
        "60",
        "--dump-header",
        str(headers_path),
        "--output",
        "NUL",
        "--silent",
        "--show-error",
        url,
    ]
    try:
        completed = run_curl(command, config, timeout=75)
        headers = headers_path.read_text(encoding="iso-8859-1") if headers_path.is_file() else ""
    finally:
        headers_path.unlink(missing_ok=True)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"curl exit code {completed.returncode}"
        raise RuntimeError(f"无法读取远端 ZIP 长度：{detail}")
    matches = re.findall(r"(?im)^content-range:\s*bytes\s+\d+-\d+/(\d+)", headers)
    if not matches:
        raise RuntimeError("远端 ZIP 未返回 Content-Range，总长度不可验证")
    return int(matches[-1])


def download_resumable(url: str, target: Path, retries: int, username: str, key: str) -> None:
    """Download fixed verified ranges so a stalled long connection cannot corrupt a ZIP."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 0:
        return

    partial = target.with_name(f"{target.name}.part")
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("未找到 curl，无法对 Kaggle 结果 ZIP 执行断点续传")

    config = curl_config(username, key)
    total_size = remote_archive_size(curl, url, config, target.parent)
    if partial.exists() and partial.stat().st_size > total_size:
        raise ValueError(f"本地 .part 超过远端 ZIP 长度：{partial}")

    while (partial.stat().st_size if partial.exists() else 0) < total_size:
        offset = partial.stat().st_size if partial.exists() else 0
        length = min(RESULT_DOWNLOAD_CHUNK_BYTES, total_size - offset)
        end = offset + length - 1
        chunk_path = target.parent / f".{target.name}.chunk-{uuid.uuid4().hex}"
        command = [
            curl,
            "--config",
            "-",
            "--fail",
            "--location",
            "--http1.1",
            "--range",
            f"{offset}-{end}",
            "--connect-timeout",
            "15",
            "--max-time",
            "90",
            "--speed-limit",
            "1024",
            "--speed-time",
            "45",
            "--retry",
            str(min(retries, 2)),
            "--retry-delay",
            "5",
            "--retry-max-time",
            "180",
            "--retry-all-errors",
            "--output",
            str(chunk_path),
            "--silent",
            "--show-error",
            url,
        ]
        try:
            completed = run_curl(command, config, timeout=240)
            actual_length = chunk_path.stat().st_size if chunk_path.is_file() else 0
            if completed.returncode != 0 or actual_length != length:
                detail = completed.stderr.strip() or f"curl exit code {completed.returncode}"
                raise RuntimeError(
                    f"下载分段 {offset}-{end} 失败：期望 {length} 字节，实际 {actual_length}；{detail}"
                )
            with chunk_path.open("rb") as source, partial.open("ab") as output:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
            print(f"[download] {partial.stat().st_size}/{total_size} bytes", flush=True)
        finally:
            chunk_path.unlink(missing_ok=True)
    partial.replace(target)


def normalized_zip_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    path = PurePosixPath(normalized)
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError(f"ZIP 包含不安全路径：{name}")
    return path


def validate_archive(
    archive_path: Path,
    run_name: str,
    development_only: bool = False,
    final_release: bool = False,
) -> list[str]:
    with zipfile.ZipFile(archive_path) as archive:
        corrupt_entry = archive.testzip()
        if corrupt_entry is not None:
            raise zipfile.BadZipFile(f"ZIP CRC 校验失败：{corrupt_entry}")
        files: list[str] = []
        seen_files: set[str] = set()
        for item in archive.infolist():
            path = normalized_zip_path(item.filename)
            unix_mode = item.external_attr >> 16
            if unix_mode and (unix_mode & 0o170000) == 0o120000:
                raise ValueError(f"ZIP 不允许符号链接：{item.filename}")
            if not item.is_dir():
                normalized_name = path.as_posix()
                if normalized_name in seen_files:
                    raise ValueError(f"ZIP 包含重复文件条目：{normalized_name}")
                seen_files.add(normalized_name)
                files.append(normalized_name)

    expected_prefix = f"{run_name}/"
    if not files or any(not item.startswith(expected_prefix) for item in files):
        roots = sorted({PurePosixPath(item).parts[0] for item in files})
        raise ValueError(f"ZIP 根目录不符合预期：期望 {run_name}，实际 {roots}")
    required_files = required_run_files(development_only, final_release)
    missing = [item for item in required_files if f"{run_name}/{item}" not in files]
    if missing:
        raise FileNotFoundError(f"ZIP 缺少必需训练证据：{missing}")
    with zipfile.ZipFile(archive_path) as archive:
        config_name = f"{run_name}/requested_config.json"
        try:
            requested_config = json.loads(archive.read(config_name).decode("utf-8"))
        except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"ZIP 内 requested_config.json 无法解析：{config_name}") from error
    if requested_config.get("run_name") != run_name:
        raise ValueError("ZIP 内 requested_config.json 的 run_name 不匹配")
    if bool(requested_config.get("development_only", False)) != development_only:
        raise ValueError("ZIP 内 development_only 与归档参数不一致")
    return files


def extract_verified_archive(
    archive_path: Path,
    destination: Path,
    run_name: str,
    development_only: bool = False,
    final_release: bool = False,
) -> None:
    destination_parent = destination.parent.resolve()
    if destination.exists():
        raise FileExistsError(f"拒绝覆盖已有 run：{destination}")
    if destination_parent != DEFAULT_RUNS_ROOT.resolve() and not destination_parent.is_dir():
        raise FileNotFoundError(f"目标 runs 根目录不存在：{destination_parent}")

    # Keep the extraction root short on Windows: the Kaggle raw-output path
    # plus the long run name can otherwise exceed the legacy path limit.
    with tempfile.TemporaryDirectory(prefix="l1_extract_", dir=destination_parent) as temp_dir:
        temp_root = Path(temp_dir)
        with zipfile.ZipFile(archive_path) as archive:
            for item in archive.infolist():
                path = normalized_zip_path(item.filename)
                target = (temp_root / Path(*path.parts)).resolve()
                if not target.is_relative_to(temp_root.resolve()):
                    raise ValueError(f"ZIP 解压路径越界：{item.filename}")
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(item) as source, target.open("wb") as output:
                        shutil.copyfileobj(source, output, length=CHUNK_SIZE)

        extracted_run = temp_root / run_name
        if not extracted_run.is_dir():
            raise FileNotFoundError(f"ZIP 解压后缺少 run 根目录：{extracted_run}")
        staging = destination_parent / f".{run_name}.collecting-{uuid.uuid4().hex}"
        try:
            shutil.copytree(extracted_run, staging)
            missing = [
                relative
                for relative in required_run_files(development_only, final_release)
                if not (staging / relative).is_file()
            ]
            if missing:
                raise FileNotFoundError(f"临时归档目录缺少必需训练证据：{missing}")
            if destination.exists():
                raise FileExistsError(f"归档期间目标已出现，拒绝覆盖：{destination}")
            staging.replace(destination)
        finally:
            if staging.exists():
                shutil.rmtree(staging)


def write_collection_manifest(
    destination: Path,
    kernel: str,
    archive_path: Path,
    files: list[str],
    development_only: bool = False,
    final_release: bool = False,
) -> None:
    manifest = {
        "kernel": kernel,
        "archive_path": str(archive_path),
        "archive_sha256": sha256_file(archive_path),
        "archive_file_count": len(files),
        "development_only": development_only,
        "final_release": final_release,
        "required_files": list(required_run_files(development_only, final_release)),
    }
    (destination / "kaggle_collection.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    api = KaggleApi()
    api.authenticate()
    status = kernel_status(api, args.kernel)
    print(json.dumps({"status": status}, ensure_ascii=False), flush=True)

    if args.check_only or not (args.download or args.install):
        return 0
    if status != "COMPLETE":
        raise RuntimeError(f"Kaggle 任务尚未完成，当前状态为 {status}")

    archive_name = f"{args.run_name}_results.zip"
    raw_root = (args.raw_output_root / args.kernel.replace("/", "__")).resolve()
    archive_path = raw_root / archive_name
    username, key = kaggle_credentials(api)
    archive_url = direct_archive_download_url(args.kernel, archive_name)
    download_resumable(archive_url, archive_path, args.retries, username, key)
    files = validate_archive(
        archive_path,
        args.run_name,
        args.development_only,
        args.final_release,
    )

    print(
        json.dumps(
            {
                "archive": str(archive_path),
                "archive_sha256": sha256_file(archive_path),
                "archive_file_count": len(files),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    if args.install:
        destination = (args.runs_root / args.run_name).resolve()
        extract_verified_archive(
            archive_path,
            destination,
            args.run_name,
            args.development_only,
            args.final_release,
        )
        write_collection_manifest(
            destination,
            args.kernel,
            archive_path,
            files,
            args.development_only,
            args.final_release,
        )
        print(f"[installed] {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
