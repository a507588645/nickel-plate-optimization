#!/usr/bin/env python3
"""Safely advance one L1 Kaggle training step after the current run is archived."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi


TERMINAL_FAILURE_STATES = {
    "CANCELLED",
    "CANCEL_REQUESTED",
    "CANCEL_ACKNOWLEDGED",
    "ERROR",
    "FAILED",
    "KILLED",
    "NEW_SCRIPT",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "监控一个 L1 Kaggle run；仅在其 COMPLETE、下载校验和归档全部成功后，"
            "一次性提交下一个已验证的 Kernel。"
        )
    )
    parser.add_argument("--current-kernel", required=True, help="当前 Kernel：owner/slug。")
    parser.add_argument("--current-run-name", required=True, help="当前归档 run 名称。")
    parser.add_argument("--next-kernel-dir", required=True, type=Path, help="下一个 Kernel 本地定义目录。")
    parser.add_argument("--next-kernel", required=True, help="下一个 Kernel：owner/slug。")
    parser.add_argument("--state-dir", required=True, type=Path, help="仅存放监控日志和提交状态的派生目录。")
    parser.add_argument("--interval-seconds", type=int, default=300, help="运行中状态的轮询间隔。")
    parser.add_argument(
        "--kernel-timeout",
        type=int,
        default=43200,
        help="提交下一个 Kernel 时请求的最大秒数；服务端全局上限优先。",
    )
    parser.add_argument("--once", action="store_true", help="仅执行一次状态检查，不等待或提交。")
    return parser.parse_args()


def normalized_status(api: KaggleApi, kernel: str) -> str:
    response = api.kernels_status(kernel)
    return str(getattr(response, "status", "UNKNOWN")).rsplit(".", 1)[-1]


def write_event(path: Path, event: str, **fields: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    with path.open("a", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def load_state(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_checked(command: list[str], event_log: Path, event: str) -> None:
    write_event(event_log, f"{event}_started", command=command)
    result = subprocess.run(command, text=True, capture_output=True)
    write_event(
        event_log,
        f"{event}_finished",
        returncode=result.returncode,
        stdout=result.stdout[-12000:],
        stderr=result.stderr[-12000:],
    )
    if result.returncode != 0:
        raise RuntimeError(f"{event} failed with exit code {result.returncode}")


def validate_next_kernel_dir(path: Path, expected_kernel: str) -> None:
    metadata_path = path / "kernel-metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"下一个 Kernel 缺少 metadata：{metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("id") != expected_kernel:
        raise ValueError(f"下一个 Kernel ID 不匹配：{metadata.get('id')} != {expected_kernel}")
    if metadata.get("enable_gpu") is not True or metadata.get("is_private") is not True:
        raise ValueError("下一个 Kernel 必须保持 private GPU 配置")
    code_file = metadata.get("code_file")
    if not isinstance(code_file, str) or not (path / code_file).is_file():
        raise FileNotFoundError("下一个 Kernel code_file 不存在")


def submit_next(args: argparse.Namespace, event_log: Path, submission_state: Path) -> None:
    prior_state = load_state(submission_state)
    if prior_state is not None:
        raise RuntimeError(
            f"下一个 Kernel 已有提交状态文件：{submission_state}；为避免重复 GPU run，监控器拒绝重提。"
        )

    validate_next_kernel_dir(args.next_kernel_dir, args.next_kernel)
    request = {
        "state": "push_requested",
        "next_kernel": args.next_kernel,
        "next_kernel_dir": str(args.next_kernel_dir.resolve()),
        "requested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    write_state(submission_state, request)
    command = [
        "kaggle",
        "kernels",
        "push",
        "--path",
        str(args.next_kernel_dir.resolve()),
        "--timeout",
        str(args.kernel_timeout),
    ]
    try:
        run_checked(command, event_log, "next_kernel_push")
    except Exception as error:
        request["state"] = "push_failed"
        request["error"] = str(error)
        request["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        write_state(submission_state, request)
        raise
    request["state"] = "submitted"
    request["finished_at_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_state(submission_state, request)


def main() -> int:
    args = parse_args()
    if args.interval_seconds < 30:
        raise ValueError("--interval-seconds 不得小于 30 秒")

    args.next_kernel_dir = args.next_kernel_dir.resolve()
    args.state_dir = args.state_dir.resolve()
    args.state_dir.mkdir(parents=True, exist_ok=True)
    event_log = args.state_dir / "monitor_events.jsonl"
    submission_state = args.state_dir / "next_kernel_submission.json"
    collector = Path(__file__).with_name("collect_l1_kaggle_run.py")
    if not collector.is_file():
        raise FileNotFoundError(f"缺少归档校验器：{collector}")

    api = KaggleApi()
    api.authenticate()
    transient_errors = 0
    write_event(event_log, "monitor_started", current_kernel=args.current_kernel, next_kernel=args.next_kernel)

    while True:
        try:
            status = normalized_status(api, args.current_kernel)
            transient_errors = 0
            write_event(event_log, "status", kernel=args.current_kernel, status=status)
        except Exception as error:
            transient_errors += 1
            write_event(event_log, "status_query_error", error=str(error), consecutive_errors=transient_errors)
            if args.once or transient_errors >= 12:
                raise RuntimeError("Kaggle 状态查询连续失败，未提交下一个 Kernel") from error
            time.sleep(args.interval_seconds)
            continue

        if status == "COMPLETE":
            command = [
                sys.executable,
                str(collector),
                "--kernel",
                args.current_kernel,
                "--run-name",
                args.current_run_name,
                "--install",
            ]
            try:
                run_checked(command, event_log, "collect_current_run")
                submit_next(args, event_log, submission_state)
            except Exception as error:
                write_event(event_log, "monitor_stopped", reason="post_complete_validation_or_submission_failed", error=str(error))
                raise
            write_event(event_log, "monitor_completed", current_kernel=args.current_kernel, next_kernel=args.next_kernel)
            return 0

        if status in TERMINAL_FAILURE_STATES:
            write_event(event_log, "monitor_stopped", reason="current_kernel_failed", status=status)
            raise RuntimeError(f"当前 Kernel 以终止状态结束：{status}；未提交下一个 Kernel")

        if args.once:
            return 0
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
