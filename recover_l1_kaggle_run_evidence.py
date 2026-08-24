#!/usr/bin/env python3
"""Recover selected evidence from a completed, cancelled, or errored Kaggle L1 job.

The script only reads Kaggle working-output files and writes a new local recovery
directory. It never reads the frozen release-validation set or changes a source
dataset, a submitted notebook, or an existing local run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.kernels.services.kernels_api_service import ApiListKernelSessionOutputRequest

from collect_l1_kaggle_run import (
    bounded_sdk_session,
    configure_sdk_timeout,
    download_resumable,
    kaggle_credentials,
    kernel_status,
    split_kernel,
)


CHUNK_SIZE = 1024 * 1024
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover selected Kaggle L1 job evidence into a new local directory.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--kernel", required=True, help="Kaggle kernel in owner/slug form.")
    parser.add_argument("--run-name", required=True, help="Expected run directory under Kaggle working/runs.")
    parser.add_argument("--output-root", type=Path, required=True, help="New local recovery directory.")
    parser.add_argument("--max-pages", type=int, default=20, help="Maximum Kaggle output pages to inspect.")
    parser.add_argument("--include-last", action="store_true", help="Also recover weights/last.pt when a full local evidence copy is needed.")
    parser.add_argument("--dry-run", action="store_true", help="Only verify remote availability; do not write files.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def required_remote_files(run_name: str, include_last: bool) -> tuple[str, ...]:
    prefix = f"runs/{run_name}"
    required = [
        f"{prefix}/args.yaml",
        f"{prefix}/results.csv",
        f"{prefix}/weights/best.pt",
    ]
    if include_last:
        required.append(f"{prefix}/weights/last.pt")
    return tuple(required)


def list_required_files(api: KaggleApi, kernel: str, expected: tuple[str, ...], max_pages: int) -> tuple[dict[str, Any], int]:
    if max_pages <= 0:
        raise ValueError("--max-pages must be positive")
    owner, slug = split_kernel(kernel)
    found: dict[str, Any] = {}
    page_token: str | None = None
    pages = 0
    with bounded_sdk_session(api) as client:
        configure_sdk_timeout(client)
        while True:
            request = ApiListKernelSessionOutputRequest()
            request.user_name = owner
            request.kernel_slug = slug
            request.page_size = 500
            if page_token:
                request.page_token = page_token
            response = client.kernels.kernels_api_client.list_kernel_session_output(request)
            pages += 1
            for item in getattr(response, "files", []) or []:
                name = str(getattr(item, "file_name", "") or "")
                if name in expected:
                    found[name] = item
            if len(found) == len(expected):
                return found, pages
            page_token = str(getattr(response, "next_page_token", "") or "")
            if not page_token:
                break
            if pages >= max_pages:
                break
    missing = [name for name in expected if name not in found]
    raise FileNotFoundError(
        f"Kaggle working output lacks required recovery files after {pages} page(s): {missing}"
    )


def local_target(output_root: Path, run_name: str, remote_name: str) -> Path:
    prefix = PurePosixPath("runs") / run_name
    path = PurePosixPath(remote_name)
    try:
        relative = path.relative_to(prefix)
    except ValueError as error:
        raise ValueError(f"Unexpected remote output path: {remote_name}") from error
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Unsafe remote output path: {remote_name}")
    return output_root / run_name / Path(*relative.parts)


def download_file(url: str, target: Path, username: str, key: str) -> dict[str, Any]:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite recovered file: {target}")
    # The normal HTTP stream can be truncated by this network. Reuse the
    # range-verified curl downloader already used for Kaggle result archives.
    download_resumable(url, target, retries=6, username=username, key=key)
    if not target.is_file() or target.stat().st_size <= 0:
        raise ValueError(f"Recovered Kaggle file is missing or empty: {target}")
    return {"bytes": target.stat().st_size, "sha256": sha256_file(target)}


def main() -> int:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Recovery output root already exists: {output_root}")
    api = KaggleApi()
    api.authenticate()
    status = kernel_status(api, args.kernel)
    expected = required_remote_files(args.run_name, args.include_last)
    found, pages = list_required_files(api, args.kernel, expected, args.max_pages)
    availability = {
        name: bool(getattr(found[name], "url", ""))
        for name in expected
    }
    if not all(availability.values()):
        unavailable = [name for name, available in availability.items() if not available]
        raise FileNotFoundError(f"Kaggle output metadata has no download URL for: {unavailable}")
    summary: dict[str, Any] = {
        "schema_version": 1,
        "kernel": args.kernel,
        "kernel_status": status,
        "run_name": args.run_name,
        "development_only": True,
        "frozen_test_set": None,
        "output_pages_scanned": pages,
        "include_last": bool(args.include_last),
        "remote_files": list(expected),
        "downloaded_files": {},
    }
    if args.dry_run:
        print(json.dumps({**summary, "availability": availability}, ensure_ascii=False, indent=2))
        return 0
    username, key = kaggle_credentials(api)
    output_root.mkdir(parents=True)
    try:
        for remote_name in expected:
            target = local_target(output_root, args.run_name, remote_name)
            artifact = download_file(str(getattr(found[remote_name], "url")), target, username, key)
            summary["downloaded_files"][str(target.relative_to(output_root).as_posix())] = artifact
            print(json.dumps({"downloaded": str(target), **artifact}, ensure_ascii=False), flush=True)
        results_path = output_root / args.run_name / "results.csv"
        if not results_path.is_file():
            raise FileNotFoundError("Recovered results.csv is missing after download")
        summary["results_csv_sha256"] = sha256_file(results_path)
        (output_root / "recovery_manifest.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        # Preserve successfully downloaded evidence but leave no false completion manifest.
        raise
    print(json.dumps({"recovery_manifest": str(output_root / "recovery_manifest.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
