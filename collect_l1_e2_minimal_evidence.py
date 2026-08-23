#!/usr/bin/env python3
"""Build a verifiable minimal E2 evidence ZIP from a large Kaggle result ZIP.

Kaggle exposes the whole notebook output as one ZIP.  E2 notebooks also retain
large evaluation workspaces, so downloading that complete ZIP is wasteful.  This
tool reads the remote ZIP central directory with authenticated HTTP ranges,
copies only the required E2 members into a new local ZIP, and records the source
member metadata.  It never alters source datasets or Kaggle outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import tempfile
import uuid
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from kaggle.api.kaggle_api_extended import KaggleApi

from collect_l1_kaggle_run import (
    curl_config,
    direct_archive_download_url,
    kaggle_credentials,
    kernel_status,
    remote_archive_size,
    run_curl,
)
from verify_l1_e2_mosaic_pair import REQUIRED_FILES


CHUNK_SIZE = 1024 * 1024
DEFAULT_RANGE_CHUNK_BYTES = 4 * 1024 * 1024
CENTRAL_DIRECTORY_HEADER = struct.Struct("<4s6H3L5H2L")
LOCAL_FILE_HEADER = struct.Struct("<4s5H3L2H")
END_OF_CENTRAL_DIRECTORY = struct.Struct("<4s4H2LH")
ZIP64_END_OF_CENTRAL_DIRECTORY = struct.Struct("<4sQ2H2L4Q")
ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR = struct.Struct("<4sLQL")


@dataclass(frozen=True)
class RemoteZipMember:
    name: str
    flags: int
    compression: int
    crc32: int
    compressed_size: int
    uncompressed_size: int
    local_header_offset: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Kaggle E2 大型结果 ZIP 中仅提取验证所需成员，生成新的最小证据 ZIP。"
    )
    parser.add_argument("--kernel", required=True, help="Kaggle kernel，格式为 owner/slug。")
    parser.add_argument("--run-name", required=True, help="原始结果 ZIP 内的运行目录名。")
    parser.add_argument("--output-archive", type=Path, required=True, help="新建的最小证据 ZIP 路径。")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        help="记录远端 ZIP、成员元数据和本地哈希的 JSON 路径；默认与输出 ZIP 同名。",
    )
    parser.add_argument("--retries", type=int, default=5, help="每段下载的最大重试次数。")
    parser.add_argument(
        "--range-chunk-bytes",
        type=int,
        default=DEFAULT_RANGE_CHUNK_BYTES,
        help="远端成员下载分段大小，默认 4 MiB。",
    )
    parser.add_argument("--dry-run", action="store_true", help="只检查本地输出路径，不访问 Kaggle。")
    return parser.parse_args(argv)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_zip_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    if not path.parts or path.is_absolute() or ".." in path.parts or ":" in path.parts[0]:
        raise ValueError(f"远端 ZIP 包含不安全路径：{name}")
    return path


class RangeFetcher:
    def __init__(
        self,
        url: str,
        curl_config_text: str,
        work_dir: Path,
        retries: int,
    ) -> None:
        self.url = url
        self.curl_config_text = curl_config_text
        self.work_dir = work_dir
        self.retries = retries
        self.curl = shutil.which("curl.exe") or shutil.which("curl")
        if self.curl is None:
            raise FileNotFoundError("未找到 curl，无法读取 Kaggle ZIP 分段")

    def fetch(self, start: int, end: int) -> bytes:
        if start < 0 or end < start:
            raise ValueError(f"无效远端字节范围：{start}-{end}")
        expected_size = end - start + 1
        destination = self.work_dir / f"range-{start}-{end}-{uuid.uuid4().hex}.bin"
        try:
            for attempt in range(1, self.retries + 1):
                temporary = destination.with_name(f".{destination.name}.part")
                command = [
                    self.curl,
                    "--config",
                    "-",
                    "--fail",
                    "--location",
                    "--http1.1",
                    "--range",
                    f"{start}-{end}",
                    "--connect-timeout",
                    "15",
                    "--max-time",
                    "180",
                    "--speed-limit",
                    "1024",
                    "--speed-time",
                    "45",
                    "--retry",
                    "2",
                    "--retry-delay",
                    "5",
                    "--retry-max-time",
                    "180",
                    "--retry-all-errors",
                    "--output",
                    str(temporary),
                    "--silent",
                    "--show-error",
                    self.url,
                ]
                try:
                    completed = run_curl(command, self.curl_config_text, timeout=240)
                    actual_size = temporary.stat().st_size if temporary.is_file() else 0
                    if completed.returncode == 0 and actual_size == expected_size:
                        temporary.replace(destination)
                        return destination.read_bytes()
                    detail = completed.stderr.strip() or f"curl exit code {completed.returncode}"
                finally:
                    temporary.unlink(missing_ok=True)
                if attempt == self.retries:
                    raise RuntimeError(
                        f"Kaggle ZIP 分段 {start}-{end} 下载失败：期望 {expected_size} 字节；{detail}"
                    )
            raise AssertionError("unreachable")
        finally:
            destination.unlink(missing_ok=True)


def fetch_span(fetcher: RangeFetcher, start: int, size: int, chunk_bytes: int) -> bytes:
    if size < 0:
        raise ValueError("ZIP 成员长度不能为负数")
    parts: list[bytes] = []
    offset = start
    remaining = size
    while remaining:
        length = min(remaining, chunk_bytes)
        parts.append(fetcher.fetch(offset, offset + length - 1))
        offset += length
        remaining -= length
    return b"".join(parts)


def read_zip64_extra(
    extra: bytes,
    *,
    needs_uncompressed_size: bool,
    needs_compressed_size: bool,
    needs_local_header_offset: bool,
) -> tuple[int | None, int | None, int | None]:
    position = 0
    while position + 4 <= len(extra):
        header_id, length = struct.unpack_from("<HH", extra, position)
        position += 4
        payload = extra[position : position + length]
        position += length
        if header_id != 0x0001:
            continue
        cursor = 0

        def take_uint64(required: bool) -> int | None:
            nonlocal cursor
            if not required:
                return None
            if cursor + 8 > len(payload):
                raise ValueError("ZIP64 扩展字段不完整")
            value = struct.unpack_from("<Q", payload, cursor)[0]
            cursor += 8
            return value

        return (
            take_uint64(needs_uncompressed_size),
            take_uint64(needs_compressed_size),
            take_uint64(needs_local_header_offset),
        )
    if any((needs_uncompressed_size, needs_compressed_size, needs_local_header_offset)):
        raise ValueError("ZIP64 成员缺少扩展字段")
    return None, None, None


def central_directory_location(fetcher: RangeFetcher, total_size: int, chunk_bytes: int) -> tuple[int, int]:
    tail_size = min(total_size, max(1024 * 1024, chunk_bytes))
    tail_start = total_size - tail_size
    tail = fetcher.fetch(tail_start, total_size - 1)
    signature = b"PK\x05\x06"
    position = tail.rfind(signature)
    while position >= 0:
        if position + END_OF_CENTRAL_DIRECTORY.size <= len(tail):
            values = END_OF_CENTRAL_DIRECTORY.unpack_from(tail, position)
            _, disk_number, directory_disk, _, total_entries, directory_size, directory_offset, comment_length = values
            if position + END_OF_CENTRAL_DIRECTORY.size + comment_length == len(tail):
                if disk_number != 0 or directory_disk != 0:
                    raise ValueError("不支持多磁盘 ZIP")
                if total_entries != 0xFFFF and directory_size != 0xFFFFFFFF and directory_offset != 0xFFFFFFFF:
                    return directory_offset, directory_size
                locator_position = position - ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR.size
                if locator_position < 0:
                    raise ValueError("ZIP64 归档缺少目录定位器")
                locator = ZIP64_END_OF_CENTRAL_DIRECTORY_LOCATOR.unpack_from(tail, locator_position)
                if locator[0] != b"PK\x06\x07" or locator[1] != 0 or locator[3] != 1:
                    raise ValueError("ZIP64 目录定位器无效")
                zip64_offset = locator[2]
                raw = fetcher.fetch(zip64_offset, zip64_offset + ZIP64_END_OF_CENTRAL_DIRECTORY.size - 1)
                zip64 = ZIP64_END_OF_CENTRAL_DIRECTORY.unpack(raw)
                if zip64[0] != b"PK\x06\x06" or zip64[4] != 0 or zip64[5] != 0:
                    raise ValueError("ZIP64 目录记录无效或跨磁盘")
                return int(zip64[9]), int(zip64[8])
        position = tail.rfind(signature, 0, position)
    raise ValueError("未在远端 ZIP 末尾找到中央目录")


def read_central_directory(fetcher: RangeFetcher, total_size: int, chunk_bytes: int) -> dict[str, RemoteZipMember]:
    directory_offset, directory_size = central_directory_location(fetcher, total_size, chunk_bytes)
    if directory_offset < 0 or directory_size <= 0 or directory_offset + directory_size > total_size:
        raise ValueError("远端 ZIP 中央目录范围无效")
    raw = fetch_span(fetcher, directory_offset, directory_size, chunk_bytes)
    members: dict[str, RemoteZipMember] = {}
    position = 0
    while position < len(raw):
        if position + CENTRAL_DIRECTORY_HEADER.size > len(raw):
            raise ValueError("远端 ZIP 中央目录记录被截断")
        fields = CENTRAL_DIRECTORY_HEADER.unpack_from(raw, position)
        if fields[0] != b"PK\x01\x02":
            raise ValueError("远端 ZIP 中央目录签名无效")
        (
            _,
            _,
            _,
            flags,
            compression,
            _,
            _,
            crc32,
            compressed_size,
            uncompressed_size,
            name_length,
            extra_length,
            comment_length,
            disk_start,
            _,
            _,
            local_header_offset,
        ) = fields
        start = position + CENTRAL_DIRECTORY_HEADER.size
        end = start + name_length + extra_length + comment_length
        if end > len(raw):
            raise ValueError("远端 ZIP 中央目录文件名或扩展字段被截断")
        raw_name = raw[start : start + name_length]
        extra = raw[start + name_length : start + name_length + extra_length]
        encoding = "utf-8" if flags & 0x800 else "cp437"
        name = normalized_zip_path(raw_name.decode(encoding)).as_posix()
        zip64_uncompressed, zip64_compressed, zip64_offset = read_zip64_extra(
            extra,
            needs_uncompressed_size=uncompressed_size == 0xFFFFFFFF,
            needs_compressed_size=compressed_size == 0xFFFFFFFF,
            needs_local_header_offset=local_header_offset == 0xFFFFFFFF,
        )
        if zip64_uncompressed is not None:
            uncompressed_size = zip64_uncompressed
        if zip64_compressed is not None:
            compressed_size = zip64_compressed
        if zip64_offset is not None:
            local_header_offset = zip64_offset
        if disk_start not in (0, 0xFFFF):
            raise ValueError("远端 ZIP 成员位于不支持的磁盘")
        if name in members:
            raise ValueError(f"远端 ZIP 含有重名成员：{name}")
        members[name] = RemoteZipMember(
            name=name,
            flags=flags,
            compression=compression,
            crc32=crc32,
            compressed_size=int(compressed_size),
            uncompressed_size=int(uncompressed_size),
            local_header_offset=int(local_header_offset),
        )
        position = end
    return members


def local_member_data_offset(fetcher: RangeFetcher, member: RemoteZipMember) -> int:
    raw = fetcher.fetch(member.local_header_offset, member.local_header_offset + LOCAL_FILE_HEADER.size - 1)
    fields = LOCAL_FILE_HEADER.unpack(raw)
    if fields[0] != b"PK\x03\x04":
        raise ValueError(f"远端 ZIP 本地文件头无效：{member.name}")
    name_length, extra_length = fields[-2:]
    return member.local_header_offset + LOCAL_FILE_HEADER.size + name_length + extra_length


def copy_member(
    fetcher: RangeFetcher,
    member: RemoteZipMember,
    output: zipfile.ZipFile,
    chunk_bytes: int,
) -> dict[str, Any]:
    if member.flags & 0x1:
        raise ValueError(f"不支持加密 ZIP 成员：{member.name}")
    if member.compression not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
        raise ValueError(f"不支持的 ZIP 压缩方法 {member.compression}：{member.name}")
    data_offset = local_member_data_offset(fetcher, member)
    zip_info = zipfile.ZipInfo(member.name)
    zip_info.compress_type = zipfile.ZIP_STORED
    digest = hashlib.sha256()
    crc32 = 0
    uncompressed_size = 0
    offset = data_offset
    remaining = member.compressed_size
    decompressor = zlib.decompressobj(-15) if member.compression == zipfile.ZIP_DEFLATED else None
    with output.open(zip_info, mode="w", force_zip64=True) as destination:
        while remaining:
            length = min(remaining, chunk_bytes)
            compressed = fetcher.fetch(offset, offset + length - 1)
            offset += length
            remaining -= length
            raw = decompressor.decompress(compressed) if decompressor is not None else compressed
            if raw:
                destination.write(raw)
                digest.update(raw)
                crc32 = zlib.crc32(raw, crc32)
                uncompressed_size += len(raw)
        if decompressor is not None:
            final = decompressor.flush()
            if not decompressor.eof:
                raise ValueError(f"ZIP Deflate 流未正常结束：{member.name}")
            if final:
                destination.write(final)
                digest.update(final)
                crc32 = zlib.crc32(final, crc32)
                uncompressed_size += len(final)
    if uncompressed_size != member.uncompressed_size:
        raise ValueError(
            f"远端 ZIP 成员解压长度不符：{member.name}，期望 {member.uncompressed_size}，实际 {uncompressed_size}"
        )
    if (crc32 & 0xFFFFFFFF) != member.crc32:
        raise ValueError(f"远端 ZIP 成员 CRC 不符：{member.name}")
    return {
        "name": member.name,
        "source_compression": member.compression,
        "source_crc32": f"{member.crc32:08x}",
        "source_compressed_size": member.compressed_size,
        "source_uncompressed_size": member.uncompressed_size,
        "source_local_header_offset": member.local_header_offset,
        "sha256": digest.hexdigest(),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.retries < 1:
        raise ValueError("--retries 必须至少为 1")
    if not 64 * 1024 <= args.range_chunk_bytes <= 16 * 1024 * 1024:
        raise ValueError("--range-chunk-bytes 必须位于 64 KiB 到 16 MiB 之间")
    output_archive = args.output_archive.resolve()
    source_manifest = (args.source_manifest or output_archive.with_suffix(".source_manifest.json")).resolve()
    if output_archive == source_manifest:
        raise ValueError("输出 ZIP 与 source manifest 不能使用同一路径")
    if output_archive.exists() or source_manifest.exists():
        raise FileExistsError("拒绝覆盖已有最小证据 ZIP 或 source manifest")
    required_members = tuple(f"{args.run_name}/{relative}" for relative in REQUIRED_FILES)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "kernel": args.kernel,
                    "run_name": args.run_name,
                    "output_archive": str(output_archive),
                    "source_manifest": str(source_manifest),
                    "required_member_count": len(required_members),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    output_archive.parent.mkdir(parents=True, exist_ok=True)
    source_manifest.parent.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    status = kernel_status(api, args.kernel)
    if status != "COMPLETE":
        raise RuntimeError(f"Kaggle 任务尚未完成：{args.kernel}，当前状态 {status}")
    username, key = kaggle_credentials(api)
    archive_name = f"{args.run_name}_results.zip"
    url = direct_archive_download_url(args.kernel, archive_name)
    config = curl_config(username, key)
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if curl is None:
        raise FileNotFoundError("未找到 curl，无法读取 Kaggle ZIP")
    total_size = remote_archive_size(curl, url, config, output_archive.parent)
    with tempfile.TemporaryDirectory(prefix="l1_e2_minimal_evidence_", dir=output_archive.parent) as temporary:
        temporary_root = Path(temporary)
        fetcher = RangeFetcher(url, config, temporary_root, args.retries)
        members = read_central_directory(fetcher, total_size, args.range_chunk_bytes)
        missing = [name for name in required_members if name not in members]
        if missing:
            raise FileNotFoundError(f"远端结果 ZIP 缺少 E2 必要证据：{missing}")
        temporary_archive = temporary_root / output_archive.name
        selected_metadata: list[dict[str, Any]] = []
        with zipfile.ZipFile(temporary_archive, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
            for name in required_members:
                selected_metadata.append(copy_member(fetcher, members[name], output, args.range_chunk_bytes))
        with zipfile.ZipFile(temporary_archive) as output:
            corrupt = output.testzip()
            if corrupt is not None:
                raise zipfile.BadZipFile(f"新建最小证据 ZIP CRC 失败：{corrupt}")
            actual_names = {info.filename for info in output.infolist() if not info.is_dir()}
        if actual_names != set(required_members):
            raise ValueError("新建最小证据 ZIP 的成员清单不等于 E2 必要证据清单")
        temporary_archive.replace(output_archive)

    manifest = {
        "schema_version": 1,
        "purpose": "minimal evidence archive for L1 E2 paired Mosaic verification",
        "source": {
            "kernel": args.kernel,
            "archive_name": archive_name,
            "download_endpoint": direct_archive_download_url(args.kernel, archive_name),
            "total_bytes": total_size,
            "full_archive_sha256": None,
            "full_archive_sha256_note": "Not available because only required ZIP members were range-read.",
        },
        "run_name": args.run_name,
        "required_members": list(required_members),
        "selected_members": selected_metadata,
        "output_archive": str(output_archive),
        "output_archive_sha256": sha256_file(output_archive),
        "output_archive_bytes": output_archive.stat().st_size,
    }
    write_json(source_manifest, manifest)
    print(
        json.dumps(
            {
                "output_archive": str(output_archive),
                "source_manifest": str(source_manifest),
                "source_total_bytes": total_size,
                "output_archive_bytes": output_archive.stat().st_size,
                "selected_member_count": len(selected_metadata),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
