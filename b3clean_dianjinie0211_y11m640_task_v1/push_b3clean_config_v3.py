#!/usr/bin/env python3
"""Push the B3-clean config staging dir as a new version of
``lly3323w/dianjinie0211-b3-config-v1`` via the in-process Kaggle API.

The plain ``kaggle datasets version`` CLI hangs on this host (see worklog
2026-09-05 01:55); the working path is: apply the IPv4 patch from
``kaggle_ipv4_cli.py``, set a default socket timeout, then call
``dataset_create_version`` directly.
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

import kaggle_ipv4_cli  # noqa: E402
import urllib3.util.connection as urllib3_connection  # noqa: E402
from kaggle.api.kaggle_api_extended import KaggleApi, ResumableFileUpload  # noqa: E402
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadRequest  # noqa: E402

STAGING = ROOT / "kaggle_config_upload_v1"
VERSION_NOTES = "v3: b3clean split (grade-test capture keys excluded; train 653 / val 178)"


def apply_patches() -> None:
    urllib3_connection.create_connection = kaggle_ipv4_cli.ipv4_only_connection

    def from_dict(other: dict[str, object], context: object) -> ResumableFileUpload:
        request = ApiStartBlobUploadRequest()
        request.from_dict(other["start_blob_upload_request"])
        response_payload = other.get("start_blob_upload_response")
        from kaggle.api.kaggle_api_extended import ApiStartBlobUploadResponse

        restored = ResumableFileUpload(other["path"], request, context)
        restored.timestamp = other.get("timestamp")
        if response_payload is not None:
            response = ApiStartBlobUploadResponse()
            response.from_dict(response_payload)
            restored.start_blob_upload_response = response
            restored.upload_complete = bool(other.get("upload_complete") or False)
        return restored

    ResumableFileUpload.from_dict = staticmethod(from_dict)


def main() -> int:
    if not (STAGING / "dataset-metadata.json").is_file():
        raise FileNotFoundError(f"缺少 dataset-metadata.json：{STAGING}")
    socket.setdefaulttimeout(60)
    apply_patches()
    api = KaggleApi()
    api.authenticate()
    print(f"[push] dataset_create_version {STAGING} ...", flush=True)
    result = api.dataset_create_version(str(STAGING), VERSION_NOTES, dir_mode="skip")
    print(f"[push] result: {result}", flush=True)
    status = api.dataset_status("lly3323w/dianjinie0211-b3-config-v1")
    print(f"[push] dataset status: {status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
