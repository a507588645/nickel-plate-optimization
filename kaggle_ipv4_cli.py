#!/usr/bin/env python3
"""Run the installed Kaggle CLI while preferring IPv4 for upload reliability."""

from __future__ import annotations

import socket

from kaggle.cli import main
import urllib3.util.connection as urllib3_connection
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
from kaggle.api.kaggle_api_extended import ResumableFileUpload


_ORIGINAL_CREATE_CONNECTION = urllib3_connection.create_connection


def ipv4_only_connection(
    address: tuple[str, int],
    timeout: object = urllib3_connection._DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
    socket_options: list[tuple[object, ...]] | None = None,
) -> socket.socket:
    """Resolve Kaggle upload hosts over IPv4 before falling back to urllib3."""
    host, port = address
    last_error: OSError | None = None
    for family, socktype, protocol, _, sockaddr in socket.getaddrinfo(
        host, port, socket.AF_INET, socket.SOCK_STREAM
    ):
        sock: socket.socket | None = None
        try:
            sock = socket.socket(family, socktype, protocol)
            if timeout is not urllib3_connection._DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            if socket_options:
                for option in socket_options:
                    sock.setsockopt(*option)
            sock.connect(sockaddr)
            return sock
        except OSError as error:
            last_error = error
            if sock is not None:
                sock.close()
    if last_error is not None:
        raise last_error
    return _ORIGINAL_CREATE_CONNECTION(address, timeout, source_address, socket_options)


def run() -> None:
    urllib3_connection.create_connection = ipv4_only_connection
    # Kaggle 1.7.4.5 serializes a prior resumable upload as ``type`` while
    # the installed SDK constructor accepts no keyword fields. Rehydrate it
    # through from_dict so interrupted uploads can resume instead of hanging.
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
    main()


if __name__ == "__main__":
    run()
