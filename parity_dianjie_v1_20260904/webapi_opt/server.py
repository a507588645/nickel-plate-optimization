from __future__ import annotations

import logging

import uvicorn

from app_logging import configure_logging
from app_settings import get_app_settings


def _gpu_available() -> bool:
    try:
        import torch
    except Exception:
        return False
    return torch.cuda.is_available()


def main() -> None:
    settings = get_app_settings()
    configure_logging(settings)
    logger = logging.getLogger("jinchuan.bootstrap")

    workers = settings.server.workers
    if workers > 1 and _gpu_available() and settings.runtime.allow_multi_worker_with_gpu is False:
        logger.warning(
            "workers_downgraded_to_one",
            extra={
                "event": "workers_downgraded_to_one",
                "mode": settings.runtime.mode,
                "port": settings.server.port,
                "workers": workers,
            },
        )
        workers = 1

    uvicorn.run(
        "fastapi_app:app",
        host=settings.server.host,
        port=settings.server.port,
        reload=settings.server.reload,
        workers=workers,
        log_config=None,
    )


if __name__ == "__main__":
    main()
