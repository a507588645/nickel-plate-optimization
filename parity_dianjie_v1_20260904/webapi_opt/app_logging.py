from __future__ import annotations

import contextvars
import json
import logging
from datetime import datetime, timezone
from typing import Any


_request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_id", default=None)
_request_method_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_method", default=None)
_request_path_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar("request_path", default=None)
_STANDARD_RECORD_KEYS = set(logging.makeLogRecord({}).__dict__.keys())


def set_request_context(*, request_id: str, method: str, path: str) -> None:
    _request_id_ctx.set(request_id)
    _request_method_ctx.set(method)
    _request_path_ctx.set(path)


def clear_request_context() -> None:
    _request_id_ctx.set(None)
    _request_method_ctx.set(None)
    _request_path_ctx.set(None)


class RequestContextFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(record, "request_id", None) or _request_id_ctx.get()
        record.request_method = getattr(record, "request_method", None) or _request_method_ctx.get()
        record.request_path = getattr(record, "request_path", None) or _request_path_ctx.get()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _STANDARD_RECORD_KEYS or key.startswith("_"):
                continue
            if value is None:
                continue
            try:
                json.dumps(value, ensure_ascii=False)
                payload[key] = value
            except TypeError:
                payload[key] = str(value)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(settings) -> None:
    level_name = str(settings.server.log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    handler = logging.StreamHandler()
    handler.addFilter(RequestContextFilter())
    if settings.logging.json_logs:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s [%(request_id)s] %(message)s"
        ))

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    for logger_name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True
