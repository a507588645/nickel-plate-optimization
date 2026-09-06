from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Gauge, Histogram, generate_latest


REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS_TOTAL = Counter(
    "jinchuan_http_requests_total",
    "Total HTTP requests.",
    ["method", "path", "status_code"],
    registry=REGISTRY,
)
HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "jinchuan_http_request_latency_seconds",
    "HTTP request latency in seconds.",
    ["method", "path"],
    registry=REGISTRY,
)
HTTP_REQUEST_EXCEPTIONS_TOTAL = Counter(
    "jinchuan_http_request_exceptions_total",
    "Unhandled HTTP exceptions.",
    ["method", "path"],
    registry=REGISTRY,
)
MODELS_READY = Gauge(
    "jinchuan_models_ready",
    "Whether all configured models are loaded.",
    registry=REGISTRY,
)
MODEL_LOAD_SECONDS = Histogram(
    "jinchuan_model_load_seconds",
    "Model/component preload duration.",
    ["model_name"],
    registry=REGISTRY,
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
