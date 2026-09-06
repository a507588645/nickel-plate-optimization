from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from app_metrics import MODEL_LOAD_SECONDS, MODELS_READY
from inference import (
    available_detectors,
    load_dianji_detector,
    load_dianjie_detector,
    load_resnet,
    load_slf_resnet,
    load_yolov5seg,
    unload_models,
    use_gpu,
)
from inference.models import MODELS_CONFIG, load_dianji_grade_model


def _models_summary() -> dict[str, Any]:
    return {
        "detectors": MODELS_CONFIG.get("detectors", {}),
        "classifiers": MODELS_CONFIG.get("classifiers", {}),
        "grade_models": MODELS_CONFIG.get("grade_models", {}),
        "segments": MODELS_CONFIG.get("segments", {}),
        "available_detectors": available_detectors(),
    }


@dataclass
class RuntimeState:
    started_at: float = field(default_factory=time.time)
    ready: bool = False
    status: str = "starting"
    last_error: str | None = None
    models_loaded: bool = False
    device: str = "cuda" if use_gpu else "cpu"
    models: dict[str, Any] = field(default_factory=_models_summary)


runtime_state = RuntimeState()


def preload_models(logger: logging.Logger) -> None:
    if runtime_state.models_loaded:
        return

    runtime_state.status = "loading"
    runtime_state.last_error = None
    load_steps = [
        ("dianjie_detector", load_dianjie_detector),
        ("dianji_grade_lgbm", load_dianji_grade_model),
        ("dianjie_resnet", load_resnet),
        ("dianji_resnet", load_slf_resnet),
        ("dianji_detector", load_dianji_detector),
        ("nietiao_segment", load_yolov5seg),
    ]

    try:
        for model_name, loader in load_steps:
            start_ts = time.perf_counter()
            loader()
            MODEL_LOAD_SECONDS.labels(model_name=model_name).observe(time.perf_counter() - start_ts)
            logger.info("model_loaded", extra={"event": "model_loaded", "model_name": model_name})
        runtime_state.ready = True
        runtime_state.status = "ready"
        runtime_state.models_loaded = True
        MODELS_READY.set(1)
    except Exception as exc:
        runtime_state.ready = False
        runtime_state.status = "failed"
        runtime_state.last_error = repr(exc)
        MODELS_READY.set(0)
        logger.exception("model_preload_failed", extra={"event": "model_preload_failed"})
        raise


def shutdown_models(logger: logging.Logger) -> None:
    if not runtime_state.models_loaded:
        runtime_state.ready = False
        runtime_state.status = "stopped"
        MODELS_READY.set(0)
        return

    unload_models()
    runtime_state.ready = False
    runtime_state.status = "stopped"
    runtime_state.models_loaded = False
    MODELS_READY.set(0)
    logger.info("models_unloaded", extra={"event": "models_unloaded"})
