from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app_paths import project_path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ServerSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    host: str = "127.0.0.1"
    port: int = 5005
    workers: int = 1
    reload: bool = False
    log_level: str = "INFO"
    docs_enabled: bool = True


class LoggingSettings(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    json_logs: bool = Field(default=True, alias="json")
    request_id_header: str = "X-Request-ID"


class CorsSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    allow_origins: list[str] = Field(default_factory=lambda: [
        "tauri://localhost",
        "http://tauri.localhost",
        "http://localhost",
        "http://127.0.0.1",
    ])
    allow_methods: list[str] = Field(default_factory=lambda: ["*"])
    allow_headers: list[str] = Field(default_factory=lambda: ["*"])


class ManagementSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    health_path: str = "/health"
    config_path: str = "/config"
    metrics_path: str = "/metrics"


class RuntimeSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "desktop"
    preload_models: bool = True
    graceful_shutdown_timeout_seconds: int = 15
    allow_multi_worker_with_gpu: bool = False
    debug_artifacts_enabled: bool = True


class NoduleRuleSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ignore_area_below: float = 13000.0
    ignore_aspect_ratio_below: float = 3.0


class GoodBoardRuleSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    a_ratio_max: float = 0.2
    b_ratio_max: float = 0.1
    airhole_ratio_max: float = 0.01
    c_ratio_max: float = 0.03


class HotWashRuleSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    a_ratio_min: float = 0.2
    b_ratio_max: float = 0.05
    c_ratio_max: float = 0.05


class AbDetectRuleSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    value1: int = 100
    value2: int = 200
    value3: int = 500


class DianjieRuleSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    nodule: NoduleRuleSettings = Field(default_factory=NoduleRuleSettings)
    non_water_labels: list[int] = Field(default_factory=lambda: [2, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
    good_board: GoodBoardRuleSettings = Field(default_factory=GoodBoardRuleSettings)
    hot_wash: HotWashRuleSettings = Field(default_factory=HotWashRuleSettings)
    c_ratio_bad_threshold: float = 0.5
    b_ratio_bad_threshold: float = 0.5


class NietiaoSegRuleSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    x_range: list[int] = Field(default_factory=lambda: [588, 1828])
    y_range: list[int] = Field(default_factory=lambda: [360, 1688])
    min_size: int = 500


class BusinessRuleSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    abdetect: AbDetectRuleSettings = Field(default_factory=AbDetectRuleSettings)
    dianjie: DianjieRuleSettings = Field(default_factory=DianjieRuleSettings)
    nietiao_seg: NietiaoSegRuleSettings = Field(default_factory=NietiaoSegRuleSettings)


class AppSettings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    server: ServerSettings = Field(default_factory=ServerSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    cors: CorsSettings = Field(default_factory=CorsSettings)
    management: ManagementSettings = Field(default_factory=ManagementSettings)
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)
    business_rules: BusinessRuleSettings = Field(default_factory=BusinessRuleSettings)

    def public_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        data["runtime_dir"] = os.getenv("JINCHUAN_RUNTIME_DIR") or "AUTO"
        return data


def _apply_env_overrides(settings: AppSettings) -> AppSettings:
    settings.server.host = os.getenv("JINCHUAN_HOST", settings.server.host)
    settings.server.port = int(os.getenv("JINCHUAN_PORT", settings.server.port))
    settings.server.workers = int(os.getenv("JINCHUAN_WORKERS", settings.server.workers))
    settings.server.reload = _env_bool("JINCHUAN_RELOAD", settings.server.reload)
    settings.server.log_level = os.getenv("JINCHUAN_LOG_LEVEL", settings.server.log_level)
    settings.server.docs_enabled = _env_bool("JINCHUAN_DOCS_ENABLED", settings.server.docs_enabled)
    settings.runtime.mode = os.getenv("JINCHUAN_MODE", settings.runtime.mode)
    settings.runtime.preload_models = _env_bool("JINCHUAN_PRELOAD_MODELS", settings.runtime.preload_models)
    settings.runtime.allow_multi_worker_with_gpu = _env_bool(
        "JINCHUAN_ALLOW_GPU_MULTI_WORKER",
        settings.runtime.allow_multi_worker_with_gpu,
    )
    settings.runtime.debug_artifacts_enabled = _env_bool(
        "JINCHUAN_DEBUG_ARTIFACTS_ENABLED",
        settings.runtime.debug_artifacts_enabled,
    )
    return settings


@lru_cache(maxsize=1)
def get_app_settings(config_path: str | Path | None = None) -> AppSettings:
    raw_path = config_path or os.getenv("JINCHUAN_APP_SETTINGS") or project_path("app_settings.yaml")
    config_file = Path(raw_path)
    with config_file.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    settings = AppSettings.model_validate(data)
    return _apply_env_overrides(settings)
