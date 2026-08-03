"""Resolve machine-local paths from a Git-tracked logical dataset catalog.

The catalog is portable; each machine keeps its absolute path mapping in the
ignored ``config/workspace.local.json`` file.  Scripts may still provide an
explicit CLI path, which always takes precedence over this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_LOCAL_CONFIG = ROOT / "config" / "workspace.local.json"
CONFIG_ENV_VAR = "NICKEL_WORKSPACE_CONFIG"


def load_local_paths(config_path: Path | None = None) -> dict[str, Path]:
    """Return logical path mappings; an absent local config is valid."""
    path = config_path or Path(os.environ.get(CONFIG_ENV_VAR, DEFAULT_LOCAL_CONFIG))
    if not path.is_file():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"workspace config is not valid JSON: {path}") from error

    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported workspace config schema: {path}")
    raw_paths = payload.get("paths")
    if not isinstance(raw_paths, dict):
        raise ValueError(f"workspace config must contain an object named paths: {path}")

    resolved: dict[str, Path] = {}
    for key, value in raw_paths.items():
        if not isinstance(key, str) or not isinstance(value, str) or not value.strip():
            raise ValueError(f"workspace path mapping must be non-empty strings: {path}")
        resolved[key] = Path(value).expanduser()
    return resolved


def resolve_path(logical_id: str, fallback: Path) -> Path:
    """Use a local mapping when present; otherwise preserve the existing default."""
    return load_local_paths().get(logical_id, fallback)
