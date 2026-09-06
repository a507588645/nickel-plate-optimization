from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def resolve_project_path(path_value: str | os.PathLike[str] | None) -> Path | None:
    if path_value is None:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


@lru_cache(maxsize=1)
def runtime_dir() -> Path:
    env_path = os.getenv("JINCHUAN_RUNTIME_DIR")
    if env_path:
        base = Path(env_path)
    elif os.name == "nt":
        base = Path(os.getenv("LOCALAPPDATA", str(PROJECT_ROOT))) / "JinchuanPythonWebapi"
    else:
        base = Path.home() / ".jinchuan-python-webapi"
    base.mkdir(parents=True, exist_ok=True)
    return base


def debug_file_path(filename: str) -> Path:
    debug_dir = runtime_dir() / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir / filename
