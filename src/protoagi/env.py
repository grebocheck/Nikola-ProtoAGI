from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path, *, override: bool = False) -> int:
    """Load simple KEY=VALUE pairs from a .env file without external deps."""
    if not path.exists():
        return 0
    loaded = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or not _valid_key(key):
            continue
        value = _clean_value(value.strip())
        if override or key not in os.environ:
            os.environ[key] = value
            loaded += 1
    return loaded


def load_project_env(project_root: Path, *, override: bool = False) -> int:
    return load_dotenv(project_root / ".env", override=override)


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _valid_key(key: str) -> bool:
    return all(ch.isalnum() or ch == "_" for ch in key)


def _clean_value(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1]
    return value
