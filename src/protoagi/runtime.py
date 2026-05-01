from __future__ import annotations

import json
from pathlib import Path
import subprocess
import time
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from .config import LlamaServerProfile


def ensure_runtime(profile: LlamaServerProfile) -> None:
    if not profile.server_exe.exists():
        raise FileNotFoundError(f"llama-server.exe not found: {profile.server_exe}")
    if not profile.model_path.exists():
        raise FileNotFoundError(f"model not found: {profile.model_path}")


def server_ready(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url.rstrip('/')}/models", timeout=2) as response:
            return response.status == 200
    except URLError:
        return False
    except TimeoutError:
        return False


def wait_for_server(base_url: str, *, timeout_seconds: int = 180) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if server_ready(base_url):
            return True
        time.sleep(1.0)
    return False


def run_server_foreground(profile: LlamaServerProfile) -> int:
    ensure_runtime(profile)
    return subprocess.call(profile.server_command(), cwd=profile.llama_dir)


def status_report(profile: LlamaServerProfile, *, base_url: str) -> dict[str, Any]:
    return {
        "model_path": str(profile.model_path),
        "model_exists": profile.model_path.exists(),
        "model_bytes": profile.model_path.stat().st_size if profile.model_path.exists() else None,
        "llama_server": str(profile.server_exe),
        "llama_server_exists": profile.server_exe.exists(),
        "base_url": base_url,
        "server_ready": server_ready(base_url),
        "server_command": profile.server_command(),
    }


def save_status(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")

