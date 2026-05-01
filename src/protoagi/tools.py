from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import PROJECT_ROOT, ToolPolicy
from .memory import MemoryStore


TEXT_EXTENSIONS = {
    ".bat",
    ".cmd",
    ".css",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

BLOCKED_SHELL_PATTERNS = [
    "remove-item",
    "rm ",
    "rmdir",
    "del ",
    "erase ",
    "format ",
    "git reset --hard",
    "git checkout --",
    "set-executionpolicy",
    "takeown",
    "icacls",
]


@dataclass(slots=True)
class ToolResult:
    ok: bool
    data: Any | None = None
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "data": self.data, "error": self.error}


@dataclass(slots=True)
class ToolContext:
    root: Path
    memory: MemoryStore
    policy: ToolPolicy


class ToolRegistry:
    def __init__(self, context: ToolContext) -> None:
        self.context = context
        self._handlers: dict[str, Callable[[dict[str, Any]], ToolResult]] = {
            "now": self._now,
            "remember": self._remember,
            "recall": self._recall,
            "list_dir": self._list_dir,
            "read_file": self._read_file,
            "write_file": self._write_file,
            "append_file": self._append_file,
            "search_workspace": self._search_workspace,
            "run_powershell": self._run_powershell,
            "web_get": self._web_get,
            "gpu_status": self._gpu_status,
        }

    def schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "now",
                    "description": "Get the current UTC time and local process details.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "remember",
                    "description": "Store a durable memory fact for future runs.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "recall",
                    "description": "Search durable memory for facts relevant to a query.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                        },
                        "required": ["query"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_dir",
                    "description": "List a directory inside the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Relative path, default '.'"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        },
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read a UTF-8-ish text file inside the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "max_chars": {"type": "integer", "minimum": 1, "maximum": 50000},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write a text file inside the workspace. Requires write permission.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "append_file",
                    "description": "Append text to a file inside the workspace. Requires write permission.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_workspace",
                    "description": "Search text files in the workspace for a literal pattern.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pattern": {"type": "string"},
                            "glob": {"type": "string", "description": "File glob, default '**/*'"},
                            "limit": {"type": "integer", "minimum": 1, "maximum": 200},
                        },
                        "required": ["pattern"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "run_powershell",
                    "description": "Run a PowerShell command in the workspace. Requires shell permission.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                            "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120},
                        },
                        "required": ["command"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "web_get",
                    "description": "Fetch a URL and return a trimmed text response.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "max_chars": {"type": "integer", "minimum": 100, "maximum": 30000},
                        },
                        "required": ["url"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "gpu_status",
                    "description": "Inspect current NVIDIA GPU memory and driver state.",
                    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
                },
            },
        ]

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            return ToolResult(False, error=f"unknown tool: {name}").as_dict()
        try:
            return handler(arguments).as_dict()
        except Exception as exc:  # noqa: BLE001 - tool errors must be returned to the model.
            return ToolResult(False, error=f"{type(exc).__name__}: {exc}").as_dict()

    def _resolve(self, user_path: str | None) -> Path:
        path = self.context.root if not user_path else Path(user_path)
        if not path.is_absolute():
            path = self.context.root / path
        resolved = path.resolve()
        root = self.context.root.resolve()
        if resolved != root and root not in resolved.parents:
            raise ValueError(f"path escapes workspace: {user_path}")
        return resolved

    def _trim(self, text: str) -> str:
        limit = self.context.policy.max_tool_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n...[trimmed {len(text) - limit} chars]"

    def _now(self, _: dict[str, Any]) -> ToolResult:
        return ToolResult(
            True,
            {
                "utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "cwd": str(self.context.root),
                "python": sys.version.split()[0],
            },
        )

    def _remember(self, args: dict[str, Any]) -> ToolResult:
        rowid = self.context.memory.remember(str(args["text"]), list(args.get("tags", [])))
        return ToolResult(True, {"id": rowid})

    def _recall(self, args: dict[str, Any]) -> ToolResult:
        facts = self.context.memory.search(str(args["query"]), limit=int(args.get("limit", 5)))
        return ToolResult(
            True,
            [
                {"id": fact.id, "text": fact.text, "tags": fact.tags, "created_at": fact.created_at}
                for fact in facts
            ],
        )

    def _list_dir(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve(args.get("path", "."))
        limit = int(args.get("limit", 100))
        if not path.exists():
            return ToolResult(False, error=f"not found: {path.relative_to(self.context.root)}")
        if not path.is_dir():
            return ToolResult(False, error=f"not a directory: {path.relative_to(self.context.root)}")
        entries = []
        for child in sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))[:limit]:
            stat = child.stat()
            entries.append(
                {
                    "name": child.name,
                    "path": str(child.relative_to(self.context.root)),
                    "type": "dir" if child.is_dir() else "file",
                    "bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                }
            )
        return ToolResult(True, entries)

    def _read_file(self, args: dict[str, Any]) -> ToolResult:
        path = self._resolve(str(args["path"]))
        max_chars = int(args.get("max_chars", 20000))
        if not path.exists():
            return ToolResult(False, error=f"not found: {path.relative_to(self.context.root)}")
        if path.suffix.lower() not in TEXT_EXTENSIONS and path.stat().st_size > 1_000_000:
            return ToolResult(False, error="refusing to read a likely binary or huge non-text file")
        raw = path.read_bytes()
        for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                continue
        else:
            return ToolResult(False, error="could not decode file")
        return ToolResult(True, {"path": str(path.relative_to(self.context.root)), "content": text[:max_chars]})

    def _write_file(self, args: dict[str, Any]) -> ToolResult:
        if not self.context.policy.allow_write:
            return ToolResult(False, error="write_file denied by tool policy")
        path = self._resolve(str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        path.write_text(content, encoding="utf-8", newline="\n")
        return ToolResult(True, {"path": str(path.relative_to(self.context.root)), "bytes": len(content.encode("utf-8"))})

    def _append_file(self, args: dict[str, Any]) -> ToolResult:
        if not self.context.policy.allow_write:
            return ToolResult(False, error="append_file denied by tool policy")
        path = self._resolve(str(args["path"]))
        path.parent.mkdir(parents=True, exist_ok=True)
        content = str(args["content"])
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        return ToolResult(True, {"path": str(path.relative_to(self.context.root)), "appended_bytes": len(content.encode("utf-8"))})

    def _search_workspace(self, args: dict[str, Any]) -> ToolResult:
        pattern = str(args["pattern"])
        glob = str(args.get("glob", "**/*"))
        limit = int(args.get("limit", 50))
        hits: list[dict[str, Any]] = []
        ignored_dirs = {".git", "__pycache__", ".venv", "tools", "data", "runs"}
        for path in self.context.root.glob(glob):
            if len(hits) >= limit:
                break
            if path.is_dir():
                continue
            if any(part in ignored_dirs for part in path.relative_to(self.context.root).parts):
                continue
            if path.suffix.lower() not in TEXT_EXTENSIONS:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for index, line in enumerate(text.splitlines(), start=1):
                if pattern in line:
                    hits.append(
                        {
                            "path": str(path.relative_to(self.context.root)),
                            "line": index,
                            "text": line.strip()[:500],
                        }
                    )
                    if len(hits) >= limit:
                        break
        return ToolResult(True, hits)

    def _run_powershell(self, args: dict[str, Any]) -> ToolResult:
        if not self.context.policy.allow_shell:
            return ToolResult(False, error="run_powershell denied by tool policy")
        command = str(args["command"])
        lowered = f" {command.lower()} "
        if not self.context.policy.allow_unsafe_shell:
            for blocked in BLOCKED_SHELL_PATTERNS:
                if blocked in lowered:
                    return ToolResult(False, error=f"blocked unsafe command pattern: {blocked.strip()}")
        timeout = min(
            int(args.get("timeout_seconds", self.context.policy.command_timeout_seconds)),
            120,
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
            cwd=self.context.root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return ToolResult(
            completed.returncode == 0,
            {
                "returncode": completed.returncode,
                "stdout": self._trim(completed.stdout),
                "stderr": self._trim(completed.stderr),
            },
            None if completed.returncode == 0 else f"command exited {completed.returncode}",
        )

    def _web_get(self, args: dict[str, Any]) -> ToolResult:
        url = str(args["url"])
        max_chars = int(args.get("max_chars", 12000))
        request = Request(url, headers={"User-Agent": "ProtoAGI/0.1"})
        try:
            with urlopen(request, timeout=30) as response:
                content_type = response.headers.get("content-type", "")
                raw = response.read(max_chars + 1)
        except URLError as exc:
            return ToolResult(False, error=str(exc))
        text = raw.decode("utf-8", errors="replace")
        return ToolResult(True, {"url": url, "content_type": content_type, "text": text[:max_chars]})

    def _gpu_status(self, _: dict[str, Any]) -> ToolResult:
        exe = "nvidia-smi"
        command = [
            exe,
            "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv",
        ]
        try:
            completed = subprocess.run(
                command,
                cwd=self.context.root,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except OSError as exc:
            return ToolResult(False, error=str(exc))
        return ToolResult(
            completed.returncode == 0,
            {"stdout": completed.stdout.strip(), "stderr": completed.stderr.strip()},
            None if completed.returncode == 0 else "nvidia-smi failed",
        )


def default_registry(memory: MemoryStore, policy: ToolPolicy, root: Path = PROJECT_ROOT) -> ToolRegistry:
    return ToolRegistry(ToolContext(root=root.resolve(), memory=memory, policy=policy))


def result_to_tool_content(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)

