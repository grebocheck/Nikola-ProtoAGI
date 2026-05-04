from __future__ import annotations

import ipaddress
import json
import re
import socket
import ssl
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.error import URLError
from urllib.parse import ParseResult, urlparse

from ..config import PROJECT_ROOT, ToolPolicy
from ..storage.memory import MemoryStore


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

# Tokens that are highly likely to indicate a destructive intent. We compile
# them as standalone words to avoid the substring evasion trap that the older
# implementation suffered from (e.g. ``Remove-Itemy`` matching ``remove-item``).
BLOCKED_SHELL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(^|\s)remove-item\b", re.IGNORECASE),
    re.compile(r"(^|\s)ri\b", re.IGNORECASE),
    re.compile(r"(^|\s)rm\b", re.IGNORECASE),
    re.compile(r"(^|\s)rmdir\b", re.IGNORECASE),
    re.compile(r"(^|\s)del(\.exe)?\b", re.IGNORECASE),
    re.compile(r"(^|\s)erase\b", re.IGNORECASE),
    re.compile(r"(^|\s)format\b", re.IGNORECASE),
    re.compile(r"git\s+reset\s+--hard", re.IGNORECASE),
    re.compile(r"git\s+checkout\s+--", re.IGNORECASE),
    re.compile(r"set-executionpolicy", re.IGNORECASE),
    re.compile(r"takeown", re.IGNORECASE),
    re.compile(r"icacls", re.IGNORECASE),
    re.compile(r"reg\s+delete", re.IGNORECASE),
    re.compile(r"shutdown\b", re.IGNORECASE),
)


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
            "remind_me": self._remind_me,
            "list_reminders": self._list_reminders,
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
                            "importance": {"type": "number", "minimum": 0, "maximum": 1},
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
                    "description": "Fetch a public URL and return a trimmed text response.",
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
            {
                "type": "function",
                "function": {
                    "name": "remind_me",
                    "description": "Create a reminder for the bot to surface later.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "in_minutes": {"type": "integer", "minimum": 1, "maximum": 60 * 24 * 365},
                            "trigger_at": {
                                "type": "string",
                                "description": "ISO 8601 UTC timestamp; takes precedence over in_minutes if set.",
                            },
                            "user_id": {"type": "string"},
                            "chat_id": {"type": "string"},
                        },
                        "required": ["text"],
                        "additionalProperties": False,
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "list_reminders",
                    "description": "List pending reminders that are due now or in the near future.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                        },
                        "additionalProperties": False,
                    },
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
        rowid = self.context.memory.store_memory(
            str(args["text"]),
            tags=list(args.get("tags", [])),
            importance=float(args.get("importance", 0.5)),
            source="agent_remember",
        )
        return ToolResult(True, {"id": rowid})

    def _recall(self, args: dict[str, Any]) -> ToolResult:
        facts = self.context.memory.search(str(args["query"]), limit=int(args.get("limit", 5)))
        return ToolResult(
            True,
            [
                {
                    "id": fact.id,
                    "text": fact.text,
                    "tags": fact.tags,
                    "created_at": fact.created_at,
                    "importance": fact.importance,
                    "kind": fact.kind,
                }
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
        if not self.context.policy.allow_unsafe_shell:
            for pattern in BLOCKED_SHELL_PATTERNS:
                match = pattern.search(command)
                if match:
                    return ToolResult(
                        False,
                        error=f"blocked unsafe command pattern: {match.group(0).strip()}",
                    )
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
        try:
            content_type, raw = _fetch_public_url(url, max_chars=max_chars)
        except URLError as exc:
            return ToolResult(False, error=str(exc))
        except OSError as exc:
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

    def _remind_me(self, args: dict[str, Any]) -> ToolResult:
        text = str(args.get("text") or "").strip()
        if not text:
            return ToolResult(False, error="reminder text is required")
        trigger_at = str(args.get("trigger_at") or "").strip()
        if not trigger_at:
            in_minutes = int(args.get("in_minutes", 60))
            trigger = datetime.now(timezone.utc) + timedelta(minutes=max(1, in_minutes))
            trigger_at = trigger.isoformat(timespec="seconds")
        reminder_id = self.context.memory.add_reminder(
            text=text,
            trigger_at=trigger_at,
            user_id=args.get("user_id"),
            chat_id=args.get("chat_id"),
        )
        return ToolResult(True, {"id": reminder_id, "trigger_at": trigger_at, "text": text})

    def _list_reminders(self, args: dict[str, Any]) -> ToolResult:
        limit = int(args.get("limit", 10))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        items = self.context.memory.due_reminders(now, limit=limit)
        return ToolResult(
            True,
            [
                {
                    "id": item.id,
                    "text": item.text,
                    "trigger_at": item.trigger_at,
                    "user_id": item.user_id,
                    "chat_id": item.chat_id,
                    "status": item.status,
                }
                for item in items
            ],
        )


def default_registry(memory: MemoryStore, policy: ToolPolicy, root: Path = PROJECT_ROOT) -> ToolRegistry:
    return ToolRegistry(ToolContext(root=root.resolve(), memory=memory, policy=policy))


def result_to_tool_content(result: dict[str, Any]) -> str:
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# SSRF guard


_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata"}


@dataclass(slots=True)
class _ValidatedPublicUrl:
    parsed: ParseResult
    family: int
    socktype: int
    proto: int
    sockaddr: tuple[Any, ...]
    ip: str
    port: int
    host_header: str


def _validate_public_url(url: str) -> str | None:
    """Return an error message if the URL points at private infra.

    The check resolves the hostname and rejects any IP that falls into a
    private, loopback, link-local, multicast, or reserved range. Schemes are
    restricted to http(s); credentials embedded in the URL are also rejected.
    """

    _, error = _prepare_public_url(url)
    return error


def _prepare_public_url(url: str) -> tuple[_ValidatedPublicUrl | None, str | None]:
    try:
        parsed = urlparse(url)
    except ValueError as exc:
        return None, f"invalid url: {exc}"
    if parsed.scheme not in ("http", "https"):
        return None, "only http(s) URLs are allowed"
    if not parsed.hostname:
        return None, "url must contain a hostname"
    if parsed.username or parsed.password:
        return None, "credentials in url are not allowed"
    host = parsed.hostname.strip().lower()
    if host in _BLOCKED_HOSTS:
        return None, f"blocked hostname: {host}"
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        return None, f"invalid url port: {exc}"
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        return None, f"hostname resolution failed: {exc}"
    first_public: tuple[int, int, int, tuple[Any, ...], str] | None = None
    for info in infos:
        family, socktype, proto, _canonname, sockaddr = info
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            return None, f"unable to parse address: {addr}"
        if (
            ip.is_loopback
            or ip.is_private
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return None, f"blocked private address: {addr}"
        if first_public is None:
            first_public = (family, socktype, proto, sockaddr, addr)
    if first_public is None:
        return None, "hostname did not resolve to a usable public address"
    family, socktype, proto, sockaddr, ip = first_public
    host_header = host
    default_port = 443 if parsed.scheme == "https" else 80
    if port != default_port:
        host_header = f"{host}:{port}"
    return (
        _ValidatedPublicUrl(
            parsed=parsed,
            family=family,
            socktype=socktype,
            proto=proto,
            sockaddr=sockaddr,
            ip=ip,
            port=port,
            host_header=host_header,
        ),
        None,
    )


def _fetch_public_url(url: str, *, max_chars: int) -> tuple[str, bytes]:
    validated, error = _prepare_public_url(url)
    if error is not None or validated is None:
        raise URLError(error or "invalid url")
    sock: socket.socket | ssl.SSLSocket | None = None
    try:
        sock = socket.socket(validated.family, validated.socktype, validated.proto)
        sock.settimeout(30)
        sock.connect(validated.sockaddr)
        if validated.parsed.scheme == "https":
            context = ssl.create_default_context()
            sock = context.wrap_socket(sock, server_hostname=validated.parsed.hostname)
        return _read_http_response(sock, validated, max_chars=max_chars)
    finally:
        if sock is not None:
            sock.close()


def _read_http_response(
    sock: socket.socket | ssl.SSLSocket,
    validated: _ValidatedPublicUrl,
    *,
    max_chars: int,
) -> tuple[str, bytes]:
    target = validated.parsed.path or "/"
    if validated.parsed.query:
        target = f"{target}?{validated.parsed.query}"
    request = (
        f"GET {target} HTTP/1.1\r\n"
        f"Host: {validated.host_header}\r\n"
        "User-Agent: ProtoAGI/0.2\r\n"
        "Accept: text/*,*/*;q=0.1\r\n"
        "Connection: close\r\n\r\n"
    ).encode("ascii")
    sock.sendall(request)
    raw = b""
    limit = max(1, int(max_chars)) + 8192
    while len(raw) < limit:
        chunk = sock.recv(16384)
        if not chunk:
            break
        raw += chunk
    header_bytes, separator, body = raw.partition(b"\r\n\r\n")
    if not separator:
        raise URLError("invalid HTTP response")
    header_text = header_bytes.decode("iso-8859-1", errors="replace")
    lines = header_text.split("\r\n")
    status_line = lines[0] if lines else ""
    try:
        status_code = int(status_line.split()[1])
    except (IndexError, ValueError):
        status_code = 0
    if status_code >= 400:
        raise URLError(f"HTTP {status_code}")
    content_type = ""
    for line in lines[1:]:
        name, sep, value = line.partition(":")
        if sep and name.strip().lower() == "content-type":
            content_type = value.strip()
            break
    return content_type, body[: max(1, int(max_chars)) + 1]


__all__ = [
    "BLOCKED_SHELL_PATTERNS",
    "ToolContext",
    "ToolRegistry",
    "ToolResult",
    "default_registry",
    "result_to_tool_content",
]
