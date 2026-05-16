"""Localhost admin server for ProtoAGI.

JSON API plus optional static-file serving for the React SPA built from
``admin_panel/web/``. The server is intentionally minimal: stdlib
``http.server`` only, no auth beyond binding to localhost.

API surface
-----------

GET endpoints:

- ``/api/health`` — counts (memory active/superseded, conflicts, goals,
  user_state) for the admin overview.
- ``/api/stats`` — legacy compatibility snapshot (used by the old
  dashboard; kept so deployments mid-migration don't break).
- ``/api/memories?limit=N&kind=...&scope=...&persona=...&search=...&pinned=true|false``
- ``/api/memory-graph?...``
- ``/api/goals?status=open|completed|abandoned|all&persona=...&limit=N``
- ``/api/conflicts?status=unresolved|superseded|kept_both|dismissed|all&persona=...&limit=N``
- ``/api/user_state?persona=...``
- ``/api/reminders``
- ``/api/chats``
- ``/api/reasoning`` (overview), ``/api/reasoning/<chat_id>?limit=N`` (entries)
- ``/api/style``
- ``/api/media/<file_id>`` — raw blob bytes (image/voice)

POST endpoints:

- ``/api/memories/<id>/delete`` ``/pin`` ``/edit``
- ``/api/memories/prune[/preview]`` ``/api/memories/consolidate[/preview]``
- ``/api/goals/<id>/update`` (status / priority / text / due_at)
- ``/api/conflicts/<id>/resolve`` (status, optional winner_id)

Static serving: when ``admin_panel/web/dist/`` exists the server hands
out its assets and falls back to ``index.html`` for any path that does
not start with ``/api/`` so the React router owns client-side routes.
Dev mode runs Vite on a separate port and proxies ``/api/*`` here.
"""

from __future__ import annotations

import json
import mimetypes
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

from .data import (
    list_conflicts,
    list_goals,
    list_stickers,
    list_user_states,
    memory_graph,
    reasoning_entries,
    reasoning_overview,
    serialize_goal,
    serialize_memory,
    stats,
    style_report,
)
from ..storage.memory import MemoryStore
from ..storage.service import MemoryService
from ..telegram.api import TelegramApi, TelegramApiError


_WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"
_STATIC_ROOT = _WEB_DIST if _WEB_DIST.exists() else None


class _ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _cors_headers(handler: BaseHTTPRequestHandler) -> None:
    # CORS is enabled across the board so the Vite dev server (on a
    # different port) can hit the API directly. Production is same-origin
    # so this is harmless.
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    handler.send_header("Access-Control-Allow-Headers", "Content-Type")


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    _cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(body)


def _bytes_response(
    handler: BaseHTTPRequestHandler,
    data: bytes,
    *,
    content_type: str,
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    _cors_headers(handler)
    handler.end_headers()
    handler.wfile.write(data)


def _send_error_json(
    handler: BaseHTTPRequestHandler,
    status: int,
    message: str,
) -> None:
    _json_response(handler, {"error": message}, status=status)


def _serve_static(handler: BaseHTTPRequestHandler, request_path: str) -> bool:
    """Try to serve a file from the bundled SPA. Returns ``True`` if handled.

    For anything that isn't a real file but isn't under ``/api/``, we
    return ``index.html`` so client-side routing works. When the build
    hasn't been produced yet we return ``False`` so the caller can fall
    back to a 404 (or to the legacy in-tree dashboard, if any).
    """

    if _STATIC_ROOT is None:
        return False
    clean = request_path.lstrip("/")
    if clean.startswith("api/") or clean == "api":
        return False
    target = _STATIC_ROOT / clean if clean else _STATIC_ROOT / "index.html"
    if target.is_dir():
        target = target / "index.html"
    if not target.exists() or not target.is_file():
        # SPA fallback so React Router can render unknown routes.
        target = _STATIC_ROOT / "index.html"
        if not target.exists():
            return False
    mime, _ = mimetypes.guess_type(str(target))
    data = target.read_bytes()
    _bytes_response(handler, data, content_type=mime or "application/octet-stream")
    return True


def make_handler(memory: MemoryStore, service: MemoryService) -> type[BaseHTTPRequestHandler]:
    class AdminHandler(BaseHTTPRequestHandler):
        server_version = "ProtoAGIAdmin/0.2"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return None

        def do_OPTIONS(self) -> None:  # noqa: N802
            self.send_response(204)
            _cors_headers(self)
            self.end_headers()

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            # JSON endpoints first; static serving is the fallback so
            # /api/* never hits the file system.
            if path == "/api/health":
                _json_response(self, service.memory_health(
                    persona_key=str(query.get("persona", [""])[0]).strip() or None
                ))
                return
            if path == "/api/stats":
                _json_response(self, stats(memory))
                return
            if path == "/api/style":
                _json_response(self, style_report(memory))
                return
            if path == "/api/memories":
                _json_response(self, _query_memories(memory, query))
                return
            if path == "/api/goals":
                _json_response(self, list_goals(
                    memory,
                    status=str(query.get("status", ["open"])[0]) or "open",
                    persona_key=str(query.get("persona", [""])[0]).strip() or None,
                    limit=_query_int(query, "limit", 100),
                ))
                return
            if path == "/api/conflicts":
                _json_response(self, list_conflicts(
                    memory,
                    status=str(query.get("status", ["unresolved"])[0]) or "unresolved",
                    persona_key=str(query.get("persona", [""])[0]).strip() or None,
                    limit=_query_int(query, "limit", 100),
                ))
                return
            if path == "/api/user_state":
                _json_response(self, list_user_states(
                    memory,
                    persona_key=str(query.get("persona", [""])[0]).strip() or None,
                ))
                return
            if path == "/api/stickers":
                _json_response(self, list_stickers(
                    memory,
                    set_name=str(query.get("pack", [""])[0]).strip() or None,
                    described=str(query.get("described", ["all"])[0]).strip() or "all",
                    limit=_query_int(query, "limit", 1000),
                ))
                return
            if path == "/api/stickers/packs":
                _json_response(self, memory.list_sticker_packs())
                return
            if path.startswith("/api/sticker_thumbnail/"):
                sticker_id = path.split("/api/sticker_thumbnail/", 1)[1]
                if not sticker_id:
                    _send_error_json(self, 400, "missing sticker_id")
                    return
                blob = memory.get_media_blob(sticker_id)
                if blob is None:
                    _send_error_json(self, 404, "thumbnail not cached yet")
                    return
                _bytes_response(self, blob.bytes, content_type=blob.mime)
                return
            if path == "/api/reasoning":
                _json_response(self, reasoning_overview(memory))
                return
            if path.startswith("/api/reasoning/"):
                chat_id = path[len("/api/reasoning/") :].strip()
                if not chat_id:
                    _send_error_json(self, 400, "missing chat_id")
                    return
                _json_response(self, reasoning_entries(
                    memory, chat_id, limit=_query_int(query, "limit", 20)
                ))
                return
            if path == "/api/memory-graph":
                _json_response(self, memory_graph(
                    memory,
                    limit=_query_int(query, "limit", 120),
                    scope=str(query.get("scope", [""])[0]).strip() or None,
                    persona_key=str(query.get("persona", [""])[0]).strip() or None,
                ))
                return
            if path.startswith("/api/media/"):
                media_id = path.split("/api/media/", 1)[1]
                item = memory.get_media_blob(media_id)
                if item is None:
                    _send_error_json(self, 404, "media not found")
                    return
                _bytes_response(self, item.bytes, content_type=item.mime)
                return
            if path == "/api/reminders":
                items = memory.due_reminders("9999-12-31T23:59:59+00:00", limit=200)
                _json_response(self, [
                    {
                        "id": rem.id,
                        "text": rem.text,
                        "trigger_at": rem.trigger_at,
                        "chat_id": rem.chat_id,
                        "user_id": rem.user_id,
                        "persona_key": rem.persona_key,
                        "status": rem.status,
                        "created_at": rem.created_at,
                    }
                    for rem in items
                ])
                return
            if path == "/api/chats":
                with memory.connect() as conn:
                    rows = conn.execute(
                        "SELECT chat_id, display_name, chat_type, reply_mode, "
                        "proactive_enabled, last_seen_at, last_user_message_at, "
                        "last_bot_message_at FROM telegram_chats "
                        "ORDER BY last_seen_at DESC"
                    ).fetchall()
                _json_response(self, [
                    {
                        "chat_id": row["chat_id"],
                        "display_name": row["display_name"],
                        "chat_type": row["chat_type"],
                        "reply_mode": row["reply_mode"],
                        "proactive_enabled": bool(row["proactive_enabled"]),
                        "last_seen_at": row["last_seen_at"],
                        "last_user_message_at": row["last_user_message_at"],
                        "last_bot_message_at": row["last_bot_message_at"],
                    }
                    for row in rows
                ])
                return

            # /api/* with no match → JSON 404 (keeps the SPA fallback
            # from masking a typo in the front-end's fetch URL).
            if path.startswith("/api/"):
                _send_error_json(self, 404, "unknown api endpoint")
                return

            if _serve_static(self, path):
                return
            _send_error_json(self, 404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            path = parsed.path
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                _send_error_json(self, 400, "invalid json")
                return

            if path == "/api/memories/prune":
                result = service.prune(
                    scope=payload.get("scope"),
                    persona_key=payload.get("persona_key"),
                    chat_id=payload.get("chat_id"),
                    score_threshold=float(payload.get("score_threshold", 0.12)),
                    keep_newer_than_days=float(payload.get("keep_newer_than_days", 30.0)),
                    dry_run=bool(payload.get("dry_run", False)),
                    return_plan=bool(payload.get("return_plan", False)),
                )
                _json_response(self, result)
                return

            if path == "/api/memories/prune/preview":
                result = service.prune(
                    scope=payload.get("scope"),
                    persona_key=payload.get("persona_key"),
                    chat_id=payload.get("chat_id"),
                    score_threshold=float(payload.get("score_threshold", 0.12)),
                    keep_newer_than_days=float(payload.get("keep_newer_than_days", 30.0)),
                    dry_run=True,
                    return_plan=True,
                )
                _json_response(self, result)
                return

            if path.startswith("/api/memories/") and path.endswith("/delete"):
                try:
                    memory_id = int(path.split("/")[-2])
                except ValueError:
                    _send_error_json(self, 400, "invalid id")
                    return
                memory.delete_memory(memory_id)
                _json_response(self, {"deleted": memory_id})
                return

            if path.startswith("/api/memories/") and path.endswith("/pin"):
                try:
                    memory_id = int(path.split("/")[-2])
                except ValueError:
                    _send_error_json(self, 400, "invalid id")
                    return
                pinned_value = payload.get("pinned")
                if pinned_value is None:
                    current = memory.get_memory(memory_id)
                    if current is None:
                        _send_error_json(self, 404, "memory not found")
                        return
                    pinned_value = not current.pinned
                updated = memory.set_pinned(memory_id, bool(pinned_value))
                if updated is None:
                    _send_error_json(self, 404, "memory not found")
                    return
                _json_response(self, {"id": memory_id, "pinned": updated.pinned})
                return

            if path.startswith("/api/memories/") and path.endswith("/edit"):
                try:
                    memory_id = int(path.split("/")[-2])
                except ValueError:
                    _send_error_json(self, 400, "invalid id")
                    return
                tags = payload.get("tags")
                try:
                    updated = memory.update_memory(
                        memory_id,
                        text=payload.get("text"),
                        importance=(
                            float(payload["importance"])
                            if "importance" in payload and payload["importance"] is not None
                            else None
                        ),
                        tags=tags if isinstance(tags, list) else None,
                    )
                except ValueError as exc:
                    _send_error_json(self, 400, str(exc))
                    return
                if updated is None:
                    _send_error_json(self, 404, "memory not found")
                    return
                _json_response(self, serialize_memory(updated))
                return

            if path == "/api/memories/consolidate":
                result = service.consolidate(
                    scope=payload.get("scope"),
                    persona_key=payload.get("persona_key"),
                    chat_id=payload.get("chat_id"),
                    dry_run=bool(payload.get("dry_run", False)),
                    return_plan=bool(payload.get("return_plan", False)),
                )
                _json_response(self, result if isinstance(result, dict) else {"merged": result})
                return

            if path == "/api/memories/consolidate/preview":
                result = service.consolidate(
                    scope=payload.get("scope"),
                    persona_key=payload.get("persona_key"),
                    chat_id=payload.get("chat_id"),
                    dry_run=True,
                    return_plan=True,
                )
                _json_response(self, result)
                return

            # ---------- Goals ----------
            if path.startswith("/api/goals/") and path.endswith("/update"):
                try:
                    goal_id = int(path.split("/")[-2])
                except ValueError:
                    _send_error_json(self, 400, "invalid id")
                    return
                kwargs: dict[str, Any] = {}
                if "status" in payload:
                    kwargs["status"] = str(payload["status"]).strip()
                if "text" in payload and payload["text"] is not None:
                    kwargs["text"] = str(payload["text"])
                if "priority" in payload and payload["priority"] is not None:
                    kwargs["priority"] = float(payload["priority"])
                if "due_at" in payload:
                    raw_due = payload["due_at"]
                    kwargs["due_at"] = (
                        str(raw_due) if isinstance(raw_due, str) and raw_due.strip() else None
                    )
                try:
                    updated = memory.update_goal(goal_id, **kwargs)
                except ValueError as exc:
                    _send_error_json(self, 400, str(exc))
                    return
                if updated is None:
                    _send_error_json(self, 404, "goal not found")
                    return
                _json_response(self, serialize_goal(updated))
                return

            # ---------- Stickers ----------
            if path == "/api/stickers/reset":
                pack = str(payload.get("pack") or "").strip() or None
                only_failed = bool(payload.get("only_failed", True))
                clear_descriptions = bool(payload.get("clear_descriptions", False))
                # Optional explicit list of stickers the operator picked
                # in the UI. When the ``sticker_ids`` key is present we
                # use it verbatim — even if empty (which means "nothing
                # selected, do not reset anything"). When the key is
                # missing entirely we fall through to pack + only_failed.
                sticker_ids: list[str] | None = None
                if "sticker_ids" in payload and isinstance(payload["sticker_ids"], list):
                    sticker_ids = [
                        str(item)
                        for item in payload["sticker_ids"]
                        if str(item).strip()
                    ]
                reset = memory.reset_sticker_describer_attempts(
                    set_name=pack,
                    sticker_ids=sticker_ids,
                    only_failed=only_failed,
                    clear_descriptions=clear_descriptions,
                )
                _json_response(self, {
                    "reset": reset,
                    "pack": pack,
                    "only_failed": only_failed,
                    "clear_descriptions": clear_descriptions,
                    "sticker_ids_count": len(sticker_ids) if sticker_ids else 0,
                })
                return
            if path.startswith("/api/stickers/") and path.endswith("/redescribe"):
                # Per-sticker re-caption: clear this row's description
                # and reset its attempt_count so the describer worker
                # picks it up on its next polling cycle.
                sticker_id = path[len("/api/stickers/"):-len("/redescribe")]
                if not sticker_id:
                    _send_error_json(self, 400, "missing sticker_id")
                    return
                if memory.get_sticker_description(sticker_id) is None:
                    _send_error_json(self, 404, "sticker not found")
                    return
                reset = memory.reset_sticker_describer_attempts(
                    sticker_id=sticker_id,
                    only_failed=False,
                    clear_descriptions=True,
                )
                _json_response(self, {"sticker_id": sticker_id, "queued": reset > 0})
                return

            # ---------- Conflicts ----------
            if path.startswith("/api/conflicts/") and path.endswith("/resolve"):
                try:
                    conflict_id = int(path.split("/")[-2])
                except ValueError:
                    _send_error_json(self, 400, "invalid id")
                    return
                status_value = str(payload.get("status") or "").strip()
                if not status_value:
                    _send_error_json(self, 400, "status required")
                    return
                winner_raw = payload.get("winner_id")
                winner_id = int(winner_raw) if isinstance(winner_raw, int) else None
                try:
                    updated = memory.resolve_conflict(
                        conflict_id,
                        status=status_value,
                        winner_id=winner_id,
                    )
                except ValueError as exc:
                    _send_error_json(self, 400, str(exc))
                    return
                if updated is None:
                    _send_error_json(self, 404, "conflict not found")
                    return
                # If the operator superseded one side, mirror that in
                # memory_items so the loser stops appearing in recall.
                if status_value == "superseded" and winner_id is not None:
                    loser = (
                        updated.memory_b_id if winner_id == updated.memory_a_id
                        else updated.memory_a_id
                    )
                    memory.supersede(loser, winner_id)
                _json_response(self, {
                    "id": updated.id,
                    "status": updated.resolution_status,
                    "winner_id": updated.resolution_winner_id,
                    "resolved_at": updated.resolved_at,
                })
                return

            _send_error_json(self, 404, "unknown endpoint")

    return AdminHandler


def _query_int(query: dict[str, list[str]], key: str, default: int) -> int:
    try:
        raw = query.get(key, [str(default)])[0]
        return int(raw)
    except (TypeError, ValueError):
        return default


def _query_memories(memory: MemoryStore, query: dict[str, list[str]]) -> list[dict[str, Any]]:
    limit = _query_int(query, "limit", 100)
    kind = str(query.get("kind", [""])[0]).strip() or None
    scope = str(query.get("scope", [""])[0]).strip() or None
    persona = str(query.get("persona", [""])[0]).strip() or None
    search = str(query.get("search", [""])[0]).strip()
    pinned_raw = str(query.get("pinned", [""])[0]).strip().lower()
    pinned_filter: bool | None = None
    if pinned_raw in ("1", "true", "yes"):
        pinned_filter = True
    elif pinned_raw in ("0", "false", "no"):
        pinned_filter = False

    if search:
        items = memory.fts_candidates(search, limit=limit * 2)
    else:
        items = memory.list_memories(
            scope=scope,
            persona_key=persona,
            kind=kind,
            limit=limit * 2,
        )
    out: list[dict[str, Any]] = []
    for item in items:
        if kind and item.kind != kind:
            continue
        if scope and item.scope != scope:
            continue
        if persona and item.persona_key != persona:
            continue
        if pinned_filter is not None and bool(item.pinned) != pinned_filter:
            continue
        out.append(serialize_memory(item))
        if len(out) >= limit:
            break
    return out


def serve(
    memory: MemoryStore,
    service: MemoryService,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> _ThreadingServer:
    handler = make_handler(memory, service)
    server = _ThreadingServer((host, port), handler)
    return server


__all__ = ["make_handler", "serve"]
