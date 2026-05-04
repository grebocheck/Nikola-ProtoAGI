"""Tiny localhost admin server for inspecting ProtoAGI state.

The server is intentionally minimal: stdlib ``http.server`` only, no
authentication beyond binding to localhost by default. It exposes a small
HTML dashboard at ``/`` plus JSON endpoints under ``/api/``:

- ``GET  /api/stats``                — counts and last-reflection timestamp
- ``GET  /api/memories?limit=...``   — recent memory items (any scope)
- ``GET  /api/reminders``            — pending reminders
- ``GET  /api/chats``                — Telegram chats
- ``POST /api/memories/<id>/delete`` — delete a memory item
- ``POST /api/memories/prune``       — run the prune pass (dry_run optional)
- ``POST /api/memories/*/preview``   — dry-run prune/consolidate with plans

The dashboard is single-file, server-rendered, no external assets, so it
works offline.
"""

from __future__ import annotations

import html
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
from typing import Any
from urllib.parse import parse_qs, urlparse

from .data import (
    memory_graph,
    serialize_memory,
    stats,
    style_report,
    style_signals_cell,
    style_trials_cell,
)
from ..storage.memory import MemoryStore
from ..storage.service import MemoryService


_stats = stats


class _ThreadingServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def _json_response(handler: BaseHTTPRequestHandler, payload: Any, status: int = 200) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _html_response(handler: BaseHTTPRequestHandler, body: str) -> None:
    encoded = body.encode("utf-8")
    handler.send_response(200)
    handler.send_header("Content-Type", "text/html; charset=utf-8")
    handler.send_header("Content-Length", str(len(encoded)))
    handler.end_headers()
    handler.wfile.write(encoded)


def _bytes_response(
    handler: BaseHTTPRequestHandler,
    data: bytes,
    *,
    content_type: str,
) -> None:
    handler.send_response(200)
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def _dashboard_html(memory: MemoryStore) -> str:
    stats_payload = stats(memory)
    style = style_report(memory)
    items = memory.list_memories(limit=20)
    chats = []
    with memory.connect() as conn:
        rows = conn.execute(
            "SELECT chat_id, display_name, chat_type, last_user_message_at, last_bot_message_at "
            "FROM telegram_chats ORDER BY last_seen_at DESC LIMIT 20"
        ).fetchall()
        for row in rows:
            chats.append(
                {
                    "chat_id": row["chat_id"],
                    "display_name": row["display_name"],
                    "chat_type": row["chat_type"],
                    "last_user_message_at": row["last_user_message_at"],
                    "last_bot_message_at": row["last_bot_message_at"],
                }
            )
    reminders = memory.due_reminders("9999-12-31T23:59:59+00:00", limit=20)
    rows_html = "".join(
        f"<tr data-id=\"{item.id}\" class=\"{'pinned' if item.pinned else ''}\">"
        f"<td>{item.id}</td>"
        f"<td>{html.escape(item.kind)}</td>"
        f"<td>{html.escape(item.scope)}</td>"
        f"<td><input class=\"imp\" type=\"number\" min=\"0\" max=\"1\" step=\"0.05\" "
        f"value=\"{item.importance:.2f}\"></td>"
        f"<td><textarea class=\"txt\" rows=\"2\">{html.escape(item.text)}</textarea></td>"
        f"<td>{html.escape(item.created_at)}</td>"
        f"<td class=\"actions\">"
        f"<button data-act=\"save\">save</button>"
        f"<button data-act=\"pin\">{('unpin' if item.pinned else 'pin')}</button>"
        f"<button data-act=\"delete\" class=\"danger\">delete</button>"
        f"</td></tr>"
        for item in items
    )
    chat_rows = "".join(
        f"<tr><td>{html.escape(str(chat['chat_id']))}</td>"
        f"<td>{html.escape(chat['display_name'] or '')}</td>"
        f"<td>{html.escape(chat['chat_type'])}</td>"
        f"<td>{html.escape(chat['last_user_message_at'] or '')}</td>"
        f"<td>{html.escape(chat['last_bot_message_at'] or '')}</td></tr>"
        for chat in chats
    )
    reminder_rows = "".join(
        f"<tr><td>{rem.id}</td><td>{html.escape(rem.text)}</td>"
        f"<td>{html.escape(rem.trigger_at)}</td>"
        f"<td>{html.escape(rem.chat_id or '')}</td>"
        f"<td>{html.escape(rem.persona_key or '')}</td></tr>"
        for rem in reminders
    )
    style_rows = "".join(
        "<tr>"
        f"<td>{html.escape(row['chat_id'])}</td>"
        f"<td>{html.escape(str(row.get('display_name') or ''))}</td>"
        f"<td>{html.escape(str(row.get('active_arm') or 'balanced'))}</td>"
        f"<td>{html.escape(style_trials_cell(row.get('arms', {})))}</td>"
        f"<td>{html.escape(style_signals_cell(row.get('signals', {})))}</td>"
        f"<td>{html.escape(str(row.get('updated_at') or ''))}</td>"
        "</tr>"
        for row in style["chats"]
    )
    aggregate_rows = "".join(
        "<tr>"
        f"<td>{html.escape(arm)}</td>"
        f"<td>{stats['trials']}</td>"
        f"<td>{stats['successes']}</td>"
        f"<td>{stats['success_rate']}</td>"
        "</tr>"
        for arm, stats in style["aggregate"].items()
    )
    stats_html = "".join(
        f"<dt>{html.escape(str(key))}</dt><dd>{html.escape(str(value))}</dd>"
        for key, value in stats_payload.items()
    )
    return f"""<!doctype html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>ProtoAGI admin</title>
<style>
 body {{ font-family: ui-sans-serif, system-ui, sans-serif; margin: 24px; max-width: 1100px; color:#222; }}
 h1, h2 {{ font-weight: 600; }}
 dl {{ display: grid; grid-template-columns: max-content 1fr; gap: 4px 16px; }}
 dt {{ color: #666; }}
 table {{ border-collapse: collapse; width: 100%; margin-bottom: 16px; }}
 th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; font-size: 14px; }}
 th {{ background: #f8f8f8; }}
 .pill {{ display: inline-block; padding: 1px 8px; border-radius: 12px; background:#eef; font-size: 12px; }}
 a {{ color: #2255cc; }}
 details {{ margin: 12px 0; }}
 tr.pinned {{ background: #fff8e1; }}
 .actions button {{ margin-right: 4px; padding: 2px 8px; cursor: pointer; }}
 .actions button.danger {{ color: #b00020; }}
 .imp {{ width: 64px; }}
 .txt {{ width: 100%; min-width: 280px; font-family: inherit; font-size: 13px; }}
 .flash {{ position: fixed; top: 12px; right: 12px; padding: 8px 14px;
          background: #2e7d32; color: white; border-radius: 4px;
          opacity: 0; transition: opacity .2s; }}
 .flash.show {{ opacity: 1; }}
 .flash.error {{ background: #b00020; }}
 .controls {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin: 8px 0 12px; }}
 .controls label {{ color: #555; font-size: 13px; }}
 .controls select, .controls input {{ margin-left: 4px; padding: 3px 6px; }}
</style>
</head>
<body>
<h1>ProtoAGI <span class=\"pill\">admin</span></h1>
<p>JSON endpoints: <a href=\"/api/stats\">/api/stats</a>,
<a href=\"/api/memories\">/api/memories</a>,
<a href=\"/api/reminders\">/api/reminders</a>,
<a href=\"/api/chats\">/api/chats</a>,
<a href=\"/api/style\">/api/style</a>,
<a href=\"/api/memory-graph\">/api/memory-graph</a>.</p>
<h2>Stats</h2>
<dl>{stats_html}</dl>
<h2>Recent memory (top 20)</h2>
<table><thead><tr><th>id</th><th>kind</th><th>scope</th><th>imp</th><th>text</th><th>created</th><th></th></tr></thead>
<tbody id=\"memory-tbody\">{rows_html}</tbody></table>
<h2>Pending reminders</h2>
<table><thead><tr><th>id</th><th>text</th><th>trigger</th><th>chat</th><th>persona</th></tr></thead>
<tbody>{reminder_rows or '<tr><td colspan=5>—</td></tr>'}</tbody></table>
<h2>Telegram chats</h2>
<table><thead><tr><th>chat_id</th><th>name</th><th>type</th><th>last user msg</th><th>last bot msg</th></tr></thead>
<tbody>{chat_rows or '<tr><td colspan=5>—</td></tr>'}</tbody></table>
<h2>Style</h2>
<table><thead><tr><th>arm</th><th>trials</th><th>successes</th><th>rate</th></tr></thead>
<tbody>{aggregate_rows}</tbody></table>
<table><thead><tr><th>chat_id</th><th>name</th><th>active arm</th><th>trials</th><th>signals</th><th>updated</th></tr></thead>
<tbody id=\"style-tbody\">{style_rows or '<tr><td colspan=6>—</td></tr>'}</tbody></table>
<h2>Memory graph</h2>
<div class=\"controls\">
  <label>scope
    <select id=\"graph-scope\">
      <option value=\"\">all</option>
      <option value=\"global\">global</option>
      <option value=\"user\">user</option>
      <option value=\"chat\">chat</option>
      <option value=\"persona\">persona</option>
    </select>
  </label>
  <label>persona <input id=\"graph-persona\" type=\"text\" placeholder=\"mykola\"></label>
  <label>limit
    <select id=\"graph-limit\">
      <option>50</option>
      <option selected>80</option>
      <option>120</option>
      <option>200</option>
      <option>500</option>
    </select>
  </label>
</div>
<canvas id=\"memory-graph\" width=\"1040\" height=\"420\" style=\"width:100%;height:420px;border:1px solid #eee;background:#fff\"></canvas>
<div id=\"flash\" class=\"flash\"></div>
<script>
const flash = (msg, isError) => {{
  const el = document.getElementById('flash');
  el.textContent = msg;
  el.classList.toggle('error', !!isError);
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 1800);
}};
const post = async (path, body) => {{
  const resp = await fetch(path, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify(body || {{}}),
  }});
  if (!resp.ok) throw new Error(await resp.text());
  return resp.json();
}};
document.getElementById('memory-tbody').addEventListener('click', async (event) => {{
  const button = event.target.closest('button[data-act]');
  if (!button) return;
  const tr = button.closest('tr');
  const id = tr.dataset.id;
  try {{
    if (button.dataset.act === 'save') {{
      const text = tr.querySelector('.txt').value;
      const importance = parseFloat(tr.querySelector('.imp').value);
      await post(`/api/memories/${{id}}/edit`, {{text, importance}});
      flash('saved');
    }} else if (button.dataset.act === 'pin') {{
      const result = await post(`/api/memories/${{id}}/pin`, {{}});
      tr.classList.toggle('pinned', result.pinned);
      button.textContent = result.pinned ? 'unpin' : 'pin';
      flash(result.pinned ? 'pinned' : 'unpinned');
    }} else if (button.dataset.act === 'delete') {{
      if (!confirm('Delete this memory?')) return;
      await post(`/api/memories/${{id}}/delete`, {{}});
      tr.remove();
      flash('deleted');
    }}
  }} catch (err) {{
    flash(err.message || 'error', true);
  }}
}});
const drawGraph = async () => {{
  const canvas = document.getElementById('memory-graph');
  const ctx = canvas.getContext('2d');
  const params = new URLSearchParams();
  const scope = document.getElementById('graph-scope').value;
  const persona = document.getElementById('graph-persona').value.trim();
  const limit = document.getElementById('graph-limit').value;
  if (scope) params.set('scope', scope);
  if (persona) params.set('persona', persona);
  params.set('limit', limit || '80');
  const graph = await fetch(`/api/memory-graph?${{params.toString()}}`).then(r => r.json());
  const nodes = graph.nodes.map((node, index) => ({{
    ...node,
    x: canvas.width / 2 + Math.cos(index) * 160,
    y: canvas.height / 2 + Math.sin(index * 1.7) * 130,
    vx: 0,
    vy: 0,
  }}));
  const byId = Object.fromEntries(nodes.map(node => [node.id, node]));
  const edges = graph.edges.filter(edge => byId[edge.source] && byId[edge.target]);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!nodes.length) {{
    ctx.fillStyle = '#666';
    ctx.font = '13px ui-sans-serif, system-ui, sans-serif';
    ctx.fillText('no graph data for current filters', 20, 32);
    return;
  }}
  for (let step = 0; step < 160; step++) {{
    for (let i = 0; i < nodes.length; i++) {{
      for (let j = i + 1; j < nodes.length; j++) {{
        const a = nodes[i], b = nodes[j];
        const dx = a.x - b.x, dy = a.y - b.y;
        const dist2 = Math.max(80, dx * dx + dy * dy);
        const force = 1600 / dist2;
        a.vx += dx * force; a.vy += dy * force;
        b.vx -= dx * force; b.vy -= dy * force;
      }}
    }}
    for (const edge of edges) {{
      const a = byId[edge.source], b = byId[edge.target];
      const dx = b.x - a.x, dy = b.y - a.y;
      a.vx += dx * 0.002; a.vy += dy * 0.002;
      b.vx -= dx * 0.002; b.vy -= dy * 0.002;
    }}
    for (const node of nodes) {{
      node.vx *= 0.82; node.vy *= 0.82;
      node.x = Math.max(18, Math.min(canvas.width - 18, node.x + node.vx));
      node.y = Math.max(18, Math.min(canvas.height - 18, node.y + node.vy));
    }}
  }}
  ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
  for (const edge of edges) {{
    const a = byId[edge.source], b = byId[edge.target];
    ctx.strokeStyle = edge.kind === 'tagged' ? '#d8d8d8' : '#8aa0d8';
    ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
  }}
  for (const node of nodes) {{
    const isTag = node.kind === 'tag';
    ctx.fillStyle = isTag ? '#f3f5f7' : '#e7f0ff';
    ctx.strokeStyle = isTag ? '#adb5bd' : '#5b7cba';
    ctx.beginPath(); ctx.arc(node.x, node.y, isTag ? 8 : 12, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = '#202124';
    ctx.fillText(String(node.label).slice(0, 24), node.x + 14, node.y + 4);
  }}
}};
for (const id of ['graph-scope', 'graph-persona', 'graph-limit']) {{
  document.getElementById(id).addEventListener('change', () => drawGraph().catch(() => {{}}));
}}
document.getElementById('graph-persona').addEventListener('input', () => drawGraph().catch(() => {{}}));
drawGraph().catch(() => {{}});
</script>
</body></html>
"""


def make_handler(memory: MemoryStore, service: MemoryService) -> type[BaseHTTPRequestHandler]:
    class AdminHandler(BaseHTTPRequestHandler):
        server_version = "ProtoAGIAdmin/0.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            # Quiet by default; admin server is intended for local dev.
            return None

        def do_GET(self) -> None:  # noqa: N802 - http.server interface
            parsed = urlparse(self.path)
            if parsed.path in ("/", "/index.html"):
                _html_response(self, _dashboard_html(memory))
                return
            if parsed.path == "/api/stats":
                _json_response(self, _stats(memory))
                return
            if parsed.path == "/api/style":
                _json_response(self, style_report(memory))
                return
            if parsed.path == "/api/memories":
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", ["50"])[0])
                items = memory.list_memories(limit=limit)
                _json_response(self, [serialize_memory(item) for item in items])
                return
            if parsed.path == "/api/memory-graph":
                params = parse_qs(parsed.query)
                limit = int(params.get("limit", ["120"])[0])
                scope = str(params.get("scope", [""])[0]).strip() or None
                persona = str(params.get("persona", [""])[0]).strip() or None
                _json_response(
                    self,
                    memory_graph(
                        memory,
                        limit=limit,
                        scope=scope,
                        persona_key=persona,
                    ),
                )
                return
            if parsed.path.startswith("/api/media/"):
                media_id = parsed.path.split("/api/media/", 1)[1]
                item = memory.get_media_blob(media_id)
                if item is None:
                    self.send_error(404, "media not found")
                    return
                _bytes_response(self, item.bytes, content_type=item.mime)
                return
            if parsed.path == "/api/reminders":
                items = memory.due_reminders("9999-12-31T23:59:59+00:00", limit=200)
                _json_response(
                    self,
                    [
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
                    ],
                )
                return
            if parsed.path == "/api/chats":
                with memory.connect() as conn:
                    rows = conn.execute(
                        "SELECT chat_id, display_name, chat_type, reply_mode, "
                        "proactive_enabled, last_seen_at, last_user_message_at, "
                        "last_bot_message_at FROM telegram_chats ORDER BY last_seen_at DESC"
                    ).fetchall()
                _json_response(
                    self,
                    [
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
                    ],
                )
                return
            self.send_error(404, "not found")

        def do_POST(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw.decode("utf-8")) if raw else {}
            except json.JSONDecodeError:
                self.send_error(400, "invalid json")
                return

            if parsed.path == "/api/memories/prune":
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

            if parsed.path == "/api/memories/prune/preview":
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

            if parsed.path.startswith("/api/memories/") and parsed.path.endswith("/delete"):
                try:
                    memory_id = int(parsed.path.split("/")[-2])
                except ValueError:
                    self.send_error(400, "invalid id")
                    return
                memory.delete_memory(memory_id)
                _json_response(self, {"deleted": memory_id})
                return

            if parsed.path.startswith("/api/memories/") and parsed.path.endswith("/pin"):
                try:
                    memory_id = int(parsed.path.split("/")[-2])
                except ValueError:
                    self.send_error(400, "invalid id")
                    return
                # Toggle by default; honor an explicit boolean if supplied.
                pinned_value = payload.get("pinned")
                if pinned_value is None:
                    current = memory.get_memory(memory_id)
                    if current is None:
                        self.send_error(404, "memory not found")
                        return
                    pinned_value = not current.pinned
                updated = memory.set_pinned(memory_id, bool(pinned_value))
                if updated is None:
                    self.send_error(404, "memory not found")
                    return
                _json_response(self, {"id": memory_id, "pinned": updated.pinned})
                return

            if parsed.path.startswith("/api/memories/") and parsed.path.endswith("/edit"):
                try:
                    memory_id = int(parsed.path.split("/")[-2])
                except ValueError:
                    self.send_error(400, "invalid id")
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
                    self.send_error(400, str(exc))
                    return
                if updated is None:
                    self.send_error(404, "memory not found")
                    return
                _json_response(self, serialize_memory(updated))
                return

            if parsed.path == "/api/memories/consolidate":
                result = service.consolidate(
                    scope=payload.get("scope"),
                    persona_key=payload.get("persona_key"),
                    chat_id=payload.get("chat_id"),
                    dry_run=bool(payload.get("dry_run", False)),
                    return_plan=bool(payload.get("return_plan", False)),
                )
                _json_response(self, result if isinstance(result, dict) else {"merged": result})
                return

            if parsed.path == "/api/memories/consolidate/preview":
                result = service.consolidate(
                    scope=payload.get("scope"),
                    persona_key=payload.get("persona_key"),
                    chat_id=payload.get("chat_id"),
                    dry_run=True,
                    return_plan=True,
                )
                _json_response(self, result)
                return

            self.send_error(404, "not found")

    return AdminHandler


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
