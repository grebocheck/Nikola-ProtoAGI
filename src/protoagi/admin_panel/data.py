"""JSON/data helpers for the local admin server."""

from __future__ import annotations

import json
import math
from typing import Any

from ..storage.memory import MemoryStore
from ..telegram.style import STYLE_ARM_ORDER, STYLE_LAST_SENT_PREFIX, STYLE_STATE_PREFIX


def stats(memory: MemoryStore) -> dict[str, Any]:
    with memory.connect() as conn:
        memories = int(conn.execute(
            "SELECT COUNT(*) FROM memory_items WHERE superseded_by IS NULL"
        ).fetchone()[0])
        superseded = int(conn.execute(
            "SELECT COUNT(*) FROM memory_items WHERE superseded_by IS NOT NULL"
        ).fetchone()[0])
        embeddings = int(conn.execute("SELECT COUNT(*) FROM memory_embeddings").fetchone()[0])
        reminders_pending = int(conn.execute(
            "SELECT COUNT(*) FROM reminders WHERE status = 'pending'"
        ).fetchone()[0])
        reminders_total = int(conn.execute("SELECT COUNT(*) FROM reminders").fetchone()[0])
        chats = int(conn.execute("SELECT COUNT(*) FROM telegram_chats").fetchone()[0])
        users = int(conn.execute("SELECT COUNT(*) FROM users").fetchone()[0])
        media_blobs = int(conn.execute("SELECT COUNT(*) FROM media_blobs").fetchone()[0])
        importance_cache = int(conn.execute("SELECT COUNT(*) FROM importance_cache").fetchone()[0])
        style_chats = int(
            conn.execute(
                "SELECT COUNT(*) FROM kv WHERE key LIKE 'telegram:style:%' "
                "AND key NOT LIKE 'telegram:style:last_sent:%'"
            ).fetchone()[0]
        )
    last_reflection = memory.get_kv("telegram:last_reflection_at")
    decision_metrics = _decision_metrics(memory.get_kv("telegram:decision_metrics"))
    return {
        "memories_active": memories,
        "memories_superseded": superseded,
        "embeddings": embeddings,
        "media_blobs": media_blobs,
        "importance_cache": importance_cache,
        "reminders_pending": reminders_pending,
        "reminders_total": reminders_total,
        "telegram_chats": chats,
        "users": users,
        "telegram_style_chats": style_chats,
        "last_reflection_at": last_reflection,
        **decision_metrics,
    }


def serialize_memory(item: Any) -> dict[str, Any]:
    return {
        "id": item.id,
        "kind": item.kind,
        "text": item.text,
        "scope": item.scope,
        "tags": list(item.tags),
        "importance": item.importance,
        "user_id": item.user_id,
        "chat_id": item.chat_id,
        "persona_key": item.persona_key,
        "media_id": item.media_id,
        "created_at": item.created_at,
        "updated_at": item.updated_at,
        "access_count": item.access_count,
        "pinned": item.pinned,
    }


def style_report(memory: MemoryStore) -> dict[str, Any]:
    chats_by_id: dict[str, dict[str, Any]] = {}
    with memory.connect() as conn:
        chat_rows = conn.execute(
            "SELECT chat_id, display_name, chat_type FROM telegram_chats"
        ).fetchall()
        for row in chat_rows:
            chats_by_id[str(row["chat_id"])] = {
                "display_name": row["display_name"],
                "chat_type": row["chat_type"],
            }
        state_rows = conn.execute(
            """
            SELECT key, value, updated_at
            FROM kv
            WHERE key LIKE ?
              AND key NOT LIKE ?
            ORDER BY updated_at DESC
            """,
            (f"{STYLE_STATE_PREFIX}%", f"{STYLE_LAST_SENT_PREFIX}%"),
        ).fetchall()

    aggregate = {
        arm: {"trials": 0, "successes": 0.0, "success_rate": 0.0}
        for arm in STYLE_ARM_ORDER
    }
    signal_totals: dict[str, int] = {}
    chats: list[dict[str, Any]] = []
    for row in state_rows:
        chat_id = str(row["key"])[len(STYLE_STATE_PREFIX) :]
        try:
            state = json.loads(str(row["value"] or "{}"))
        except json.JSONDecodeError:
            state = {}
        if not isinstance(state, dict):
            state = {}
        arms_raw = state.get("arms")
        arms = arms_raw if isinstance(arms_raw, dict) else {}
        arm_payload: dict[str, dict[str, Any]] = {}
        for arm in STYLE_ARM_ORDER:
            stats = arms.get(arm)
            if not isinstance(stats, dict):
                stats = {}
            trials = max(0, int(stats.get("trials", 0)))
            successes = max(0.0, float(stats.get("successes", 0.0)))
            aggregate[arm]["trials"] += trials
            aggregate[arm]["successes"] += successes
            arm_payload[arm] = {
                "trials": trials,
                "successes": round(successes, 3),
                "success_rate": round(successes / trials, 3) if trials else 0.0,
            }
        signals_raw = state.get("signals")
        signals = signals_raw if isinstance(signals_raw, dict) else {}
        signal_payload = {str(key): int(value) for key, value in signals.items()}
        for key, value in signal_payload.items():
            signal_totals[key] = signal_totals.get(key, 0) + value
        chat = chats_by_id.get(chat_id, {})
        chats.append(
            {
                "chat_id": chat_id,
                "display_name": chat.get("display_name", ""),
                "chat_type": chat.get("chat_type", ""),
                "active_arm": str(state.get("last_choice") or "balanced"),
                "arms": arm_payload,
                "signals": signal_payload,
                "updated_at": state.get("updated_at") or row["updated_at"],
            }
        )
    for item in aggregate.values():
        trials = int(item["trials"])
        successes = float(item["successes"])
        item["successes"] = round(successes, 3)
        item["success_rate"] = round(successes / trials, 3) if trials else 0.0
    return {
        "chats": chats,
        "aggregate": aggregate,
        "signals": signal_totals,
    }


def style_trials_cell(arms: Any) -> str:
    if not isinstance(arms, dict):
        return ""
    parts = []
    for arm in STYLE_ARM_ORDER:
        stats = arms.get(arm)
        if not isinstance(stats, dict):
            continue
        parts.append(f"{arm}:{int(stats.get('trials', 0))}")
    return " ".join(parts)


def style_signals_cell(signals: Any) -> str:
    if not isinstance(signals, dict):
        return ""
    return " ".join(f"{key}:{int(value)}" for key, value in sorted(signals.items()))


def memory_graph(
    memory: MemoryStore,
    *,
    limit: int = 120,
    scope: str | None = None,
    persona_key: str | None = None,
) -> dict[str, Any]:
    items = memory.list_memories(
        scope=scope or None,
        persona_key=persona_key or None,
        limit=max(1, min(limit, 500)),
        include_superseded=True,
    )
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    for item in items:
        node_id = f"memory:{item.id}"
        nodes[node_id] = {
            "id": node_id,
            "kind": "memory",
            "label": f"#{item.id} {item.kind}",
            "text": item.text[:160],
            "scope": item.scope,
            "importance": item.importance,
        }
        for tag in item.tags[:20]:
            tag_id = f"tag:{tag}"
            nodes.setdefault(
                tag_id,
                {"id": tag_id, "kind": "tag", "label": tag, "text": tag},
            )
            edges.append({"source": node_id, "target": tag_id, "kind": "tagged"})
        if item.supersedes_id:
            edges.append(
                {
                    "source": node_id,
                    "target": f"memory:{item.supersedes_id}",
                    "kind": "supersedes",
                }
            )
        if item.superseded_by:
            edges.append(
                {
                    "source": node_id,
                    "target": f"memory:{item.superseded_by}",
                    "kind": "superseded_by",
                }
            )
    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "filters": {
            "scope": scope or "",
            "persona_key": persona_key or "",
            "limit": max(1, min(limit, 500)),
        },
    }


def _decision_metrics(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {
            "telegram_decisions": 0,
            "telegram_decision_avg_llm_calls": 0.0,
            "telegram_tool_decision_avg_llm_calls": 0.0,
            "telegram_decision_p95_llm_calls": 0,
        }
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    decisions = int(payload.get("decisions", 0))
    llm_calls = int(payload.get("llm_calls", 0))
    tool_decisions = int(payload.get("tool_decisions", 0))
    tool_llm_calls = int(payload.get("tool_llm_calls", 0))
    histogram = payload.get("llm_call_histogram")
    if not isinstance(histogram, dict):
        histogram = {}
    return {
        "telegram_decisions": decisions,
        "telegram_decision_avg_llm_calls": round(llm_calls / decisions, 3)
        if decisions
        else 0.0,
        "telegram_tool_decision_avg_llm_calls": round(tool_llm_calls / tool_decisions, 3)
        if tool_decisions
        else 0.0,
        "telegram_decision_p95_llm_calls": _histogram_percentile(histogram, 0.95),
        "telegram_decision_max_llm_calls": int(payload.get("max_llm_calls", 0)),
    }


def _histogram_percentile(histogram: dict[str, Any], percentile: float) -> int:
    total = sum(int(value) for value in histogram.values())
    if total <= 0:
        return 0
    target = max(1, int(math.ceil(total * percentile)))
    seen = 0
    for raw_bucket, raw_count in sorted(
        histogram.items(),
        key=lambda item: int(item[0]) if str(item[0]).isdigit() else 0,
    ):
        seen += int(raw_count)
        if seen >= target:
            return int(raw_bucket)
    return 0


__all__ = [
    "memory_graph",
    "serialize_memory",
    "stats",
    "style_report",
    "style_signals_cell",
    "style_trials_cell",
]
