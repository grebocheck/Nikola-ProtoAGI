"""Stateless projection and prompt-compaction helpers for the orchestrator.

These functions were split out of ``orchestrator/__init__.py`` during the
Phase R refactor. They hold no bot state: each maps a typed row / decision
into the compact dict the model sees, or trims an oversized prompt payload
down to a configured budget. The ``NikolaBot`` class imports them back.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..json_io import Decision
from ..tool_runner import TelegramToolEvent
from ...storage.memory import Goal, MemoryItem


def _decision_to_payload(decision: Decision) -> dict[str, Any]:
    return {
        "should_reply": decision.should_reply,
        "reply": decision.reply,
        "replies": list(decision.replies),
        "reply_to": decision.reply_to,
        "stickers": list(decision.stickers),
        "voice_reply": decision.voice_reply,
        "memories": list(decision.memories),
        "self_memories": list(decision.self_memories),
        "reminders": list(decision.reminders),
        "tool_request": decision.tool_request,
        "next_check_minutes": decision.next_check_minutes,
        "goals": list(decision.goals),
    }


def _strip_telegram_prefix(user_id: str) -> str:
    """Strip the ``telegram:`` prefix that ``_user_id_for`` puts on Telegram ids.

    user_state rows and other persona-scoped data use the prefixed form
    so we can distinguish Telegram users from agent callers, but the
    Telegram message log stores raw ids. Anything that needs to join
    across the two has to peel the prefix.
    """

    if user_id.startswith("telegram:"):
        return user_id[len("telegram:"):]
    return user_id


def _format_origin_ref(chat_id: str | int, message_id: int | str | None) -> str | None:
    """Build a stable provenance reference like ``telegram:555:42``.

    Returns ``None`` when there is no message_id to anchor against, so
    legacy callers that pass nothing don't pollute the column with
    half-references.
    """

    if message_id is None:
        return None
    try:
        message_int = int(message_id)
    except (TypeError, ValueError):
        return None
    if message_int <= 0:
        return None
    return f"telegram:{chat_id}:{message_int}"


def _memory_pair_view(item: MemoryItem) -> dict[str, Any]:
    """Compact projection of a MemoryItem for the conflict-resolution prompt.

    Strips embedding, access counters and other noise the model doesn't
    need, while keeping the bits that actually help adjudication:
    id (so the verdict can name a winner), text, tags, created_at,
    origin (provenance), and importance.
    """

    return {
        "id": item.id,
        "text": item.text,
        "tags": list(item.tags),
        "created_at": item.created_at,
        "origin": getattr(item, "origin_message_id", None),
        "importance": item.importance,
    }


def _goal_summary(goal: Goal) -> dict[str, Any]:
    """Compact projection of a Goal row for prompt context.

    Strips heavy fields and adds a derived ``age_days`` / ``due_in_hours``
    so the model does not have to parse timestamps itself.
    """

    now = datetime.now(timezone.utc)
    try:
        created = datetime.fromisoformat(goal.created_at.replace("Z", "+00:00"))
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        age_days = max(0.0, round((now - created).total_seconds() / 86400.0, 1))
    except ValueError:
        age_days = 0.0
    due_in_hours: float | None = None
    if goal.due_at:
        try:
            due = datetime.fromisoformat(goal.due_at.replace("Z", "+00:00"))
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            due_in_hours = round((due - now).total_seconds() / 3600.0, 1)
        except ValueError:
            due_in_hours = None
    return {
        "id": goal.id,
        "text": goal.text,
        "priority": goal.priority,
        "age_days": age_days,
        "due_at": goal.due_at,
        "due_in_hours": due_in_hours,
    }


def _tool_event_payload(event: TelegramToolEvent) -> dict[str, Any]:
    return {
        "name": event.name,
        "arguments": event.arguments,
        "result": event.result,
    }


def _first_reply_excerpt(decision: Decision) -> str:
    for candidate in (decision.reply, *decision.replies):
        text = (candidate or "").strip()
        if text:
            return text
    return ""


def _is_unavailable_media_description(text: str) -> bool:
    value = str(text or "").strip().lower()
    if not value:
        return True
    markers = (
        "опис недоступ",
        "опис не вдався",
        "опис порож",
        "не вдалося витягнути кадр",
        "description unavailable",
        "no still frame",
        "no visual description",
    )
    return any(marker in value for marker in markers)


_PROMPT_LEVELS: tuple[dict[str, int], ...] = (
    {
        "text": 700,
        "incoming": 1400,
        "history": 10,
        "telegram": 10,
        "memory": 6,
        "self": 6,
        "goals": 5,
        "tools": 4,
    },
    {
        "text": 520,
        "incoming": 1000,
        "history": 8,
        "telegram": 8,
        "memory": 5,
        "self": 4,
        "goals": 4,
        "tools": 3,
    },
    {
        "text": 360,
        "incoming": 760,
        "history": 6,
        "telegram": 6,
        "memory": 4,
        "self": 3,
        "goals": 3,
        "tools": 2,
    },
    {
        "text": 240,
        "incoming": 520,
        "history": 4,
        "telegram": 4,
        "memory": 2,
        "self": 2,
        "goals": 2,
        "tools": 1,
    },
    {
        "text": 160,
        "incoming": 320,
        "history": 2,
        "telegram": 2,
        "memory": 1,
        "self": 1,
        "goals": 1,
        "tools": 1,
    },
)

_PROMPT_TEXT_KEYS = {
    "content",
    "display_name",
    "incoming_text",
    "message",
    "reason",
    "reasoning",
    "reply",
    "sender",
    "sender_name",
    "summary",
    "text",
    "with_text",
}
_PROMPT_LAST_ITEMS = {
    "recent_messages",
    "recent_telegram_messages",
    "recent_user_facts",
    "recent_self_memories",
}
_PROMPT_FIRST_ITEMS = {
    "due_goals",
    "goals",
    "known_persona_self_memory",
    "memory",
    "open_goals",
    "persona_self_lore",
    "relevant_memory",
    "reminders",
    "tool_results",
}


def _compact_prompt_payload(value: Any, *, level: int) -> Any:
    limits = _prompt_limits(level)
    if isinstance(value, dict):
        compact: dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key)
            if key_text == "metadata":
                continue
            if key_text == "persona":
                compact[key_text] = _compact_persona_payload(item)
                continue
            if key_text == "original_context":
                compact[key_text] = _compact_prompt_payload(item, level=level + 1)
                continue
            if key_text == "tool_results":
                compact[key_text] = _compact_prompt_list(
                    key_text,
                    item,
                    level=level + 1,
                    limits=limits,
                )
                continue
            if key_text == "incoming_text":
                compact[key_text] = _clip_text(str(item or ""), limits["incoming"])
                continue
            if key_text in _PROMPT_TEXT_KEYS and isinstance(item, str):
                compact[key_text] = _clip_text(item, limits["text"])
                continue
            if isinstance(item, list):
                compact[key_text] = _compact_prompt_list(
                    key_text,
                    item,
                    level=level,
                    limits=limits,
                )
                continue
            compact[key_text] = _compact_prompt_payload(item, level=level)
        if level >= 4:
            compact["context_truncated"] = True
        return compact
    if isinstance(value, list):
        return [_compact_prompt_payload(item, level=level) for item in value[: limits["memory"]]]
    if isinstance(value, str):
        return _clip_text(value, limits["text"])
    return value


def _compact_prompt_list(
    key: str,
    value: Any,
    *,
    level: int,
    limits: dict[str, int],
) -> list[Any]:
    if not isinstance(value, list):
        return []
    if key in _PROMPT_LAST_ITEMS:
        cap = limits["history"] if key != "recent_telegram_messages" else limits["telegram"]
        selected = value[-cap:]
    elif key in _PROMPT_FIRST_ITEMS:
        if key in {"open_goals", "due_goals", "goals", "reminders"}:
            cap = limits["goals"]
        elif key == "tool_results":
            cap = limits["tools"]
        elif key in {"known_persona_self_memory", "persona_self_lore", "recent_self_memories"}:
            cap = limits["self"]
        else:
            cap = limits["memory"]
        selected = value[:cap]
    else:
        selected = value[: limits["memory"]]
    return [_compact_prompt_payload(item, level=level) for item in selected]


def _prompt_limits(level: int) -> dict[str, int]:
    if level < 0:
        level = 0
    if level >= len(_PROMPT_LEVELS):
        return _PROMPT_LEVELS[-1]
    return _PROMPT_LEVELS[level]


def _clip_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    if max_chars <= 16:
        return text[:max_chars]
    return text[: max_chars - 13].rstrip() + "...[trimmed]"


def _compact_persona_payload(value: Any) -> Any:
    if not isinstance(value, dict):
        return value
    compact: dict[str, Any] = {}
    for key in ("key", "display_name", "aliases"):
        if key in value:
            compact[key] = value[key]
    return compact or value


def _minimal_prompt_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Last-resort valid JSON when an operator sets a tiny prompt budget."""

    compact = _compact_prompt_payload(payload, level=9)
    if not isinstance(compact, dict):
        return {"payload": compact, "context_truncated": True}
    minimal: dict[str, Any] = {"context_truncated": True}
    for key in (
        "persona",
        "fictional_self_enabled",
        "chat",
        "sender",
        "incoming_text",
        "adaptive_reply_style",
        "available_tools",
        "available_sticker_packs",
    ):
        if key in compact:
            minimal[key] = compact[key]
    for key in ("recent_messages", "recent_telegram_messages"):
        items = compact.get(key)
        if isinstance(items, list) and items:
            minimal[key] = items[-1:]
    for key in ("relevant_memory", "memory", "known_persona_self_memory", "open_goals", "due_goals"):
        items = compact.get(key)
        if isinstance(items, list) and items:
            minimal[key] = items[:1]
    state = compact.get("known_user_state")
    if isinstance(state, dict):
        minimal["known_user_state"] = {
            name: state[name]
            for name in ("summary", "mood", "confidence", "age_hours")
            if name in state
        }
    return minimal
