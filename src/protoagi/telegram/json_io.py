"""Decision/initiative payload parsing.

The local model sometimes wraps JSON in markdown fences or adds Harmony-style
analysis text before the JSON object. ``extract_json_object`` is a forgiving
parser that tolerates both, and ``decision_from_payload`` /
``initiative_from_payload`` validate the resulting structure into typed
dataclasses.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .stickers import normalize_sticker_pack


@dataclass(slots=True)
class Decision:
    should_reply: bool
    reply: str
    memories: list[str]
    self_memories: list[str] = field(default_factory=list)
    replies: list[str] = field(default_factory=list)
    reply_to: str | int | None = None
    stickers: list[dict[str, str]] = field(default_factory=list)
    reminders: list[dict[str, Any]] = field(default_factory=list)
    next_check_minutes: int | None = None


@dataclass(slots=True)
class InitiativeDecision:
    send: bool
    message: str
    memories: list[str]
    next_check_minutes: int
    self_memories: list[str] = field(default_factory=list)
    stickers: list[dict[str, str]] = field(default_factory=list)
    reminders: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class ImageAttachment:
    file_id: str
    mime_type: str
    label: str
    file_name: str = ""


@dataclass(slots=True)
class StickerAttachment:
    file_id: str
    emoji: str
    set_name: str
    kind: str


_MAX_PARSE_SIZE = 64_000


_STICKER_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "pack": {"type": "string"},
        "emoji": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["pack"],
    "additionalProperties": False,
}


_REMINDER_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "in_minutes": {
            "anyOf": [{"type": "null"}, {"type": "integer", "minimum": 1}]
        },
        "trigger_at": {"anyOf": [{"type": "null"}, {"type": "string"}]},
    },
    "required": ["text"],
    "additionalProperties": False,
}


DECISION_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "telegram_decision",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "should_reply": {"type": "boolean"},
                "reply": {"type": "string"},
                "replies": {"type": "array", "items": {"type": "string"}},
                "reply_to": {
                    "anyOf": [
                        {"type": "null"},
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                },
                "stickers": {"type": "array", "items": _STICKER_ITEM_SCHEMA},
                "memories": {"type": "array", "items": {"type": "string"}},
                "self_memories": {"type": "array", "items": {"type": "string"}},
                "reminders": {"type": "array", "items": _REMINDER_ITEM_SCHEMA},
                "next_check_minutes": {
                    "anyOf": [{"type": "null"}, {"type": "integer"}]
                },
            },
            "required": ["should_reply"],
            "additionalProperties": False,
        },
    },
}


INITIATIVE_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "telegram_initiative",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "send": {"type": "boolean"},
                "message": {"type": "string"},
                "stickers": {"type": "array", "items": _STICKER_ITEM_SCHEMA},
                "memories": {"type": "array", "items": {"type": "string"}},
                "self_memories": {"type": "array", "items": {"type": "string"}},
                "reminders": {"type": "array", "items": _REMINDER_ITEM_SCHEMA},
                "next_check_minutes": {"type": "integer"},
            },
            "required": ["send"],
            "additionalProperties": False,
        },
    },
}


def extract_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    if len(text) > _MAX_PARSE_SIZE:
        # Bound the cost of pathological inputs while still leaving room for
        # legitimate large JSON.
        text = text[:_MAX_PARSE_SIZE]
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        payload = json.loads(text)
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def decision_from_payload(payload: dict[str, Any]) -> Decision:
    return Decision(
        should_reply=bool(payload.get("should_reply", False)),
        reply=str(payload.get("reply", "") or "").strip(),
        memories=[str(item).strip() for item in payload.get("memories", []) if str(item).strip()],
        self_memories=[str(item).strip() for item in payload.get("self_memories", []) if str(item).strip()],
        replies=normalize_reply_messages(payload.get("replies", [])),
        reply_to=normalize_reply_to(payload.get("reply_to")),
        stickers=normalize_sticker_choices(payload.get("stickers", [])),
        reminders=normalize_reminder_requests(payload.get("reminders", [])),
        next_check_minutes=_optional_int(payload.get("next_check_minutes")),
    )


def initiative_from_payload(payload: dict[str, Any]) -> InitiativeDecision:
    return InitiativeDecision(
        send=bool(payload.get("send", False)),
        message=str(payload.get("message", "") or "").strip(),
        memories=[str(item).strip() for item in payload.get("memories", []) if str(item).strip()],
        self_memories=[str(item).strip() for item in payload.get("self_memories", []) if str(item).strip()],
        next_check_minutes=max(30, _optional_int(payload.get("next_check_minutes")) or 360),
        stickers=normalize_sticker_choices(payload.get("stickers", [])),
        reminders=normalize_reminder_requests(payload.get("reminders", [])),
    )


def normalize_reminder_requests(value: Any) -> list[dict[str, Any]]:
    """Coerce reminder requests from the model into well-typed dicts."""

    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    requests: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        normalized: dict[str, Any] = {"text": text}
        in_minutes = _optional_int(item.get("in_minutes"))
        if in_minutes is not None and in_minutes > 0:
            normalized["in_minutes"] = in_minutes
        trigger_at = item.get("trigger_at")
        if isinstance(trigger_at, str) and trigger_at.strip():
            normalized["trigger_at"] = trigger_at.strip()
        if "in_minutes" not in normalized and "trigger_at" not in normalized:
            # Default to one hour ahead so the model's intent isn't lost.
            normalized["in_minutes"] = 60
        requests.append(normalized)
        if len(requests) >= 5:
            break
    return requests


def normalize_reply_to(value: Any) -> str | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if text.lower() in {"", "none", "null", "false", "no"}:
        return None
    if text.lower() == "current":
        return "current"
    if text.isdigit():
        return int(text)
    return None


def normalize_reply_messages(value: Any) -> list[str]:
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    messages: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            messages.append(text)
    return messages[:3]


def normalize_sticker_choices(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    choices: list[dict[str, str]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        pack = normalize_sticker_pack(str(item.get("pack", "")))
        if not pack:
            continue
        choices.append(
            {
                "pack": pack,
                "emoji": str(item.get("emoji", "") or ""),
                "reason": str(item.get("reason", "") or ""),
            }
        )
    return choices[:2]


def decision_reply_texts(decision: Decision) -> list[str]:
    source = decision.replies if decision.replies else [decision.reply]
    return [str(item).strip() for item in source if str(item).strip()]


def image_to_payload(image: ImageAttachment | None) -> dict[str, str] | None:
    if image is None:
        return None
    return {
        "file_id": image.file_id,
        "mime_type": image.mime_type,
        "label": image.label,
        "file_name": image.file_name,
    }


def sticker_to_payload(sticker: StickerAttachment | None) -> dict[str, str] | None:
    if sticker is None:
        return None
    return {
        "file_id": sticker.file_id,
        "emoji": sticker.emoji,
        "set_name": sticker.set_name,
        "kind": sticker.kind,
    }


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "DECISION_JSON_SCHEMA",
    "Decision",
    "INITIATIVE_JSON_SCHEMA",
    "ImageAttachment",
    "InitiativeDecision",
    "StickerAttachment",
    "decision_from_payload",
    "decision_reply_texts",
    "extract_json_object",
    "image_to_payload",
    "initiative_from_payload",
    "normalize_reminder_requests",
    "normalize_reply_messages",
    "normalize_reply_to",
    "normalize_sticker_choices",
    "sticker_to_payload",
]
