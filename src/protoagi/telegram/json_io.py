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
    voice_reply: bool = False
    reminders: list[dict[str, Any]] = field(default_factory=list)
    tool_request: dict[str, Any] | None = None
    next_check_minutes: int | None = None
    goals: list[dict[str, Any]] = field(default_factory=list)
    # Short-lived "working memory" notes. Each is stored as a fact with
    # a few-hour expiry — useful for ephemeral context like current
    # emotional state or in-flight test scenarios that should not bleed
    # into long-term memory.
    temporary_notes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class InitiativeDecision:
    send: bool
    message: str
    memories: list[str]
    next_check_minutes: int
    self_memories: list[str] = field(default_factory=list)
    stickers: list[dict[str, str]] = field(default_factory=list)
    reminders: list[dict[str, Any]] = field(default_factory=list)
    goals: list[dict[str, Any]] = field(default_factory=list)


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
    thumbnail_file_id: str = ""
    visual_description: str = ""


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


_TOOL_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "enum": ["recall", "remind_me", "web_search"]},
        "arguments": {"type": "object"},
    },
    "required": ["name"],
    "additionalProperties": False,
}


# Goal action — let the persona open / update / close threads of intention
# right inside the decision JSON. ``action`` is required; ``goal_id`` is
# only needed when revising or closing an existing goal.
_GOAL_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["open", "update", "complete", "abandon"],
        },
        "text": {"type": "string"},
        "goal_id": {"anyOf": [{"type": "null"}, {"type": "integer"}]},
        "due_at": {"anyOf": [{"type": "null"}, {"type": "string"}]},
        "priority": {
            "anyOf": [
                {"type": "null"},
                {"type": "number", "minimum": 0, "maximum": 1},
            ]
        },
    },
    "required": ["action"],
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
                "voice_reply": {"type": "boolean"},
                "memories": {"type": "array", "items": {"type": "string"}},
                "self_memories": {"type": "array", "items": {"type": "string"}},
                "reminders": {"type": "array", "items": _REMINDER_ITEM_SCHEMA},
                "tool_request": {
                    "anyOf": [{"type": "null"}, _TOOL_REQUEST_SCHEMA]
                },
                "next_check_minutes": {
                    "anyOf": [{"type": "null"}, {"type": "integer"}]
                },
                "goals": {"type": "array", "items": _GOAL_ACTION_SCHEMA},
                "temporary_notes": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["should_reply"],
            "additionalProperties": False,
        },
    },
}


USER_STATE_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "user_state",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "mood": {"type": "string"},
                "themes": {"type": "array", "items": {"type": "string"}},
                "open_questions": {"type": "array", "items": {"type": "string"}},
                "preferences": {"type": "object"},
                "summary": {"type": "string"},
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
            },
            "required": ["summary"],
            "additionalProperties": False,
        },
    },
}


CONFLICT_RESOLUTION_JSON_SCHEMA: dict[str, Any] = {
    "type": "json_schema",
    "json_schema": {
        "name": "conflict_resolution",
        "strict": False,
        "schema": {
            "type": "object",
            "properties": {
                "verdict": {
                    "type": "string",
                    "enum": ["superseded", "kept_both", "dismissed"],
                },
                "winner_id": {
                    "anyOf": [{"type": "null"}, {"type": "integer"}]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "reasoning": {"type": "string"},
            },
            "required": ["verdict", "confidence"],
            "additionalProperties": False,
        },
    },
}


@dataclass(slots=True)
class ConflictResolutionVerdict:
    """Model's call on a conflict pair.

    ``superseded``: one side is more accurate now; the loser gets a
    ``supersedes`` link pointing at the winner. ``winner_id`` MUST be
    one of the two memory ids in the conflict — otherwise we treat the
    verdict as malformed and leave the pair unresolved.

    ``kept_both``: they describe different facets, both stay active.
    ``dismissed``: false-positive match; ignore the pair going forward.
    """

    verdict: str
    winner_id: int | None
    confidence: float
    reasoning: str


def conflict_resolution_from_payload(payload: dict[str, Any]) -> ConflictResolutionVerdict:
    verdict = str(payload.get("verdict") or "").strip().lower()
    if verdict not in {"superseded", "kept_both", "dismissed"}:
        verdict = "dismissed"
    winner_raw = payload.get("winner_id")
    winner_id: int | None = None
    if isinstance(winner_raw, int) and winner_raw > 0:
        winner_id = winner_raw
    try:
        confidence = max(0.0, min(1.0, float(payload.get("confidence", 0.0))))
    except (TypeError, ValueError):
        confidence = 0.0
    reasoning = str(payload.get("reasoning") or "").strip()[:500]
    return ConflictResolutionVerdict(
        verdict=verdict,
        winner_id=winner_id,
        confidence=confidence,
        reasoning=reasoning,
    )


@dataclass(slots=True)
class UserStateUpdate:
    """Result of an LLM-driven user_state refresh.

    The summary is the only required field. Empty arrays / strings are
    valid — they mean "no theme/question worth tracking" rather than
    "no information available".
    """

    summary: str
    mood: str = ""
    themes: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    preferences: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.5


def user_state_from_payload(payload: dict[str, Any]) -> UserStateUpdate:
    summary = str(payload.get("summary") or "").strip()
    mood = str(payload.get("mood") or "").strip()
    themes_raw = payload.get("themes") or []
    themes = (
        [str(t).strip() for t in themes_raw if str(t).strip()][:10]
        if isinstance(themes_raw, list)
        else []
    )
    questions_raw = payload.get("open_questions") or []
    open_questions = (
        [str(q).strip() for q in questions_raw if str(q).strip()][:10]
        if isinstance(questions_raw, list)
        else []
    )
    preferences = payload.get("preferences") or {}
    if not isinstance(preferences, dict):
        preferences = {}
    confidence_raw = payload.get("confidence", 0.5)
    try:
        confidence = max(0.0, min(1.0, float(confidence_raw)))
    except (TypeError, ValueError):
        confidence = 0.5
    return UserStateUpdate(
        summary=summary,
        mood=mood,
        themes=themes,
        open_questions=open_questions,
        preferences=preferences,
        confidence=confidence,
    )


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
                "goals": {"type": "array", "items": _GOAL_ACTION_SCHEMA},
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
        voice_reply=_optional_bool(payload.get("voice_reply")),
        reminders=normalize_reminder_requests(payload.get("reminders", [])),
        tool_request=normalize_tool_request(payload.get("tool_request")),
        next_check_minutes=_optional_int(payload.get("next_check_minutes")),
        goals=normalize_goal_actions(payload.get("goals", [])),
        temporary_notes=[
            str(item).strip()
            for item in payload.get("temporary_notes", [])
            if str(item).strip()
        ][:5],
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
        goals=normalize_goal_actions(payload.get("goals", [])),
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


_VALID_GOAL_ACTIONS = {"open", "update", "complete", "abandon"}


def normalize_goal_actions(value: Any) -> list[dict[str, Any]]:
    """Coerce goal action items from the model into well-typed dicts.

    Each item must specify ``action`` ∈ {open, update, complete, abandon}.
    ``open`` and ``update`` need ``text`` (an empty string is a no-op and
    gets dropped). ``complete`` / ``abandon`` need ``goal_id``. Anything
    that does not fit these rules is silently discarded so a malformed
    item does not poison the whole decision.
    """

    if isinstance(value, dict):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []
    actions: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action") or "").strip().lower()
        if action not in _VALID_GOAL_ACTIONS:
            continue
        normalized: dict[str, Any] = {"action": action}
        text = str(item.get("text") or "").strip()
        if text:
            normalized["text"] = text
        goal_id = _optional_int(item.get("goal_id"))
        if goal_id is not None and goal_id > 0:
            normalized["goal_id"] = goal_id
        due_at = item.get("due_at")
        if isinstance(due_at, str) and due_at.strip():
            normalized["due_at"] = due_at.strip()
        priority = item.get("priority")
        if isinstance(priority, (int, float)):
            normalized["priority"] = max(0.0, min(1.0, float(priority)))
        # Validation per action type.
        if action == "open" and "text" not in normalized:
            continue
        if action == "update" and "goal_id" not in normalized:
            continue
        if action in ("complete", "abandon") and "goal_id" not in normalized:
            continue
        actions.append(normalized)
        if len(actions) >= 5:
            break
    return actions


def normalize_tool_request(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    name = str(value.get("name") or value.get("tool") or "").strip()
    if name not in {"recall", "remind_me", "web_search"}:
        return None
    arguments = value.get("arguments")
    if not isinstance(arguments, dict):
        arguments = {}
    return {"name": name, "arguments": dict(arguments)}


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
    payload = {
        "file_id": sticker.file_id,
        "emoji": sticker.emoji,
        "set_name": sticker.set_name,
        "kind": sticker.kind,
    }
    if sticker.thumbnail_file_id:
        payload["thumbnail_file_id"] = sticker.thumbnail_file_id
    if sticker.visual_description:
        payload["visual_description"] = sticker.visual_description
    return payload


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", ""}:
            return False
    return default


__all__ = [
    "CONFLICT_RESOLUTION_JSON_SCHEMA",
    "ConflictResolutionVerdict",
    "DECISION_JSON_SCHEMA",
    "Decision",
    "INITIATIVE_JSON_SCHEMA",
    "ImageAttachment",
    "InitiativeDecision",
    "StickerAttachment",
    "USER_STATE_JSON_SCHEMA",
    "UserStateUpdate",
    "conflict_resolution_from_payload",
    "user_state_from_payload",
    "decision_from_payload",
    "decision_reply_texts",
    "extract_json_object",
    "image_to_payload",
    "initiative_from_payload",
    "normalize_goal_actions",
    "normalize_reminder_requests",
    "normalize_reply_messages",
    "normalize_reply_to",
    "normalize_sticker_choices",
    "normalize_tool_request",
    "sticker_to_payload",
]
