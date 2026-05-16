"""Typed storage models and vector helpers."""

from __future__ import annotations

import array
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Memory item kinds. ``fact`` is the legacy bucket for unsorted writes coming
# from older callers that do not classify their memories.
KIND_FACT = "fact"
KIND_SEMANTIC = "semantic"
KIND_EPISODIC = "episodic"
KIND_PROCEDURAL = "procedural"
KIND_PERSONA_SELF = "persona_self"
ALL_KINDS = (KIND_FACT, KIND_SEMANTIC, KIND_EPISODIC, KIND_PROCEDURAL, KIND_PERSONA_SELF)

# Memory scopes describe how broadly a memory applies.
SCOPE_GLOBAL = "global"
SCOPE_USER = "user"
SCOPE_CHAT = "chat"
SCOPE_PERSONA = "persona"
ALL_SCOPES = (SCOPE_GLOBAL, SCOPE_USER, SCOPE_CHAT, SCOPE_PERSONA)

# Goal lifecycle states. ``open`` is the only state surfaced to the model as
# something it can still act on; the other two are terminal.
GOAL_STATUS_OPEN = "open"
GOAL_STATUS_COMPLETED = "completed"
GOAL_STATUS_ABANDONED = "abandoned"
ALL_GOAL_STATUSES = (GOAL_STATUS_OPEN, GOAL_STATUS_COMPLETED, GOAL_STATUS_ABANDONED)

# Conflict resolution states. A conflict is a pair of semantically-similar
# memory items that the consolidate pass did not auto-merge. ``unresolved``
# means the system has flagged the pair but not chosen a winner yet.
CONFLICT_STATUS_UNRESOLVED = "unresolved"
CONFLICT_STATUS_SUPERSEDED = "superseded"  # one side replaced the other
CONFLICT_STATUS_KEPT_BOTH = "kept_both"    # they describe distinct facets after all
CONFLICT_STATUS_DISMISSED = "dismissed"    # false positive; ignore
ALL_CONFLICT_STATUSES = (
    CONFLICT_STATUS_UNRESOLVED,
    CONFLICT_STATUS_SUPERSEDED,
    CONFLICT_STATUS_KEPT_BOTH,
    CONFLICT_STATUS_DISMISSED,
)


@dataclass(slots=True)
class MemoryFact:
    """Backwards-compatible view of a memory item used by older callers."""

    id: int
    text: str
    tags: list[str]
    created_at: str
    importance: float = 0.5
    kind: str = KIND_FACT
    scope: str = SCOPE_GLOBAL
    origin_message_id: str | None = None


@dataclass(slots=True)
class MemoryItem:
    id: int
    kind: str
    text: str
    scope: str
    user_id: str | None
    chat_id: str | None
    persona_key: str | None
    media_id: str | None
    importance: float
    confidence: float
    source: str | None
    supersedes_id: int | None
    superseded_by: int | None
    pinned: bool
    created_at: str
    updated_at: str | None
    last_accessed_at: str | None
    access_count: int
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    # Provenance: the Telegram message_id (or other external identifier)
    # that this memory was extracted from. Lets the model answer
    # "звідки я це знаю" by cross-referencing recent_telegram_messages.
    origin_message_id: str | None = None
    # Optional ISO 8601 expiry for short-lived "working memory" notes.
    # ``None`` (the default) means permanent — current behavior. When
    # set, recall and consolidation routines skip the row past this
    # timestamp; the reflection pass eventually hard-deletes it.
    expires_at: str | None = None


@dataclass(slots=True)
class MediaBlob:
    file_id: str
    mime: str
    sha256: str
    bytes: bytes
    caption: str
    created_at: str


@dataclass(slots=True)
class RecallResult:
    item: MemoryItem
    score: float
    bm25: float = 0.0
    cosine: float = 0.0


@dataclass(slots=True)
class UserProfile:
    user_id: str
    display_name: str
    language: str | None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


@dataclass(slots=True)
class Reminder:
    id: int
    user_id: str | None
    chat_id: str | None
    persona_key: str | None
    trigger_at: str
    text: str
    status: str
    created_at: str


@dataclass(slots=True)
class StickerDescription:
    """Cached vision-model description of a Telegram sticker.

    Generated once per sticker (background pass on bot startup), then
    reused for sticker selection. ``embedding`` is the optional dense
    vector we use to retrieve relevant stickers per turn. ``failure_reason``
    captures the last describer error so we can skip repeatedly-broken
    stickers without retrying them every restart.
    """

    sticker_id: str
    set_name: str
    emoji: str
    description: str
    embedding: list[float] | None
    embedding_model: str | None
    failure_reason: str | None
    attempt_count: int
    last_used_at: str | None
    created_at: str
    updated_at: str | None


@dataclass(slots=True)
class MemoryConflict:
    """A pair of memory items the system thinks might be in tension.

    The pair is normalized so ``memory_a_id < memory_b_id`` — that gives
    us a unique (a, b) index regardless of write order. Similarity is
    the cosine value at detection time; it doesn't get re-computed when
    items change. ``resolution_status`` is the action the persona /
    operator takes once the conflict is reviewed.
    """

    id: int
    memory_a_id: int
    memory_b_id: int
    similarity: float
    persona_key: str | None
    detected_at: str
    resolution_status: str
    resolution_winner_id: int | None
    resolved_at: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class UserState:
    """Persona-specific working model of a Telegram user.

    Each (user_id, persona_key) pair has at most one row. The state is
    refreshed periodically (every reflection pass, throttled by age) by
    an LLM call that consolidates recent messages + previous state into
    a new state. Mykola and Solomiya keep separate states because each
    persona attends to a different cross-section of the same person.

    The fields are deliberately small and structured (rather than a
    single free-form blob) so the model can scan them at a glance during
    decision-making.
    """

    user_id: str
    persona_key: str
    mood: str
    themes: list[str]
    open_questions: list[str]
    preferences: dict[str, Any]
    summary: str
    confidence: float
    last_updated_at: str
    messages_at_last_update: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Goal:
    """A persistent commitment / open thread the persona is tracking.

    Lives in its own table (not ``memory_items``) because it has a discrete
    lifecycle and is queried by status/due-date rather than by free-text
    recall. The persona scope is intentional: ``mykola`` and ``solomiya``
    keep their own goal lists, even when they share the chat memory.
    """

    id: int
    persona_key: str
    text: str
    status: str
    priority: float
    user_id: str | None
    chat_id: str | None
    origin_message_id: int | None
    due_at: str | None
    last_touched_at: str
    created_at: str
    updated_at: str | None
    closed_at: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TelegramChat:
    chat_id: str
    chat_type: str
    title: str | None
    username: str | None
    first_name: str | None
    last_name: str | None
    display_name: str
    reply_mode: str
    proactive_enabled: bool
    first_seen_at: str
    last_seen_at: str
    last_user_message_at: str | None
    last_bot_message_at: str | None
    last_initiative_at: str | None
    next_initiative_at: str | None
    metadata: dict[str, Any]


@dataclass(slots=True)
class TelegramMessage:
    chat_id: str
    message_id: int
    role: str
    sender_id: str | None
    sender_name: str
    text: str
    created_at: str
    metadata: dict[str, Any]


def pack_embedding(vector: Sequence[float]) -> bytes:
    """Pack a float vector into a compact little-endian float32 BLOB."""
    arr = array.array("f", (float(value) for value in vector))
    if hasattr(arr, "byteswap") and array.array("f").itemsize == 4:
        import sys

        if sys.byteorder == "big":
            arr.byteswap()
    return arr.tobytes()


def unpack_embedding(blob: bytes) -> list[float]:
    arr = array.array("f")
    arr.frombytes(blob)
    import sys

    if sys.byteorder == "big":
        arr.byteswap()
    return list(arr)


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = 0.0
    norm_left = 0.0
    norm_right = 0.0
    for l_value, r_value in zip(left, right):
        dot += l_value * r_value
        norm_left += l_value * l_value
        norm_right += r_value * r_value
    if norm_left <= 0.0 or norm_right <= 0.0:
        return 0.0
    return dot / (math.sqrt(norm_left) * math.sqrt(norm_right))


__all__ = [
    "ALL_CONFLICT_STATUSES",
    "ALL_GOAL_STATUSES",
    "ALL_KINDS",
    "ALL_SCOPES",
    "CONFLICT_STATUS_DISMISSED",
    "CONFLICT_STATUS_KEPT_BOTH",
    "CONFLICT_STATUS_SUPERSEDED",
    "CONFLICT_STATUS_UNRESOLVED",
    "GOAL_STATUS_ABANDONED",
    "GOAL_STATUS_COMPLETED",
    "GOAL_STATUS_OPEN",
    "Goal",
    "KIND_EPISODIC",
    "KIND_FACT",
    "KIND_PERSONA_SELF",
    "KIND_PROCEDURAL",
    "KIND_SEMANTIC",
    "MediaBlob",
    "MemoryConflict",
    "MemoryFact",
    "MemoryItem",
    "RecallResult",
    "Reminder",
    "SCOPE_CHAT",
    "SCOPE_GLOBAL",
    "SCOPE_PERSONA",
    "SCOPE_USER",
    "StickerDescription",
    "TelegramChat",
    "TelegramMessage",
    "UserProfile",
    "UserState",
    "cosine_similarity",
    "pack_embedding",
    "unpack_embedding",
    "utc_now",
]
