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
    "ALL_KINDS",
    "ALL_SCOPES",
    "KIND_EPISODIC",
    "KIND_FACT",
    "KIND_PERSONA_SELF",
    "KIND_PROCEDURAL",
    "KIND_SEMANTIC",
    "MediaBlob",
    "MemoryFact",
    "MemoryItem",
    "RecallResult",
    "Reminder",
    "SCOPE_CHAT",
    "SCOPE_GLOBAL",
    "SCOPE_PERSONA",
    "SCOPE_USER",
    "TelegramChat",
    "TelegramMessage",
    "UserProfile",
    "cosine_similarity",
    "pack_embedding",
    "unpack_embedding",
    "utc_now",
]
