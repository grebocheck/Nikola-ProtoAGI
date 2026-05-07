"""Bounded capture of deepseek ``reasoning_content`` per chat.

llama-server returns the chain-of-thought in ``message.reasoning_content`` when
started with ``--reasoning auto --reasoning-format deepseek``. We do not want
that text in the user-facing reply, but it is invaluable for debugging "why
did the bot decide to stay silent / what did it almost say".

``ReasoningLog`` keeps a small ring buffer per chat in the existing ``kv``
table. Storage shape::

    telegram:reasoning:<chat_id> -> JSON {"entries": [{...}, ...]}

Entries are appended on the right; oldest dropped when ``max_entries`` is
exceeded. The store is opt-in: if ``enabled=False`` the log silently no-ops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ..harmony import extract_analysis_content
from ..storage.memory import MemoryStore


REASONING_KV_PREFIX = "telegram:reasoning:"


@dataclass(slots=True, frozen=True)
class ReasoningLogConfig:
    enabled: bool = False
    max_entries_per_chat: int = 20
    max_chars_per_entry: int = 3000


@dataclass(slots=True, frozen=True)
class ReasoningEntry:
    chat_id: str
    message_id: int | None
    captured_at: str
    decision_kind: str
    incoming_text: str
    reasoning: str
    reply_excerpt: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "chat_id": self.chat_id,
            "message_id": self.message_id,
            "captured_at": self.captured_at,
            "decision_kind": self.decision_kind,
            "incoming_text": self.incoming_text,
            "reasoning": self.reasoning,
            "reply_excerpt": self.reply_excerpt,
        }


class ReasoningLog:
    def __init__(self, memory: MemoryStore, config: ReasoningLogConfig) -> None:
        self.memory = memory
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def record(
        self,
        *,
        chat_id: str,
        message_id: int | None,
        captured_at: str,
        decision_kind: str,
        incoming_text: str,
        reasoning: str,
        reply_excerpt: str,
    ) -> None:
        if not self.config.enabled:
            return
        text = (reasoning or "").strip()
        if not text:
            return
        entry = ReasoningEntry(
            chat_id=str(chat_id),
            message_id=message_id,
            captured_at=captured_at,
            decision_kind=decision_kind,
            incoming_text=_clip(incoming_text, self.config.max_chars_per_entry),
            reasoning=_clip(text, self.config.max_chars_per_entry),
            reply_excerpt=_clip(reply_excerpt, self.config.max_chars_per_entry),
        )
        existing = self._load(entry.chat_id)
        existing.append(entry.as_dict())
        if len(existing) > self.config.max_entries_per_chat:
            existing = existing[-self.config.max_entries_per_chat :]
        self.memory.set_kv(
            self._key(entry.chat_id),
            json.dumps({"entries": existing}, ensure_ascii=False),
        )

    def list_for_chat(self, chat_id: str) -> list[dict[str, Any]]:
        return list(self._load(str(chat_id)))

    def list_chats(self, *, limit: int = 50) -> list[dict[str, Any]]:
        with self.memory.connect() as conn:
            rows = conn.execute(
                "SELECT key, value, updated_at FROM kv WHERE key LIKE ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (REASONING_KV_PREFIX + "%", max(1, min(limit, 200))),
            ).fetchall()
        chats: list[dict[str, Any]] = []
        for row in rows:
            key = str(row["key"])
            chat_id = key[len(REASONING_KV_PREFIX) :]
            try:
                payload = json.loads(str(row["value"] or "{}"))
            except json.JSONDecodeError:
                payload = {}
            entries = payload.get("entries") if isinstance(payload, dict) else []
            count = len(entries) if isinstance(entries, list) else 0
            chats.append(
                {
                    "chat_id": chat_id,
                    "entries": count,
                    "updated_at": row["updated_at"],
                }
            )
        return chats

    def clear_chat(self, chat_id: str) -> None:
        self.memory.set_kv(self._key(str(chat_id)), json.dumps({"entries": []}))

    def _load(self, chat_id: str) -> list[dict[str, Any]]:
        raw = self.memory.get_kv(self._key(chat_id))
        if not raw:
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return []
        if not isinstance(payload, dict):
            return []
        entries = payload.get("entries")
        return [item for item in entries if isinstance(item, dict)] if isinstance(entries, list) else []

    def _key(self, chat_id: str) -> str:
        return REASONING_KV_PREFIX + chat_id


def extract_reasoning_text(message: dict[str, Any]) -> str:
    """Pull reasoning text out of an OpenAI-style chat completion message.

    Order of precedence:

    1. deepseek-style ``message.reasoning_content`` (when llama-server
       parses the reasoning channel itself);
    2. ``<think>...</think>`` blocks inside ``content`` (used by some
       providers and the legacy llama.cpp reasoning format);
    3. Harmony ``<|channel|>analysis<|message|>...`` blocks inside
       ``content`` — what gpt-oss models emit when llama-server runs
       with ``--skip-chat-parsing`` (the production setting in
       ``start-nikola-stack.ps1``).
    """

    if not isinstance(message, dict):
        return ""
    candidate = message.get("reasoning_content")
    if isinstance(candidate, str) and candidate.strip():
        return candidate
    content = message.get("content")
    if not isinstance(content, str):
        return ""
    if "<think>" in content:
        start = content.find("<think>") + len("<think>")
        end = content.find("</think>", start)
        if end > start:
            return content[start:end].strip()
    harmony = extract_analysis_content(content)
    if harmony:
        return harmony
    return ""


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


__all__ = [
    "REASONING_KV_PREFIX",
    "ReasoningEntry",
    "ReasoningLog",
    "ReasoningLogConfig",
    "extract_reasoning_text",
]
