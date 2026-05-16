"""Pre-LLM gate for group-chat reactivity.

In private chats the bot always pays for a constrained-JSON ``decide_incoming``
call — that is fine, the user is talking to it. In groups every ambient line
would otherwise pay the same cost only to have the model decide to stay silent.

``GroupReactivityGate`` is a deterministic, cheap classifier that runs before
``decide_incoming``. It lets a turn through when one of the following holds:

* the chat is private (group rules do not apply);
* the message is addressed (alias / @username / "/" / reply to bot);
* the lower-cased text contains one of the configured trigger keywords;
* the per-chat cooldown window has elapsed *and* a deterministic sampler
  crosses ``passive_reply_ratio``.

When the gate skips, the orchestrator stays silent without running the LLM.
The gate is intentionally side-effect free except for the cooldown stamps it
records in ``kv``; tests inject a fake clock and a fake sampler.
"""

from __future__ import annotations

import hashlib
import re
import time
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable

from ..storage.memory import MemoryStore


GATE_KV_PREFIX = "telegram:group_gate:"


@dataclass(slots=True, frozen=True)
class GateDecision:
    allow: bool
    reason: str


@dataclass(slots=True, frozen=True)
class GroupGateConfig:
    cooldown_seconds: int = 120
    passive_reply_ratio: float = 0.04
    trigger_keywords: tuple[str, ...] = ()


class GroupReactivityGate:
    def __init__(
        self,
        memory: MemoryStore,
        config: GroupGateConfig,
        *,
        clock: Callable[[], float] | None = None,
        sampler: Callable[[], float] | None = None,
    ) -> None:
        self.memory = memory
        self.config = config
        self._clock = clock or time.time
        self._sampler = sampler
        self._keyword_patterns = tuple(_compile_keyword(kw) for kw in config.trigger_keywords)

    def evaluate(
        self,
        *,
        chat_id: str,
        chat_type: str,
        text: str,
        addressed: bool,
    ) -> GateDecision:
        if chat_type == "private":
            return GateDecision(True, "private")
        if addressed:
            return GateDecision(True, "addressed")
        if self._matches_keyword(text):
            return GateDecision(True, "keyword")
        last_passive = self._last_passive_at(chat_id)
        now = self._clock()
        if last_passive is not None and now - last_passive < self.config.cooldown_seconds:
            return GateDecision(False, "cooldown")
        if self.config.passive_reply_ratio <= 0.0:
            return GateDecision(False, "passive_disabled")
        roll = self._sampler() if self._sampler is not None else _deterministic_roll(chat_id, text)
        if roll >= self.config.passive_reply_ratio:
            return GateDecision(False, "passive_skip")
        self._record_passive(chat_id, now)
        return GateDecision(True, "passive_sample")

    def _matches_keyword(self, text: str) -> bool:
        if not self._keyword_patterns:
            return False
        normalized = _normalize_text(text)
        return any(pattern.search(normalized) for pattern in self._keyword_patterns)

    def _last_passive_at(self, chat_id: str) -> float | None:
        raw = self.memory.get_kv(GATE_KV_PREFIX + chat_id)
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def _record_passive(self, chat_id: str, now: float) -> None:
        self.memory.set_kv(GATE_KV_PREFIX + chat_id, f"{now:.0f}")


def _normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").casefold()


def _deterministic_roll(chat_id: str, text: str) -> float:
    key = f"{chat_id}\0{_normalize_text(text)}".encode("utf-8", errors="replace")
    digest = hashlib.sha256(key).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _compile_keyword(raw: str) -> "re.Pattern[str]":
    keyword = _normalize_text(raw).strip()
    # Word-boundary-ish match that works for Cyrillic too. ``\b`` anchors to
    # any non-letter on either side; we wrap with a class that includes
    # underscore for safety.
    pattern = rf"(?:^|[^\w]){re.escape(keyword)}(?:$|[^\w])"
    return re.compile(pattern, re.UNICODE)


def parse_trigger_keywords(raw: str | Iterable[str] | None) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, str):
        parts = raw.split(",")
    else:
        parts = list(raw)
    cleaned = []
    for part in parts:
        item = (part or "").strip()
        if item:
            cleaned.append(item)
    return tuple(cleaned)


__all__ = [
    "GATE_KV_PREFIX",
    "GateDecision",
    "GroupGateConfig",
    "GroupReactivityGate",
    "parse_trigger_keywords",
]
