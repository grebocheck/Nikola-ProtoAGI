from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from ..memory import MemoryStore, utc_now


STYLE_ARMS = {
    "concise": {
        "reply_length": "short",
        "formality": "plain",
        "sticker_frequency": "low",
        "instruction": "Prefer one compact reply. Skip extra polish and avoid multi-paragraph answers.",
    },
    "balanced": {
        "reply_length": "medium",
        "formality": "natural",
        "sticker_frequency": "normal",
        "instruction": "Use a natural medium-length reply with the persona's usual warmth.",
    },
    "expressive": {
        "reply_length": "roomy",
        "formality": "playful",
        "sticker_frequency": "high",
        "instruction": "Allow a little more warmth, texture, and sticker use when the chat is light.",
    },
}
STYLE_ARM_ORDER = ("balanced", "concise", "expressive")

STYLE_STATE_PREFIX = "telegram:style:"
STYLE_LAST_SENT_PREFIX = "telegram:style:last_sent:"
STYLE_FEEDBACK_WINDOW = timedelta(hours=6)


@dataclass(slots=True)
class StyleChoice:
    arm: str
    payload: dict[str, Any]


class ReplyStyleTuner:
    """Small per-chat bandit for reply style hints.

    It stores all state in ``kv`` so older databases do not need another
    migration. The scoring is deterministic UCB-style rather than random
    Thompson sampling, which keeps tests and local debugging reproducible.
    """

    def __init__(self, memory: MemoryStore) -> None:
        self.memory = memory

    def choose(self, chat_id: str | int) -> StyleChoice:
        state = self._load_state(chat_id)
        trials_total = sum(int(item.get("trials", 0)) for item in state["arms"].values())
        best_arm = "balanced"
        best_score = -1.0
        for arm in STYLE_ARM_ORDER:
            stats = state["arms"].setdefault(arm, {"trials": 0, "successes": 0.0})
            trials = int(stats.get("trials", 0))
            successes = float(stats.get("successes", 0.0))
            mean = (successes + 1.0) / (trials + 2.0)
            explore = math.sqrt(math.log(max(2, trials_total + 1)) / (trials + 1))
            score = mean + 0.35 * explore
            if score > best_score:
                best_score = score
                best_arm = arm
        state["last_choice"] = best_arm
        state["updated_at"] = utc_now()
        self._save_state(chat_id, state)
        payload = dict(STYLE_ARMS[best_arm])
        payload["arm"] = best_arm
        payload["confidence"] = round(min(1.0, max(0.0, best_score / 2.0)), 3)
        return StyleChoice(best_arm, payload)

    def record_sent(
        self,
        chat_id: str | int,
        *,
        arm: str,
        reply_chars: int,
        sticker_count: int,
        message_count: int,
    ) -> None:
        self.memory.set_kv(
            self._last_sent_key(chat_id),
            json.dumps(
                {
                    "arm": arm if arm in STYLE_ARMS else "balanced",
                    "sent_at": utc_now(),
                    "reply_chars": max(0, int(reply_chars)),
                    "sticker_count": max(0, int(sticker_count)),
                    "message_count": max(0, int(message_count)),
                    "engaged": False,
                },
                ensure_ascii=False,
            ),
        )

    def record_incoming_reply(self, chat_id: str | int) -> None:
        self._record_signal(chat_id, "reply", 1.0)

    def record_reaction(self, chat_id: str | int, emoji: str = "") -> None:
        weight = 1.5 if emoji.strip() in {"❤️", "❤", "👍", "🔥", "😁", "😂", "🤣", "✨"} else 1.0
        self._record_signal(chat_id, "reaction", weight)

    def record_edit(self, chat_id: str | int) -> None:
        # Edits are weak engagement: the user cared enough to correct their
        # message, but it is less direct than a reply/reaction to the bot.
        self._record_signal(chat_id, "edit", 0.35)

    def state_payload(self, chat_id: str | int) -> dict[str, Any]:
        state = self._load_state(chat_id)
        return {
            "chat_id": str(chat_id),
            "arms": state["arms"],
            "signals": state["signals"],
            "last_choice": state.get("last_choice", "balanced"),
            "updated_at": state.get("updated_at"),
        }

    def _record_signal(self, chat_id: str | int, signal: str, weight: float) -> None:
        last = self._load_last_sent(chat_id)
        if not last or last.get("engaged"):
            return
        try:
            sent_at = datetime.fromisoformat(str(last.get("sent_at") or ""))
        except ValueError:
            return
        if sent_at.tzinfo is None:
            sent_at = sent_at.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - sent_at > STYLE_FEEDBACK_WINDOW:
            return
        arm = str(last.get("arm") or "balanced")
        state = self._load_state(chat_id)
        stats = state["arms"].setdefault(arm, {"trials": 0, "successes": 0.0})
        stats["trials"] = int(stats.get("trials", 0)) + 1
        stats["successes"] = float(stats.get("successes", 0.0)) + max(0.0, weight)
        state["signals"][signal] = int(state["signals"].get(signal, 0)) + 1
        state["updated_at"] = utc_now()
        self._save_state(chat_id, state)
        last["engaged"] = True
        last["engagement_signal"] = signal
        last["engagement_weight"] = weight
        self.memory.set_kv(self._last_sent_key(chat_id), json.dumps(last, ensure_ascii=False))

    def _load_state(self, chat_id: str | int) -> dict[str, Any]:
        raw = self.memory.get_kv(self._state_key(chat_id))
        try:
            state = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            state = {}
        if not isinstance(state, dict):
            state = {}
        arms = state.get("arms")
        if not isinstance(arms, dict):
            arms = {}
        for arm in STYLE_ARM_ORDER:
            stats = arms.get(arm)
            if not isinstance(stats, dict):
                stats = {"trials": 0, "successes": 0.0}
            stats["trials"] = max(0, int(stats.get("trials", 0)))
            stats["successes"] = max(0.0, float(stats.get("successes", 0.0)))
            arms[arm] = stats
        signals = state.get("signals")
        if not isinstance(signals, dict):
            signals = {}
        state["arms"] = arms
        state["signals"] = {str(key): int(value) for key, value in signals.items()}
        state.setdefault("last_choice", "balanced")
        return state

    def _save_state(self, chat_id: str | int, state: dict[str, Any]) -> None:
        self.memory.set_kv(self._state_key(chat_id), json.dumps(state, ensure_ascii=False))

    def _load_last_sent(self, chat_id: str | int) -> dict[str, Any] | None:
        raw = self.memory.get_kv(self._last_sent_key(chat_id))
        try:
            payload = json.loads(raw) if raw else None
        except json.JSONDecodeError:
            return None
        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _state_key(chat_id: str | int) -> str:
        return f"{STYLE_STATE_PREFIX}{chat_id}"

    @staticmethod
    def _last_sent_key(chat_id: str | int) -> str:
        return f"{STYLE_LAST_SENT_PREFIX}{chat_id}"


__all__ = ["ReplyStyleTuner", "STYLE_ARMS", "STYLE_ARM_ORDER", "StyleChoice"]
