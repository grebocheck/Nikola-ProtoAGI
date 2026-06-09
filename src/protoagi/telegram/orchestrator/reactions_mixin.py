"""Bot-side emoji reaction policy, split out of the orchestrator core.

This mixin owns everything about *reacting* to a Telegram message with an
emoji (``setMessageReaction``): the per-chat cooldown, the learned
denylist of emoji a chat rejected, and the guarded send. It is mixed into
``NikolaBot``; the attributes and host methods it leans on are declared
below so the module type-checks under ``mypy --strict`` on its own.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any

from ..api import TelegramApi, TelegramApiError
from ..config import TelegramConfig
from ..reactions import (
    REACTION_COOLDOWN_KV_PREFIX,
    REACTION_DENYLIST_KV_PREFIX,
    REACTION_SENT_COUNT_KV_PREFIX,
    filter_allowed_emoji,
    parse_denylist,
    serialize_denylist,
)
from ..style import ReplyStyleTuner
from ...storage.memory import MemoryStore, TelegramChat


class TelegramReactionsMixin:
    # --- state owned by the host NikolaBot ---
    memory: MemoryStore
    telegram: TelegramApi
    telegram_config: TelegramConfig
    _style_tuner: ReplyStyleTuner

    if TYPE_CHECKING:
        # Real implementations live on NikolaBot; declared here so the
        # mixin's method bodies resolve under strict type checking.
        def _chat_allowed(self, chat_id: str) -> bool: ...
        def _chat_lock(self, chat_id: str) -> threading.Lock: ...

    def _handle_message_reaction(self, reaction: dict[str, Any]) -> bool:
        chat = reaction.get("chat") or {}
        if "id" not in chat:
            return False
        chat_id = str(chat["id"])
        if not self._chat_allowed(chat_id):
            return False
        emoji = ""
        new_reaction = reaction.get("new_reaction") or []
        if isinstance(new_reaction, list) and new_reaction:
            first = new_reaction[0]
            if isinstance(first, dict):
                emoji = str(first.get("emoji") or "")
        self._style_tuner.record_reaction(chat_id, emoji)
        return True

    def _reaction_denylist(self, chat_id: str) -> set[str]:
        raw = self.memory.get_kv(REACTION_DENYLIST_KV_PREFIX + chat_id)
        return parse_denylist(raw)

    def _reaction_cooldown_active(self, chat_id: str) -> bool:
        raw = self.memory.get_kv(REACTION_COOLDOWN_KV_PREFIX + chat_id)
        if not raw:
            return False
        try:
            last = float(raw)
        except ValueError:
            return False
        return (time.time() - last) < self.telegram_config.reaction_cooldown_seconds

    def _mark_reaction_sent(self, chat_id: str) -> None:
        self.memory.set_kv(REACTION_COOLDOWN_KV_PREFIX + chat_id, f"{time.time():.0f}")
        counter_key = REACTION_SENT_COUNT_KV_PREFIX + chat_id
        raw = self.memory.get_kv(counter_key)
        try:
            count = int(raw) if raw else 0
        except ValueError:
            count = 0
        self.memory.set_kv(counter_key, str(count + 1))

    def _record_reaction_denylist(self, chat_id: str, emoji: str) -> None:
        denylist = self._reaction_denylist(chat_id)
        if emoji in denylist:
            return
        denylist.add(emoji)
        self.memory.set_kv(REACTION_DENYLIST_KV_PREFIX + chat_id, serialize_denylist(denylist))

    def _apply_reactions(
        self,
        chat: TelegramChat,
        default_message_id: int | None,
        reactions: list[dict[str, Any]],
    ) -> None:
        if not self.telegram_config.reaction_enabled or not reactions:
            return
        choice = reactions[0]
        emoji_raw = str(choice.get("emoji") or "").strip()
        if not emoji_raw:
            return
        target_id = choice.get("message_id") or default_message_id
        if target_id is None:
            return
        try:
            target_int = int(target_id)
        except (TypeError, ValueError):
            return
        if target_int <= 0:
            return
        denylist = self._reaction_denylist(chat.chat_id)
        emoji = filter_allowed_emoji(emoji_raw, denylist)
        if emoji is None:
            print(
                f"[reaction] chat={chat.chat_id} skipped: emoji {ascii(emoji_raw)} not in whitelist or denylisted",
                flush=True,
            )
            return
        is_big = bool(choice.get("big"))
        with self._chat_lock(chat.chat_id):
            if self._reaction_cooldown_active(chat.chat_id):
                print(
                    f"[reaction] chat={chat.chat_id} skipped: cooldown active",
                    flush=True,
                )
                return
            try:
                self.telegram.set_message_reaction(
                    chat.chat_id,
                    target_int,
                    emoji,
                    is_big=is_big,
                )
            except TelegramApiError as exc:
                message = str(exc)
                if "REACTION_INVALID" in message or "REACTION_NOT_VALID" in message:
                    self._record_reaction_denylist(chat.chat_id, emoji)
                print(
                    f"[reaction] chat={chat.chat_id} api error for {ascii(emoji)}: {message}",
                    flush=True,
                )
                return
            self._mark_reaction_sent(chat.chat_id)
            print(
                f"[reaction] chat={chat.chat_id} sent {ascii(emoji)} on msg={target_int}"
                f"{' big' if is_big else ''}",
                flush=True,
            )
