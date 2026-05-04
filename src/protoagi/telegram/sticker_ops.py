"""Sticker selection and Telegram sticker-pack caching."""

from __future__ import annotations

import hashlib
import json

from ..storage.memory import MemoryStore, TelegramChat
from .api import TelegramApi, TelegramApiError
from .config import TelegramConfig
from .json_io import Decision, decision_reply_texts
from .stickers import auto_sticker_choice, looks_serious_for_sticker, normalize_sticker_pack


class TelegramStickerMixin:
    telegram: TelegramApi
    telegram_config: TelegramConfig
    memory: MemoryStore
    _sticker_cache: dict[str, list[dict[str, str]]]

    def _filter_decision_stickers(
        self,
        chat: TelegramChat,
        incoming_text: str,
        decision: Decision,
        *,
        is_initiative: bool = False,
    ) -> None:
        """Trim stickers the model already proposed when context says no.

        The LLM is asked (in the prompt) to default to text-only replies,
        but it still occasionally over-stickerizes. We post-filter:

        - drop *all* stickers on serious topics, regardless of source;
        - drop stickers in initiative messages unless the operator opted in
          via ``sticker_initiative_enabled``;
        - drop stickers when the reply text is paragraph-length;
        - drop stickers when the cooldown window has not elapsed since the
          last sticker we sent in this chat.
        """

        if not decision.stickers:
            return
        if looks_serious_for_sticker(incoming_text):
            decision.stickers = []
            return
        if is_initiative and not self.telegram_config.sticker_initiative_enabled:
            decision.stickers = []
            return
        reply_text = " ".join(decision_reply_texts(decision)).strip()
        if reply_text and len(reply_text) > self.telegram_config.sticker_max_reply_chars:
            decision.stickers = []
            return
        if self._recent_sticker_count(chat.chat_id) >= 1:
            decision.stickers = []

    def _maybe_add_reaction_sticker(self, chat: TelegramChat, incoming_text: str, decision: Decision) -> None:
        if decision.stickers or chat.chat_type != "private":
            return
        if not decision.should_reply or not decision_reply_texts(decision):
            return
        frequency = self.telegram_config.sticker_frequency
        if frequency == "off" or looks_serious_for_sticker(incoming_text):
            return
        if self._adaptive_style_arm(chat.chat_id) == "concise":
            # Bandit picked a quieter style for this chat; honour it.
            return
        choice = auto_sticker_choice(
            incoming_text,
            " ".join(decision_reply_texts(decision)),
            max_reply_chars=self.telegram_config.sticker_max_reply_chars,
        )
        if not choice:
            return
        if self._recent_sticker_count(chat.chat_id) >= 1:
            return
        thresholds = {"low": 12, "normal": 25, "high": 40, "always": 100}
        threshold = thresholds.get(frequency, 12)
        seed = f"{chat.chat_id}|{incoming_text}|{decision.reply}|{'|'.join(decision.replies)}"
        bucket = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket >= threshold:
            return
        decision.stickers.append(choice)

    def _adaptive_style_arm(self, chat_id: str | int) -> str:
        """Best-effort lookup of the current style-tuner arm for ``chat_id``.

        Returns ``"balanced"`` when the tuner is not wired in (older test
        fixtures) or when its state is unreadable, so callers can treat
        ``concise`` as the only special case worth silencing on.
        """

        tuner = getattr(self, "_style_tuner", None)
        if tuner is None:
            return "balanced"
        try:
            payload = tuner.state_payload(chat_id)
        except Exception:  # noqa: BLE001 - best effort hint
            return "balanced"
        return str(payload.get("last_choice") or "balanced")

    def _recent_sticker_count(self, chat_id: str | int) -> int:
        limit = max(6, self.telegram_config.sticker_cooldown_messages * 3)
        messages = self.memory.recent_telegram_messages(chat_id, limit=limit)
        user_messages_since_sticker = 0
        for item in reversed(messages):
            if str(item.get("text", "")).startswith("[sticker:"):
                return 1 if user_messages_since_sticker < self.telegram_config.sticker_cooldown_messages else 0
            if item.get("role") == "user":
                user_messages_since_sticker += 1
        return 0

    def _select_sticker_file_id(self, choice: dict[str, str], chat_id: str) -> str | None:
        pack = normalize_sticker_pack(choice.get("pack", ""))
        if not pack:
            return None
        stickers = self._load_sticker_pack(pack)
        if not stickers:
            return None
        emoji = str(choice.get("emoji", "") or "").strip()
        candidates = [item for item in stickers if emoji and emoji in item.get("emoji", "")]
        if not candidates:
            candidates = stickers
        seed = f"{chat_id}:{pack}:{emoji}:{choice.get('reason', '')}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        index = int.from_bytes(digest[:4], "big") % len(candidates)
        return candidates[index].get("file_id")

    def _load_sticker_pack(self, pack: str) -> list[dict[str, str]]:
        if pack in self._sticker_cache:
            return self._sticker_cache[pack]
        key = f"telegram:stickers:{pack}"
        cached = self.memory.get_kv(key)
        if cached:
            try:
                stickers = json.loads(cached)
                if isinstance(stickers, list):
                    self._sticker_cache[pack] = stickers
                    return stickers
            except json.JSONDecodeError:
                pass
        try:
            sticker_set = self.telegram.get_sticker_set(pack)
        except TelegramApiError:
            self._sticker_cache[pack] = []
            return []
        stickers = [
            {
                "file_id": str(item.get("file_id", "")),
                "emoji": str(item.get("emoji", "") or ""),
            }
            for item in sticker_set.get("stickers", [])
            if item.get("file_id")
        ]
        self.memory.set_kv(key, json.dumps(stickers, ensure_ascii=False))
        self._sticker_cache[pack] = stickers
        return stickers


__all__ = ["TelegramStickerMixin"]
