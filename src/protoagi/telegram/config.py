"""Telegram bot configuration loaded from env."""

from __future__ import annotations

import os
from dataclasses import dataclass

from ..env import env_bool, env_int
from ..persona import get_persona, resolve_persona_key


@dataclass(slots=True)
class TelegramConfig:
    token: str
    persona_key: str = "mykola"
    bot_name: str = "Микола"
    allowed_chat_ids: set[str] | None = None
    reply_mode: str = "smart"
    poll_timeout_seconds: int = 25
    max_reply_chars: int = 3900
    max_history_messages: int = 14
    max_memory_facts: int = 6
    decision_max_tokens: int = 768
    max_reply_messages: int = 3
    sticker_frequency: str = "normal"
    sticker_cooldown_messages: int = 3
    fictional_self_enabled: bool = True
    global_memory: bool = True
    proactive_enabled: bool = True
    proactive_check_seconds: int = 300
    proactive_cooldown_seconds: int = 6 * 60 * 60
    proactive_disable_notification: bool = True
    vision_base_url: str = ""
    vision_model: str = ""
    vision_max_bytes: int = 8 * 1024 * 1024
    vision_timeout_seconds: int = 120

    def __post_init__(self) -> None:
        self.set_persona(self.persona_key)

    def set_persona(self, persona_key: str) -> None:
        self.persona_key = resolve_persona_key(persona_key)
        self.bot_name = get_persona(self.persona_key).display_name

    @classmethod
    def from_env(cls) -> "TelegramConfig":
        token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        allowed = _parse_chat_ids(os.environ.get("TELEGRAM_ALLOWED_CHAT_IDS", ""))
        persona_key = os.environ.get("PROTOAGI_TELEGRAM_PERSONA") or os.environ.get("NIKOLA_PERSONA", "")
        reply_mode = os.environ.get("NIKOLA_REPLY_MODE", "smart").strip() or "smart"
        if reply_mode not in {"smart", "always", "mention", "silent"}:
            reply_mode = "smart"
        sticker_frequency = os.environ.get("NIKOLA_STICKER_FREQUENCY", "normal").strip().lower() or "normal"
        if sticker_frequency not in {"off", "low", "normal", "high", "always"}:
            sticker_frequency = "normal"
        return cls(
            token=token,
            persona_key=persona_key,
            allowed_chat_ids=allowed,
            reply_mode=reply_mode,
            poll_timeout_seconds=env_int("TELEGRAM_POLL_TIMEOUT", 25),
            max_reply_messages=env_int("TELEGRAM_MAX_REPLY_MESSAGES", 3),
            sticker_frequency=sticker_frequency,
            sticker_cooldown_messages=env_int("NIKOLA_STICKER_COOLDOWN_MESSAGES", 3),
            fictional_self_enabled=env_bool("NIKOLA_FICTIONAL_SELF", True),
            global_memory=env_bool("PROTOAGI_TELEGRAM_GLOBAL_MEMORY", True),
            proactive_enabled=env_bool("NIKOLA_PROACTIVE", True),
            proactive_check_seconds=env_int("NIKOLA_PROACTIVE_CHECK_SECONDS", 300),
            proactive_cooldown_seconds=env_int("NIKOLA_PROACTIVE_COOLDOWN_SECONDS", 6 * 60 * 60),
            vision_base_url=os.environ.get("PROTOAGI_VISION_BASE_URL", "").strip(),
            vision_model=os.environ.get("PROTOAGI_VISION_MODEL", "").strip(),
            vision_max_bytes=env_int("PROTOAGI_VISION_MAX_BYTES", 8 * 1024 * 1024),
            vision_timeout_seconds=env_int("PROTOAGI_VISION_TIMEOUT_SECONDS", 120),
        )


def _parse_chat_ids(raw: str) -> set[str] | None:
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return ids or None


__all__ = ["TelegramConfig"]
