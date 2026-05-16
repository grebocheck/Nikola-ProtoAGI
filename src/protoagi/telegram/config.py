"""Telegram bot configuration loaded from env."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from ..env import env_bool, env_int
from ..persona import get_persona, resolve_persona_key
from ..web_search import WebSearchConfig
from .group_gate import GroupGateConfig, parse_trigger_keywords
from .reasoning_log import ReasoningLogConfig


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
    # Token budgets share the global ``PROTOAGI_MAX_TOKENS`` cap with the
    # deepseek reasoning channel. Keep them generous so a longer chain of
    # thought does not silently truncate the JSON decision before the
    # closing brace.
    decision_max_tokens: int = 2560
    reply_max_tokens: int = 2048
    reflection_max_tokens: int = 768
    llm_context_size: int = 8192
    prompt_context_max_chars: int = 6500
    max_reply_messages: int = 3
    # Default policy is "low": stickers feel like a punctuation mark, not a
    # filler. Cooldown is measured in user messages between stickers.
    # Replies longer than ``sticker_max_reply_chars`` are also kept text-only
    # because a sticker rarely fits a long human thought.
    sticker_frequency: str = "low"
    sticker_cooldown_messages: int = 6
    sticker_max_reply_chars: int = 180
    sticker_initiative_enabled: bool = False
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
    voice_base_url: str = ""
    voice_model: str = ""
    voice_max_bytes: int = 16 * 1024 * 1024
    voice_timeout_seconds: int = 120
    store_voice: bool = True
    tts_enabled: bool = False
    tts_base_url: str = ""
    tts_model: str = ""
    tts_voice: str = "alloy"
    tts_max_chars: int = 600
    tts_response_format: str = "opus"
    tts_speed: float = 1.0
    group_gate: GroupGateConfig = field(default_factory=GroupGateConfig)
    web_search: WebSearchConfig = field(default_factory=WebSearchConfig)
    reasoning_log: ReasoningLogConfig = field(default_factory=ReasoningLogConfig)

    def __post_init__(self) -> None:
        self.set_persona(self.persona_key)
        if self.llm_context_size < 8192:
            scaled = max(1200, int(self.llm_context_size * 0.75))
            self.prompt_context_max_chars = min(self.prompt_context_max_chars, scaled)

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
        sticker_frequency = os.environ.get("NIKOLA_STICKER_FREQUENCY", "low").strip().lower() or "low"
        if sticker_frequency not in {"off", "low", "normal", "high", "always"}:
            sticker_frequency = "low"
        return cls(
            token=token,
            persona_key=persona_key,
            allowed_chat_ids=allowed,
            reply_mode=reply_mode,
            poll_timeout_seconds=env_int("TELEGRAM_POLL_TIMEOUT", 25),
            max_reply_messages=env_int("TELEGRAM_MAX_REPLY_MESSAGES", 3),
            decision_max_tokens=env_int("PROTOAGI_DECISION_MAX_TOKENS", 2560),
            reply_max_tokens=env_int("PROTOAGI_REPLY_MAX_TOKENS", 2048),
            reflection_max_tokens=env_int("PROTOAGI_REFLECTION_MAX_TOKENS", 768),
            llm_context_size=env_int("PROTOAGI_CONTEXT_SIZE", 8192),
            prompt_context_max_chars=env_int("PROTOAGI_TELEGRAM_CONTEXT_MAX_CHARS", 6500),
            sticker_frequency=sticker_frequency,
            sticker_cooldown_messages=env_int("NIKOLA_STICKER_COOLDOWN_MESSAGES", 6),
            sticker_max_reply_chars=env_int("NIKOLA_STICKER_MAX_REPLY_CHARS", 180),
            sticker_initiative_enabled=env_bool("NIKOLA_STICKER_INITIATIVE", False),
            fictional_self_enabled=env_bool("NIKOLA_FICTIONAL_SELF", True),
            global_memory=env_bool("PROTOAGI_TELEGRAM_GLOBAL_MEMORY", True),
            proactive_enabled=env_bool("NIKOLA_PROACTIVE", True),
            proactive_check_seconds=env_int("NIKOLA_PROACTIVE_CHECK_SECONDS", 300),
            proactive_cooldown_seconds=env_int("NIKOLA_PROACTIVE_COOLDOWN_SECONDS", 6 * 60 * 60),
            vision_base_url=os.environ.get("PROTOAGI_VISION_BASE_URL", "").strip(),
            vision_model=os.environ.get("PROTOAGI_VISION_MODEL", "").strip(),
            vision_max_bytes=env_int("PROTOAGI_VISION_MAX_BYTES", 8 * 1024 * 1024),
            vision_timeout_seconds=env_int("PROTOAGI_VISION_TIMEOUT_SECONDS", 120),
            voice_base_url=os.environ.get("PROTOAGI_VOICE_BASE_URL", "").strip(),
            voice_model=os.environ.get("PROTOAGI_VOICE_MODEL", "").strip(),
            voice_max_bytes=env_int("PROTOAGI_VOICE_MAX_BYTES", 16 * 1024 * 1024),
            voice_timeout_seconds=env_int("PROTOAGI_VOICE_TIMEOUT_SECONDS", 120),
            store_voice=env_bool("PROTOAGI_STORE_VOICE", True),
            tts_enabled=env_bool("PROTOAGI_TTS_ENABLED", False),
            tts_base_url=os.environ.get("PROTOAGI_TTS_BASE_URL", "").strip(),
            tts_model=os.environ.get("PROTOAGI_TTS_MODEL", "").strip(),
            tts_voice=os.environ.get("PROTOAGI_TTS_VOICE", "alloy").strip() or "alloy",
            tts_max_chars=env_int("PROTOAGI_TTS_MAX_CHARS", 600),
            tts_response_format=os.environ.get("PROTOAGI_TTS_RESPONSE_FORMAT", "opus").strip().lower() or "opus",
            tts_speed=_env_float("PROTOAGI_TTS_SPEED", 1.0),
            group_gate=_load_group_gate_config(),
            web_search=_load_web_search_config(),
            reasoning_log=_load_reasoning_log_config(),
        )


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _parse_chat_ids(raw: str) -> set[str] | None:
    ids = {part.strip() for part in raw.split(",") if part.strip()}
    return ids or None


def _load_reasoning_log_config() -> ReasoningLogConfig:
    return ReasoningLogConfig(
        enabled=env_bool("PROTOAGI_CAPTURE_REASONING", False),
        max_entries_per_chat=env_int("PROTOAGI_REASONING_MAX_ENTRIES", 20),
        max_chars_per_entry=env_int("PROTOAGI_REASONING_MAX_CHARS", 3000),
    )


def _load_web_search_config() -> WebSearchConfig:
    return WebSearchConfig(
        base_url=os.environ.get("PROTOAGI_WEB_SEARCH_URL", "").strip(),
        timeout_seconds=env_int("PROTOAGI_WEB_SEARCH_TIMEOUT_SECONDS", 10),
        max_results=env_int("PROTOAGI_WEB_SEARCH_MAX_RESULTS", 5),
        cache_seconds=env_int("PROTOAGI_WEB_SEARCH_CACHE_SECONDS", 900),
    )


def _load_group_gate_config() -> GroupGateConfig:
    raw_ratio = os.environ.get("NIKOLA_GROUP_PASSIVE_REPLY_RATIO", "")
    try:
        ratio = float(raw_ratio) if raw_ratio.strip() else 0.04
    except ValueError:
        ratio = 0.04
    if ratio < 0.0:
        ratio = 0.0
    if ratio > 1.0:
        ratio = 1.0
    return GroupGateConfig(
        cooldown_seconds=env_int("NIKOLA_GROUP_REPLY_COOLDOWN_SECONDS", 120),
        passive_reply_ratio=ratio,
        trigger_keywords=parse_trigger_keywords(
            os.environ.get("NIKOLA_GROUP_TRIGGER_KEYWORDS", "")
        ),
    )


__all__ = ["TelegramConfig"]
