"""Plain text helpers for the Telegram pipeline.

These helpers carry no state of their own; they are pulled out of the
monolithic bot module to keep the orchestrator small and the regex/scrubbing
logic independently testable.
"""

from __future__ import annotations

import re
from typing import Any


GENERIC_STICKER_FILLER_RE = re.compile(
    r"(ось\s*(ще)?\s*один|сподіва(юся|юсь)|підня(в|ло|ти).*настр|настрій)",
    re.IGNORECASE,
)
ASSISTANTY_SENTENCE_RE = re.compile(
    r"(^|(?<=[.!?…])\s*)[^.!?…]*(готов[аийі]*\s+допомогти|чим\s+можу\s+допомогти|якщо\s+треба)[^.!?…]*[.!?…]?",
    re.IGNORECASE,
)
GENERIC_CHECKIN_RE = re.compile(
    r"(^|(?<=[.!?…])\s*)"
    r"(як\s+(твої\s+)?справи\s*(сьогодні)?\??|як\s+пройшов\s+твій\s+день\??|чи\s+щось\s+цікаве\s+сталося\??)"
    r"\s*",
    re.IGNORECASE,
)


def parse_command(text: str, bot_username: str = "") -> tuple[str | None, str]:
    if not text.startswith("/"):
        return None, ""
    first, _, rest = text.partition(" ")
    command = first[1:]
    if "@" in command:
        name, _, target = command.partition("@")
        if bot_username and target.lower() != bot_username.lower():
            return None, ""
        command = name
    return command.lower(), rest.strip()


def strip_speaker_prefixes(text: str, speaker_names: list[str]) -> str:
    cleaned = str(text or "").strip()
    names = [name.strip() for name in speaker_names if name and name.strip()]
    if not names:
        return cleaned
    pattern = re.compile(
        r"^\s*(?:" + "|".join(re.escape(name) for name in sorted(set(names), key=len, reverse=True)) + r")\s*[:：]\s*",
        re.IGNORECASE,
    )
    for _ in range(4):
        updated = pattern.sub("", cleaned, count=1).strip()
        if updated == cleaned:
            break
        cleaned = updated
    return cleaned


def strip_assistanty_phrases(text: str) -> str:
    cleaned = str(text or "").strip()
    cleaned = ASSISTANTY_SENTENCE_RE.sub(" ", cleaned)
    cleaned = GENERIC_CHECKIN_RE.sub(" ", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    cleaned = re.sub(r"\s+([,.!?…])", r"\1", cleaned)
    return cleaned


def split_telegram_message(text: str, *, max_chars: int = 3900) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks: list[str] = []
    while len(text) > max_chars:
        split_at = text.rfind("\n", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = text.rfind(" ", 0, max_chars)
        if split_at < max_chars // 2:
            split_at = max_chars
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


def display_sender(user: dict[str, Any]) -> str:
    first = str(user.get("first_name") or "")
    last = str(user.get("last_name") or "")
    username = str(user.get("username") or "")
    name = " ".join(part for part in (first, last) if part).strip()
    if name:
        return name
    if username:
        return f"@{username}"
    return str(user.get("id", "someone"))


__all__ = [
    "ASSISTANTY_SENTENCE_RE",
    "GENERIC_CHECKIN_RE",
    "GENERIC_STICKER_FILLER_RE",
    "display_sender",
    "parse_command",
    "split_telegram_message",
    "strip_assistanty_phrases",
    "strip_speaker_prefixes",
]
