"""Identity and image-blindness guards.

Two related safety concerns:

1. The bot must not pretend to be human when asked directly. The regex below
   triggers a deterministic honest reply that overrides anything the model
   produced.
2. When a vision description is unavailable the model sometimes blurts out
   "I can't see the image"; we strip those replies so the bot either reacts
   to the caption or stays silent.
"""

from __future__ import annotations

import re

from ..persona import PersonaProfile


IDENTITY_QUESTION_RE = re.compile(
    r"(ти\s+(реально\s+)?(людина|бот|нейронк|штучн|ai|іі|жив[аий])|"
    r"ти\s+ж\s+(людина|бот)|"
    r"are\s+you\s+(really\s+)?(a\s+)?(human|bot|ai)|"
    r"are\s+you\s+real)",
    re.IGNORECASE,
)
DECEPTIVE_IDENTITY_REPLY_RE = re.compile(
    r"(^\s*(так|ага|звісно|авжеж|yes|yeah|sure)\b.{0,120}\b(людин|human|жив[аий])|"
    r"^\s*(ні|no)\b.{0,120}\b(не\s+бот|not\s+a\s+bot))",
    re.IGNORECASE | re.DOTALL,
)
IMAGE_BLIND_REPLY_RE = re.compile(
    r"(не\s+бач[уишить]*\s+(це\s+)?зображення|не\s+можу\s+(побачити|роздивитись)|"
    r"can't\s+see\s+(the\s+)?image|cannot\s+see\s+(the\s+)?image|don't\s+see\s+(the\s+)?image)",
    re.IGNORECASE,
)


def is_identity_question(text: str) -> bool:
    return bool(IDENTITY_QUESTION_RE.search(str(text or "")))


def is_deceptive_identity_reply(text: str) -> bool:
    return bool(DECEPTIVE_IDENTITY_REPLY_RE.search(str(text or "")))


def is_image_blind_reply(text: str) -> bool:
    return bool(IMAGE_BLIND_REPLY_RE.search(str(text or "")))


def honest_identity_reply(persona: PersonaProfile) -> str:
    if persona.key == "solomiya":
        return "Не фізична людина, ні. Але в цьому чаті я Соломія: з характером, памʼяттю і своїми маленькими звичками."
    return f"Не фізична людина. Я {persona.display_name} у цьому чаті: локальний співрозмовник з памʼяттю і своїм стилем."


__all__ = [
    "DECEPTIVE_IDENTITY_REPLY_RE",
    "IDENTITY_QUESTION_RE",
    "IMAGE_BLIND_REPLY_RE",
    "honest_identity_reply",
    "is_deceptive_identity_reply",
    "is_identity_question",
    "is_image_blind_reply",
]
