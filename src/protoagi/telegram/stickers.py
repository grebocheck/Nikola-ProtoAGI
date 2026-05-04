"""Sticker pack metadata and selection helpers."""

from __future__ import annotations

import re


STICKER_PACKS = {
    "Bocchi_the_Rock_sticker_pack2": "expressive awkward, shy, funny, surprised anime reactions",
    "SenkoSan": "warm, caring, cozy, gentle reactions",
    "M1ku_Hatsune": "playful, energetic, music-like, cheerful reactions",
}
STICKER_ALIASES = {
    "bocchi": "Bocchi_the_Rock_sticker_pack2",
    "bocchi_the_rock": "Bocchi_the_Rock_sticker_pack2",
    "senko": "SenkoSan",
    "senkosan": "SenkoSan",
    "miku": "M1ku_Hatsune",
    "hatsune": "M1ku_Hatsune",
    "m1ku": "M1ku_Hatsune",
}


SERIOUS_STICKER_RE = re.compile(
    r"(помер|смерт|війна|тривог|ракета|обстр|болить|лікар|депрес|панік|суїцид|"
    r"зле|погано|страшно|ненавид|розлуч|плач|сльоз|хвор)",
    re.IGNORECASE,
)


def normalize_sticker_pack(value: str) -> str | None:
    value = (value or "").strip()
    if value in STICKER_PACKS:
        return value
    return STICKER_ALIASES.get(value.lower())


def looks_serious_for_sticker(text: str) -> bool:
    return bool(SERIOUS_STICKER_RE.search(str(text or "")))


# NOTE: Cyrillic «а» (U+0430) and Latin «a» (U+0061) are *different*
# characters. The class below intentionally lists each script separately;
# mixing them in one ``[…]`` silently broke the previous version, which is
# how the bot ended up with no auto-reaction on Ukrainian laughter.
_LAUGH_RE = re.compile(
    r"(а[ха]{2,}|ха[ха]{2,}|hah[ah]+|lol+|оруу+|жиза|🤣|😂|сміш(но|нюк))",
    re.IGNORECASE,
)
# Warm-pack triggers must look like a short, emotionally direct message:
# bare "Дякую за пораду, я завтра спробую цей підхід" should not get a
# sticker, but a one-line "ой, дякую тобі ❤️" should.
_WARM_RE = re.compile(
    r"(♥|❤|🤗|обійм[ауи]+|обніма[юй]|дякую тобі|щиро дякую|спокійної ночі|чмоки)",
    re.IGNORECASE,
)
_GAME_RE = re.compile(
    r"(🎮|купив (геймпад|контролер)|новий геймпад|playstation|xbox|steam деке)",
    re.IGNORECASE,
)


def auto_sticker_choice(
    incoming_text: str,
    reply_text: str = "",
    *,
    max_reply_chars: int = 180,
) -> dict[str, str] | None:
    """Pick a sticker reaction only when the trigger is unambiguous.

    The previous version fired on any single hint word ("чай", "грати",
    "ахах"), which made the bot over-stickerize. The current rules require
    one of:

    - explicit laughter (≥3 repeated chars, "lol", "🤣", "😂");
    - explicit warmth (heart emoji, "обійми/обнімаю", "дякую тобі");
    - explicit gameplay context (controller emoji or specific verbs).

    A long reply (above ``max_reply_chars`` after stripping) also returns
    ``None`` — stickers don't pair with a paragraph-shaped human thought.
    """

    reply_clean = reply_text.strip()
    if len(reply_clean) > max_reply_chars:
        return None
    incoming = incoming_text.strip()
    text = f"{incoming}\n{reply_clean}"
    if _GAME_RE.search(text):
        return {"pack": "M1ku_Hatsune", "emoji": "✨", "reason": "playful game-chat reaction"}
    if _LAUGH_RE.search(text):
        return {"pack": "Bocchi_the_Rock_sticker_pack2", "emoji": "🙂", "reason": "light funny reaction"}
    if _WARM_RE.search(text):
        return {"pack": "SenkoSan", "emoji": "🙂", "reason": "warm reaction"}
    return None


__all__ = [
    "SERIOUS_STICKER_RE",
    "STICKER_ALIASES",
    "STICKER_PACKS",
    "auto_sticker_choice",
    "looks_serious_for_sticker",
    "normalize_sticker_pack",
]
