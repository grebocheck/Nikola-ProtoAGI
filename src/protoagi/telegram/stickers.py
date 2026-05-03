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


def auto_sticker_choice(incoming_text: str, reply_text: str = "") -> dict[str, str] | None:
    text = f"{incoming_text}\n{reply_text}".lower()
    if "🎮" in text or re.search(r"(гейм|контрол|джой|гра|пад|xbox|playstation|steam)", text, re.IGNORECASE):
        return {"pack": "M1ku_Hatsune", "emoji": "✨", "reason": "playful game-chat reaction"}
    if re.search(r"(ахах|хаха|лол|оруу|бє+|жиза|сміш)", text, re.IGNORECASE):
        return {"pack": "Bocchi_the_Rock_sticker_pack2", "emoji": "🙂", "reason": "light funny reaction"}
    if re.search(r"(дякую|мил|обій|чай|кава|спок|сон|приєм)", text, re.IGNORECASE):
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
