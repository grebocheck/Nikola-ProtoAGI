"""Allowed-emoji whitelist and KV prefixes for bot-side reactions.

Telegram limits which emoji a non-premium bot may attach via
``setMessageReaction`` to a chat's free reaction set. The official list is
not exposed for every chat (``getAvailableReactions`` would help but adds an
extra round-trip per turn), so we hardcode the well-known free-tier set and
fall back to a per-chat denylist that grows whenever the API rejects an emoji
with ``REACTION_INVALID``.
"""

from __future__ import annotations


REACTION_COOLDOWN_KV_PREFIX = "telegram:reaction_cooldown:"
REACTION_DENYLIST_KV_PREFIX = "telegram:reaction_denylist:"
REACTION_SENT_COUNT_KV_PREFIX = "telegram:reaction_sent_count:"


# Free-tier emoji that any bot can use in regular chats. Conservative list
# of single-codepoint and well-known compound emoji; the per-chat denylist
# captures anything Telegram rejects at runtime so we self-heal.
ALLOWED_REACTION_EMOJI: frozenset[str] = frozenset(
    [
        "👍", "👎", "❤", "❤️", "🔥", "🥰", "👏", "😁", "🤔", "🤯",
        "😱", "🤬", "😢", "🎉", "🤩", "🤮", "💩", "🙏", "👌", "🕊",
        "🤡", "🥱", "🥴", "😍", "🐳", "🌚", "🌭", "💯", "🤣", "⚡",
        "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈",
        "😴", "😭", "🤓", "👻", "👀", "🎃", "🙈", "😇", "😨", "🤝",
        "✍", "✍️", "🤗", "🫡", "🎅", "🎄", "☃", "☃️", "💅", "🤪",
        "🗿", "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾",
        "🤷", "😡",
    ]
)


def normalize_reaction_emoji(value: object) -> str:
    """Strip whitespace, return empty string for anything non-textual."""

    if not isinstance(value, str):
        return ""
    return value.strip()


def filter_allowed_emoji(emoji: str, denylist: frozenset[str] | set[str]) -> str | None:
    """Return ``emoji`` if it is in the static whitelist and not denylisted.

    The whitelist is conservative (avoids premium-only emoji), the denylist
    captures any emoji that Telegram has refused for this chat at runtime.
    """

    cleaned = normalize_reaction_emoji(emoji)
    if not cleaned:
        return None
    if cleaned in denylist:
        return None
    if cleaned not in ALLOWED_REACTION_EMOJI:
        return None
    return cleaned


def parse_denylist(raw: str | None) -> set[str]:
    if not raw:
        return set()
    return {item for item in raw.split("\n") if item}


def serialize_denylist(items: set[str]) -> str:
    return "\n".join(sorted(items))


__all__ = [
    "ALLOWED_REACTION_EMOJI",
    "REACTION_COOLDOWN_KV_PREFIX",
    "REACTION_DENYLIST_KV_PREFIX",
    "REACTION_SENT_COUNT_KV_PREFIX",
    "filter_allowed_emoji",
    "normalize_reaction_emoji",
    "parse_denylist",
    "serialize_denylist",
]
