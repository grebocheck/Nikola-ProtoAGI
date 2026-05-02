from __future__ import annotations

import re


CHANNEL_PREFIX = r"<\|channel\|>{channel}(?:\s*<\|constrain\|>\w+)?\s*<\|message\|>"
FINAL_RE = re.compile(
    CHANNEL_PREFIX.format(channel="final") + r"(.*?)(?=<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
COMMENTARY_RE = re.compile(
    CHANNEL_PREFIX.format(channel="commentary") + r"(.*?)(?=<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
ANALYSIS_RE = re.compile(
    CHANNEL_PREFIX.format(channel="analysis")
    + r".*?(?=<\|channel\|>(?:final|commentary)(?:\s*<\|constrain\|>\w+)?\s*<\|message\|>|<\|end\|>|$)",
    re.DOTALL,
)
TOKEN_RE = re.compile(
    r"<\|channel\|>\w+(?:\s*<\|constrain\|>\w+)?\s*<\|message\|>|"
    r"<\|[A-Za-z0-9_:-]{1,64}\|>",
    re.DOTALL,
)


def clean_model_content(content: str) -> str:
    """Remove raw Harmony channel markup from user-facing text.

    llama-server can intentionally expose reasoning in `message.content` when
    started with `--reasoning-format none`. Runtime defaults should avoid that,
    but the client still sanitizes final text as a defensive layer.
    """
    if not content:
        return ""
    final_matches = FINAL_RE.findall(content)
    if final_matches:
        return _strip_tokens(final_matches[-1]).strip()
    commentary_matches = COMMENTARY_RE.findall(content)
    if commentary_matches:
        return _strip_tokens(commentary_matches[-1]).strip()
    without_analysis = ANALYSIS_RE.sub("", content)
    return _strip_tokens(without_analysis).strip()


def sanitize_model_input(content: str) -> str:
    """Remove Harmony control tokens before text is sent back to llama-server."""
    if not content:
        return ""
    return _strip_tokens(content)


def _strip_tokens(text: str) -> str:
    return TOKEN_RE.sub("", text).replace("\u202f", " ").replace("\u2011", "-")
