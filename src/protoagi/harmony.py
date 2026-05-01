from __future__ import annotations

import re


FINAL_RE = re.compile(
    r"<\|channel\|>final<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
COMMENTARY_RE = re.compile(
    r"<\|channel\|>commentary<\|message\|>(.*?)(?=<\|end\|>|<\|start\|>|$)",
    re.DOTALL,
)
ANALYSIS_RE = re.compile(
    r"<\|channel\|>analysis<\|message\|>.*?(?=<\|channel\|>(?:final|commentary)<\|message\|>|<\|end\|>|$)",
    re.DOTALL,
)
TOKEN_RE = re.compile(
    r"<\|(?:start|end|channel|message|return|call|endoftext|startoftext)\|>|"
    r"<\|channel\|>\w+<\|message\|>",
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


def _strip_tokens(text: str) -> str:
    return TOKEN_RE.sub("", text).replace("\u202f", " ").replace("\u2011", "-")
