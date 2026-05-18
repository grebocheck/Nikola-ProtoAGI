"""Re-run the sticker describer on rows that previously failed.

Useful after a fix to the validator or the vision/translation token
budgets: stickers that landed in ``failure_reason`` with messages like
``vision caption rejected: caption appears truncated`` or
``translation returned empty`` can be retried in-place instead of
waiting for the next ambient discovery pass.

Usage examples:

    # Dry run — only report what would be retried, no API calls.
    python scripts/redescribe_failed_stickers.py --dry-run

    # Retry every truncation-related failure, then every empty-translation
    # failure, up to 50 stickers total.
    python scripts/redescribe_failed_stickers.py --limit 50

    # Target specific sticker_ids (from the operator log).
    python scripts/redescribe_failed_stickers.py --sticker-id CAACAg... --sticker-id CAACAg...

The script reuses ``build_nikola_bot`` so it picks up the exact LLM,
vision endpoint, persona, and DB the live bot is configured with.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from protoagi.config import AgentConfig
from protoagi.telegram.config import TelegramConfig
from protoagi.telegram.orchestrator import build_nikola_bot
from protoagi.telegram.sticker_describer import StickerDescriberWorker


DEFAULT_FAILURE_PATTERNS = (
    "%caption appears truncated%",
    "%translation returned empty%",
)


def _find_matching_sticker_ids(
    memory,
    *,
    patterns: tuple[str, ...],
    explicit_ids: tuple[str, ...],
    limit: int,
) -> list[str]:
    if explicit_ids:
        return list(dict.fromkeys(explicit_ids))[:limit]
    matched: list[str] = []
    with memory.connect() as conn:
        for pattern in patterns:
            rows = conn.execute(
                """
                SELECT sticker_id
                FROM sticker_descriptions
                WHERE failure_reason LIKE ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (pattern, limit),
            ).fetchall()
            for row in rows:
                sticker_id = str(row["sticker_id"])
                if sticker_id and sticker_id not in matched:
                    matched.append(sticker_id)
                if len(matched) >= limit:
                    return matched
    return matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sticker-id",
        action="append",
        default=[],
        help="Retry a specific sticker_id; can be passed multiple times.",
    )
    parser.add_argument(
        "--pattern",
        action="append",
        default=[],
        help=(
            "Extra SQL LIKE pattern to match against failure_reason. "
            "Defaults to truncation and empty-translation failures."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of stickers to retry in this pass.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching stickers and exit without resetting or describing.",
    )
    args = parser.parse_args()

    patterns = tuple(args.pattern) if args.pattern else DEFAULT_FAILURE_PATTERNS
    explicit_ids = tuple(args.sticker_id)

    agent_config = AgentConfig.load()
    telegram_config = TelegramConfig.from_env()
    if not telegram_config.token:
        print(
            "TELEGRAM_BOT_TOKEN is not set — needed for getFile/downloadFile.",
            file=sys.stderr,
        )
        return 2

    bot = build_nikola_bot(agent_config=agent_config, telegram_config=telegram_config)
    # Skip bootstrap()'s webhook calls; we don't want to touch live polling.
    memory = bot.memory

    matched = _find_matching_sticker_ids(
        memory,
        patterns=patterns,
        explicit_ids=explicit_ids,
        limit=max(1, args.limit),
    )
    if not matched:
        print("No matching failed stickers found.")
        return 0
    print(f"Matched {len(matched)} sticker(s) for retry.")
    for sticker_id in matched[:10]:
        row = memory.get_sticker_description(sticker_id)
        reason = (row.failure_reason if row else "") or "<missing>"
        print(f"  - {sticker_id}: {reason}")
    if len(matched) > 10:
        print(f"  ... and {len(matched) - 10} more")

    if args.dry_run:
        return 0

    reset = memory.reset_sticker_describer_attempts(
        sticker_ids=matched,
        clear_descriptions=True,
    )
    print(f"Reset {reset} sticker row(s); running describer pass...")

    if not bot._vision.enabled:
        print("Vision LLM is not configured — cannot regenerate captions.", file=sys.stderr)
        return 2
    worker = StickerDescriberWorker(
        telegram=bot.telegram,
        vision=bot._vision,
        memory=bot.memory,
        chat_llm=bot.llm,
        embedding_client=bot.memory_service.embedding_client,
    )
    described = worker.describe_pending()
    print(f"describe_pending() processed {described} sticker(s).")

    # Report fresh state for the matched ids.
    succeeded = 0
    failed = 0
    for sticker_id in matched:
        row = memory.get_sticker_description(sticker_id)
        if row is None:
            continue
        if row.description and not row.failure_reason:
            succeeded += 1
        else:
            failed += 1
    print(f"Outcome for matched batch: succeeded={succeeded}, still_failing={failed}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
