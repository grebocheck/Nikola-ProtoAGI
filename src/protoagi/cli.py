"""ProtoAGI CLI — minimal surface.

Only two commands are exposed:

- ``protoagi telegram`` — run the Telegram conversation bot.
- ``protoagi admin``    — run the local admin web UI.

Everything else (bench / chat / serve / memory-* / backup / federation /
init-config) was retired together with the experimental loops that
produced them. Use the admin UI for memory introspection and the
PowerShell scripts under ``scripts/`` for server lifecycle.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .admin_panel.server import serve as serve_admin
from .config import AgentConfig
from .openai_compat import OpenAICompatError
from .storage.memory import MemoryStore
from .storage.service import MemoryService
from .telegram import (
    AsyncBotRunner,
    BotRunner,
    TelegramApiError,
    TelegramConfig,
    build_nikola_bot,
    is_telegram_polling_conflict,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protoagi")
    sub = parser.add_subparsers(dest="command", required=True)

    telegram = sub.add_parser("telegram", help="Run the Telegram conversation bot")
    telegram.add_argument("--token", default=None, help="Telegram bot token, defaults to TELEGRAM_BOT_TOKEN")
    telegram.add_argument("--allowed-chat-id", action="append", default=[])
    telegram.add_argument("--reply-mode", choices=["smart", "always", "mention", "silent"], default=None)
    telegram.add_argument("--base-url", default=None)
    telegram.add_argument("--model", default=None)
    telegram.add_argument("--db", default=None, help="SQLite database path for this Telegram instance.")
    telegram.add_argument("--persona", default=None, help="Persona key, e.g. mykola or solomiya.")
    telegram.add_argument("--poll-timeout", type=int, default=None)
    telegram.add_argument("--once", action="store_true", help="Process one polling batch and exit")
    telegram.add_argument("--no-proactive", action="store_true")
    telegram.add_argument("--proactive-check-seconds", type=int, default=None)
    telegram.add_argument("--proactive-cooldown-seconds", type=int, default=None)
    telegram.add_argument("--delete-webhook", action="store_true")
    telegram.add_argument("--drop-pending-updates", action="store_true")
    telegram.add_argument(
        "--single-thread",
        action="store_true",
        help="Run the legacy single-thread loop (no background reminder/reflection worker).",
    )
    telegram.add_argument(
        "--async",
        dest="async_runner",
        action="store_true",
        help="Run the asyncio polling supervisor with concurrent update handling.",
    )
    telegram.add_argument("--max-concurrent-updates", type=int, default=2)

    admin = sub.add_parser("admin", help="Run the local admin web UI.")
    admin.add_argument("--db", default=None)
    admin.add_argument("--host", default="127.0.0.1")
    admin.add_argument("--port", type=int, default=8765)

    return parser


def cmd_telegram(args: argparse.Namespace) -> int:
    agent_config = AgentConfig.load().with_cli_overrides(
        base_url=args.base_url,
        model=args.model,
        database_path=args.db,
    )
    telegram_config = TelegramConfig.from_env()
    if args.persona:
        telegram_config.set_persona(args.persona)
    if args.token:
        telegram_config.token = args.token
    if args.allowed_chat_id:
        telegram_config.allowed_chat_ids = {str(value) for value in args.allowed_chat_id}
    if args.reply_mode:
        telegram_config.reply_mode = args.reply_mode
    if args.poll_timeout is not None:
        telegram_config.poll_timeout_seconds = args.poll_timeout
    if args.no_proactive:
        telegram_config.proactive_enabled = False
    if args.proactive_check_seconds is not None:
        telegram_config.proactive_check_seconds = args.proactive_check_seconds
    if args.proactive_cooldown_seconds is not None:
        telegram_config.proactive_cooldown_seconds = args.proactive_cooldown_seconds
    if not telegram_config.token:
        print(
            "TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and set the token.",
            file=sys.stderr,
        )
        return 2

    bot = build_nikola_bot(agent_config=agent_config, telegram_config=telegram_config)
    me = bot.bootstrap(
        delete_webhook=args.delete_webhook,
        drop_pending_updates=args.drop_pending_updates,
    )
    print(
        f"{telegram_config.bot_name} online as @{me.get('username', 'unknown')} | "
        f"persona={telegram_config.persona_key} | reply_mode={telegram_config.reply_mode} | "
        f"proactive={telegram_config.proactive_enabled} | db={agent_config.database_path}"
    )
    if args.once:
        processed = bot.poll_once()
        proactive = bot.run_initiative_once() if telegram_config.proactive_enabled else 0
        delivered = bot.dispatch_due_reminders()
        print(
            f"Processed updates: {processed}; proactive messages: {proactive}; "
            f"reminders delivered: {delivered}"
        )
        return 0
    if args.single_thread:
        bot.run_forever()
        return 0
    if args.async_runner:
        import asyncio

        runner = AsyncBotRunner(
            bot,
            max_concurrent_updates=args.max_concurrent_updates,
        )
        asyncio.run(runner.run())
        return 0
    runner = BotRunner(bot)
    runner.run()
    return 0


def cmd_admin(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args)
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 2
    store = MemoryStore(db_path)
    service = MemoryService(store=store)
    server = serve_admin(store, service, host=args.host, port=args.port)
    url = f"http://{args.host}:{args.port}"
    print(f"ProtoAGI admin listening on {url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        server.server_close()
    return 0


def _resolve_db_path(args: argparse.Namespace) -> Path:
    config = AgentConfig.load()
    raw = getattr(args, "db", None)
    if raw:
        return Path(raw).resolve()
    return config.database_path


def main(argv: list[str] | None = None) -> int:
    _prefer_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "telegram":
            return cmd_telegram(args)
        if args.command == "admin":
            return cmd_admin(args)
    except OpenAICompatError as exc:
        print(f"OpenAI-compatible endpoint error: {exc}", file=sys.stderr)
        return 2
    except TelegramApiError as exc:
        if is_telegram_polling_conflict(exc):
            print(
                "Telegram polling conflict: another Telegram bot instance is "
                "already running for this token.",
                file=sys.stderr,
            )
            print("Stop the other instance before starting a new one.", file=sys.stderr)
        else:
            print(f"Telegram API error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Missing file: {exc}", file=sys.stderr)
        return 2
    return 1


def _prefer_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except (TypeError, ValueError, OSError):
                pass


if __name__ == "__main__":
    raise SystemExit(main())
