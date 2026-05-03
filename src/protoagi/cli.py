from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .admin import serve as serve_admin
from .agent import ProtoAgent
from .backup import BackupError, backup_database, default_backup_path, restore_database
from .bench import bench_endpoint, endpoint_results_to_json, run_llama_bench
from .config import AgentConfig, DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH, LlamaServerProfile, PROJECT_ROOT
from .embedding import EmbeddingClient, EmbeddingConfig
from .memory import MemoryStore
from .memory_eval import DEFAULT_CORPUS_PATH, run_eval
from .memory_service import MemoryService
from .openai_compat import OpenAICompatibleClient, OpenAICompatError
from .runtime import run_server_foreground, status_report
from .telegram_bot import AsyncBotRunner, BotRunner, TelegramApiError, TelegramConfig, build_nikola_bot, is_telegram_polling_conflict
from .tools import default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="protoagi", description="Local ProtoAGI experiment CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show runtime and server status")
    status.add_argument("--base-url", default=None)

    serve = sub.add_parser("serve", help="Run llama-server in the foreground")
    add_server_args(serve)

    chat = sub.add_parser("chat", help="Run one prompt or an interactive agent chat")
    chat.add_argument("--prompt", "-p", default=None)
    chat.add_argument("--thread-id", default=None)
    chat.add_argument("--max-steps", type=int, default=8)
    chat.add_argument("--base-url", default=None)
    chat.add_argument("--model", default=None)
    chat.add_argument("--allow-write", action="store_true")
    chat.add_argument("--deny-write", action="store_true")
    chat.add_argument("--allow-shell", action="store_true")
    chat.add_argument("--allow-unsafe-shell", action="store_true")
    chat.add_argument("--max-tokens", type=int, default=None)
    chat.add_argument("--temperature", type=float, default=None)
    chat.add_argument("--top-p", type=float, default=None)
    chat.add_argument(
        "--stream",
        action="store_true",
        help="Stream a one-off plain reply (no tools). Useful for quick sanity checks.",
    )

    bench = sub.add_parser("bench", help="Benchmark the running OpenAI-compatible endpoint")
    bench.add_argument("--prompt", default="In three bullet points, explain what makes a good local agent.")
    bench.add_argument("--rounds", type=int, default=3)
    bench.add_argument("--max-tokens", type=int, default=256)
    bench.add_argument("--base-url", default=None)
    bench.add_argument("--model", default=None)

    llama_bench = sub.add_parser("bench-llama", help="Run llama-bench against the local GGUF")
    add_server_args(llama_bench)
    llama_bench.add_argument("--output", default="runs/llama-bench.jsonl")
    llama_bench.add_argument("--n-cpu-moe", default="0,4,8,12")
    llama_bench.add_argument("--prompt-tokens", type=int, default=512)
    llama_bench.add_argument("--gen-tokens", type=int, default=128)
    llama_bench.add_argument("--repetitions", type=int, default=2)

    init = sub.add_parser("init-config", help="Create config/protoagi.json from defaults")
    init.add_argument("--force", action="store_true")

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

    memory_eval = sub.add_parser(
        "memory-eval",
        help="Run the memory recall benchmark against a fact corpus.",
    )
    memory_eval.add_argument("--corpus", default=str(DEFAULT_CORPUS_PATH))
    memory_eval.add_argument("--k", default="1,3,5")
    memory_eval.add_argument("--with-embeddings", action="store_true")
    memory_eval.add_argument("--json", action="store_true", help="Print full JSON report.")

    memory_stats = sub.add_parser("memory-stats", help="Print memory store counters.")
    memory_stats.add_argument("--db", default=None)

    memory_prune = sub.add_parser("memory-prune", help="Forget low-value memory items.")
    memory_prune.add_argument("--db", default=None)
    memory_prune.add_argument("--scope", default=None)
    memory_prune.add_argument("--persona", default=None)
    memory_prune.add_argument("--score-threshold", type=float, default=0.12)
    memory_prune.add_argument("--keep-newer-than-days", type=float, default=30.0)
    memory_prune.add_argument("--dry-run", action="store_true")
    memory_prune.add_argument("--json", action="store_true", help="Print full JSON including dry-run plan.")

    memory_consolidate = sub.add_parser(
        "memory-consolidate",
        help="Run the consolidation pass to supersede near-duplicate memories.",
    )
    memory_consolidate.add_argument("--db", default=None)
    memory_consolidate.add_argument("--scope", default=None)
    memory_consolidate.add_argument("--persona", default=None)
    memory_consolidate.add_argument("--dry-run", action="store_true")
    memory_consolidate.add_argument("--json", action="store_true", help="Print full JSON including dry-run plan.")

    backup = sub.add_parser("backup", help="Create an online SQLite backup.")
    backup.add_argument("--db", default=None)
    backup.add_argument("--to", default=None, help="Target .sqlite3 path; defaults to data/backups/<timestamp>.sqlite3")

    restore = sub.add_parser("restore", help="Validate and restore a SQLite backup.")
    restore.add_argument("--db", default=None)
    restore.add_argument("--from", dest="from_path", required=True, help="Backup .sqlite3 path to restore from.")

    admin = sub.add_parser("admin", help="Run the local admin web UI.")
    admin.add_argument("--db", default=None)
    admin.add_argument("--host", default="127.0.0.1")
    admin.add_argument("--port", type=int, default=8765)

    return parser


def add_server_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--llama-dir", default=str(PROJECT_ROOT / "tools" / "llama.cpp"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--ctx-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--ubatch-size", type=int, default=1024)
    parser.add_argument("--cpu-moe", type=int, default=4)
    parser.add_argument("--full-gpu", action="store_true")
    parser.add_argument("--flash-attn", default="on", choices=["on", "off", "auto"])
    parser.add_argument("--no-jinja", action="store_true")


def profile_from_args(args: argparse.Namespace) -> LlamaServerProfile:
    return LlamaServerProfile(
        model_path=Path(args.model_path).resolve(),
        llama_dir=Path(args.llama_dir).resolve(),
        host=args.host,
        port=args.port,
        ctx_size=args.ctx_size,
        batch_size=args.batch_size,
        ubatch_size=args.ubatch_size,
        n_cpu_moe=None if args.full_gpu else args.cpu_moe,
        flash_attn=args.flash_attn,
        jinja=not args.no_jinja,
    )


def make_agent(args: argparse.Namespace) -> ProtoAgent:
    config = AgentConfig.load().with_cli_overrides(
        base_url=getattr(args, "base_url", None),
        model=getattr(args, "model", None),
        allow_write=False if getattr(args, "deny_write", False) else (True if getattr(args, "allow_write", False) else None),
        allow_shell=True if getattr(args, "allow_shell", False) else None,
        allow_unsafe_shell=True if getattr(args, "allow_unsafe_shell", False) else None,
        max_tokens=getattr(args, "max_tokens", None),
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
    )
    memory = MemoryStore(config.database_path)
    tools = default_registry(memory, config.tool_policy)
    client = OpenAICompatibleClient(config.base_url, config.model)
    embedding_config = EmbeddingConfig(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
        timeout_seconds=config.embedding.timeout_seconds,
        request_dimensions=config.embedding.request_dimensions,
    )
    embedding_client = EmbeddingClient(embedding_config) if embedding_config.enabled else None
    memory_service = MemoryService(
        memory,
        embedding_client=embedding_client,
        embedding_backend=config.embedding.backend,
        importance_client=client if config.llm_importance else None,
        llm_importance=config.llm_importance,
    )
    return ProtoAgent(
        config=config,
        client=client,
        memory=memory,
        tools=tools,
        memory_service=memory_service,
    )


def cmd_status(args: argparse.Namespace) -> int:
    config = AgentConfig.load()
    base_url = args.base_url or config.base_url
    profile = LlamaServerProfile()
    report = status_report(profile, base_url=base_url)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    profile = profile_from_args(args)
    print("Starting llama-server:")
    print(" ".join(profile.server_command()))
    return run_server_foreground(profile)


def cmd_chat(args: argparse.Namespace) -> int:
    if args.stream and args.prompt:
        return _cmd_chat_stream(args)
    agent = make_agent(args)
    if args.prompt:
        run = agent.run(args.prompt, thread_id=args.thread_id, max_steps=args.max_steps)
        print(run.final)
        if run.tool_events:
            print(f"\n[tool events: {len(run.tool_events)} | thread: {run.thread_id}]")
        return 0

    print("ProtoAGI interactive chat. Ctrl+C or empty line to exit.")
    thread_id = args.thread_id
    while True:
        try:
            prompt = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            return 0
        try:
            run = agent.run(prompt, thread_id=thread_id, max_steps=args.max_steps)
        except OpenAICompatError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
        thread_id = run.thread_id
        print(f"\nProtoAGI> {run.final}")
        if run.tool_events:
            print(f"[tool events: {len(run.tool_events)} | thread: {run.thread_id}]")


def _cmd_chat_stream(args: argparse.Namespace) -> int:
    config = AgentConfig.load().with_cli_overrides(
        base_url=getattr(args, "base_url", None),
        model=getattr(args, "model", None),
        max_tokens=getattr(args, "max_tokens", None),
        temperature=getattr(args, "temperature", None),
        top_p=getattr(args, "top_p", None),
    )
    client = OpenAICompatibleClient(config.base_url, config.model)
    messages = [
        {"role": "system", "content": "You are ProtoAGI, answering directly without tool calls."},
        {"role": "user", "content": str(args.prompt)},
    ]
    try:
        for chunk in client.chat_completion_stream(
            messages,
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_tokens,
        ):
            print(chunk, end="", flush=True)
        print()
        return 0
    except OpenAICompatError as exc:
        print(f"streaming error: {exc}", file=sys.stderr)
        return 2


def cmd_bench(args: argparse.Namespace) -> int:
    config = AgentConfig.load().with_cli_overrides(base_url=args.base_url, model=args.model)
    client = OpenAICompatibleClient(config.base_url, config.model)
    results = bench_endpoint(
        client,
        prompt=args.prompt,
        rounds=args.rounds,
        max_tokens=args.max_tokens,
    )
    print(endpoint_results_to_json(results))
    return 0


def cmd_bench_llama(args: argparse.Namespace) -> int:
    profile = profile_from_args(args)
    values = [int(part.strip()) for part in args.n_cpu_moe.split(",") if part.strip()]
    returncode = run_llama_bench(
        profile,
        output_path=(PROJECT_ROOT / args.output).resolve(),
        n_cpu_moe_values=values,
        prompt_tokens=args.prompt_tokens,
        gen_tokens=args.gen_tokens,
        repetitions=args.repetitions,
    )
    print(f"llama-bench exit code: {returncode}")
    print(f"output: {(PROJECT_ROOT / args.output).resolve()}")
    return returncode


def cmd_init_config(args: argparse.Namespace) -> int:
    if DEFAULT_CONFIG_PATH.exists() and not args.force:
        print(f"Already exists: {DEFAULT_CONFIG_PATH}")
        return 0
    DEFAULT_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "gpt-oss-20b-MXFP4",
        "database_path": "data/protoagi.sqlite3",
        "temperature": 0.6,
        "top_p": 1.0,
        "max_tokens": 1536,
        "llm_importance": False,
        "plan_reflect": True,
        "plan_call_limit": 2,
        "tool_policy": {
            "allow_write": True,
            "allow_shell": False,
            "allow_unsafe_shell": False,
            "command_timeout_seconds": 30,
        },
    }
    DEFAULT_CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Wrote {DEFAULT_CONFIG_PATH}")
    return 0


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
        print("TELEGRAM_BOT_TOKEN is not set. Create a bot with @BotFather and set the token.", file=sys.stderr)
        return 2

    bot = build_nikola_bot(agent_config=agent_config, telegram_config=telegram_config)
    me = bot.bootstrap(delete_webhook=args.delete_webhook, drop_pending_updates=args.drop_pending_updates)
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


def _resolve_db_path(args: argparse.Namespace) -> Path:
    config = AgentConfig.load()
    raw = getattr(args, "db", None)
    if raw:
        return Path(raw).resolve()
    return config.database_path


def _build_memory_service(db_path: Path) -> tuple[MemoryStore, MemoryService]:
    config = AgentConfig.load()
    store = MemoryStore(db_path)
    embedding_config = EmbeddingConfig(
        base_url=config.embedding.base_url,
        model=config.embedding.model,
        timeout_seconds=config.embedding.timeout_seconds,
        request_dimensions=config.embedding.request_dimensions,
    )
    embedding_client = EmbeddingClient(embedding_config) if embedding_config.enabled else None
    importance_client = (
        OpenAICompatibleClient(config.base_url, config.model)
        if config.llm_importance
        else None
    )
    service = MemoryService(
        store,
        embedding_client=embedding_client,
        embedding_backend=config.embedding.backend,
        importance_client=importance_client,
        llm_importance=config.llm_importance,
    )
    return store, service


def cmd_memory_eval(args: argparse.Namespace) -> int:
    k_values = tuple(int(v.strip()) for v in args.k.split(",") if v.strip())
    config = AgentConfig.load()
    embedding_client: EmbeddingClient | None = None
    if args.with_embeddings:
        embedding_config = EmbeddingConfig(
            base_url=config.embedding.base_url,
            model=config.embedding.model,
            timeout_seconds=config.embedding.timeout_seconds,
            request_dimensions=config.embedding.request_dimensions,
        )
        if not embedding_config.enabled:
            print(
                "PROTOAGI_EMBED_MODEL is not configured; running FTS-only.",
                file=sys.stderr,
            )
        else:
            embedding_client = EmbeddingClient(embedding_config)
    report = run_eval(
        corpus_path=Path(args.corpus).resolve(),
        embedding_client=embedding_client,
        k_values=k_values,
    )
    if args.json:
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(f"queries: {len(report.queries)}")
        for k, value in sorted(report.recall_at_k.items()):
            print(f"recall@{k}: {value:.3f}")
        print(f"MRR: {report.mrr:.3f}")
        misses = [item for item in report.queries if item.rank is None]
        if misses:
            print("\nMisses:")
            for miss in misses:
                print(f"  - {miss.query!r}: expected one of {miss.retrieved[:3]}")
    return 0


def cmd_memory_stats(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args)
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 2
    store, _ = _build_memory_service(db_path)
    from .admin import _stats

    print(json.dumps(_stats(store), ensure_ascii=False, indent=2))
    return 0


def cmd_memory_prune(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args)
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 2
    _, service = _build_memory_service(db_path)
    result = service.prune(
        scope=args.scope,
        persona_key=args.persona,
        score_threshold=args.score_threshold,
        keep_newer_than_days=args.keep_newer_than_days,
        dry_run=args.dry_run,
        return_plan=bool(args.json),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def cmd_memory_consolidate(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args)
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 2
    _, service = _build_memory_service(db_path)
    result = service.consolidate(
        scope=args.scope,
        persona_key=args.persona,
        dry_run=args.dry_run,
        return_plan=bool(args.json),
    )
    if isinstance(result, dict):
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"merged": result}, ensure_ascii=False, indent=2))
    return 0


def cmd_backup(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args)
    to_path = Path(args.to).resolve() if args.to else default_backup_path(db_path)
    written = backup_database(db_path, to_path)
    print(json.dumps({"backup": str(written)}, ensure_ascii=False, indent=2))
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args)
    restored = restore_database(db_path, Path(args.from_path).resolve())
    print(json.dumps({"restored": str(restored)}, ensure_ascii=False, indent=2))
    return 0


def cmd_admin(args: argparse.Namespace) -> int:
    db_path = _resolve_db_path(args)
    if not db_path.exists():
        print(f"no database at {db_path}", file=sys.stderr)
        return 2
    store, service = _build_memory_service(db_path)
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


def main(argv: list[str] | None = None) -> int:
    _prefer_utf8_console()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "status":
            return cmd_status(args)
        if args.command == "serve":
            return cmd_serve(args)
        if args.command == "chat":
            return cmd_chat(args)
        if args.command == "bench":
            return cmd_bench(args)
        if args.command == "bench-llama":
            return cmd_bench_llama(args)
        if args.command == "init-config":
            return cmd_init_config(args)
        if args.command == "telegram":
            return cmd_telegram(args)
        if args.command == "memory-eval":
            return cmd_memory_eval(args)
        if args.command == "memory-stats":
            return cmd_memory_stats(args)
        if args.command == "memory-prune":
            return cmd_memory_prune(args)
        if args.command == "memory-consolidate":
            return cmd_memory_consolidate(args)
        if args.command == "backup":
            return cmd_backup(args)
        if args.command == "restore":
            return cmd_restore(args)
        if args.command == "admin":
            return cmd_admin(args)
    except OpenAICompatError as exc:
        print(f"OpenAI-compatible endpoint error: {exc}", file=sys.stderr)
        return 2
    except TelegramApiError as exc:
        if is_telegram_polling_conflict(exc):
            print(
                "Telegram polling conflict: another Telegram bot instance is already running for this token.",
                file=sys.stderr,
            )
            print("Stop the other instance before starting a new one.", file=sys.stderr)
        else:
            print(f"Telegram API error: {exc}", file=sys.stderr)
        return 2
    except FileNotFoundError as exc:
        print(f"Missing file: {exc}", file=sys.stderr)
        return 2
    except BackupError as exc:
        print(f"Backup error: {exc}", file=sys.stderr)
        return 2
    return 1


def _prefer_utf8_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8")
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
