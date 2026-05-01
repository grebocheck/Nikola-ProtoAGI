from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent import ProtoAgent
from .bench import bench_endpoint, endpoint_results_to_json, run_llama_bench
from .config import AgentConfig, DEFAULT_CONFIG_PATH, DEFAULT_MODEL_PATH, LlamaServerProfile, PROJECT_ROOT
from .memory import MemoryStore
from .openai_compat import OpenAICompatibleClient, OpenAICompatError
from .runtime import run_server_foreground, status_report
from .telegram_bot import TelegramConfig, build_nikola_bot
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

    telegram = sub.add_parser("telegram", help="Run Nikola, the Telegram conversation bot")
    telegram.add_argument("--token", default=None, help="Telegram bot token, defaults to TELEGRAM_BOT_TOKEN")
    telegram.add_argument("--allowed-chat-id", action="append", default=[])
    telegram.add_argument("--reply-mode", choices=["smart", "always", "mention", "silent"], default=None)
    telegram.add_argument("--base-url", default=None)
    telegram.add_argument("--model", default=None)
    telegram.add_argument("--poll-timeout", type=int, default=None)
    telegram.add_argument("--once", action="store_true", help="Process one polling batch and exit")
    telegram.add_argument("--no-proactive", action="store_true")
    telegram.add_argument("--proactive-check-seconds", type=int, default=None)
    telegram.add_argument("--proactive-cooldown-seconds", type=int, default=None)
    telegram.add_argument("--delete-webhook", action="store_true")
    telegram.add_argument("--drop-pending-updates", action="store_true")

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
    return ProtoAgent(config=config, client=client, memory=memory, tools=tools)


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
    agent_config = AgentConfig.load().with_cli_overrides(base_url=args.base_url, model=args.model)
    telegram_config = TelegramConfig.from_env()
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
        f"Микола online as @{me.get('username', 'unknown')} | "
        f"reply_mode={telegram_config.reply_mode} | proactive={telegram_config.proactive_enabled}"
    )
    if args.once:
        processed = bot.poll_once()
        proactive = bot.run_initiative_once() if telegram_config.proactive_enabled else 0
        print(f"Processed updates: {processed}; proactive messages: {proactive}")
        return 0
    bot.run_forever()
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
    except OpenAICompatError as exc:
        print(f"OpenAI-compatible endpoint error: {exc}", file=sys.stderr)
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
            except OSError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
