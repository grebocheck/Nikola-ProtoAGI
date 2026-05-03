# ProtoAGI

ProtoAGI is a local agentic research harness for `gpt-oss-20b-MXFP4.gguf`.
It is not a claim of artificial general intelligence. It is a practical system
for iterating toward stronger autonomy on limited local hardware:

- llama.cpp CUDA runtime for the local model
- OpenAI-compatible chat client
- tool-using agent loop
- SQLite long-term memory with FTS search
- workspace, shell, web, and GPU inspection tools
- Telegram conversation mode with deep profiles for Микола or Соломія
- benchmark scripts and repeatable launch profiles

## Hardware target

This workspace was prepared for:

- RTX 5070 Ti with 16 GB VRAM
- 32 GB system RAM
- Ryzen 7 7800X3D
- Windows + CUDA 13.1
- model file: `gpt-oss-20b-MXFP4.gguf`

The model is close to the VRAM limit. Start with 8k context and partial MoE CPU
offload, then increase context only after benchmarking.

## Quick start

Copy local environment defaults:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` if you want Telegram mode or custom runtime settings. `.env`
is local-only and ignored by git.

Start the local model server:

```powershell
.\scripts\start-server.ps1 -CtxSize 8192 -CpuMoE 4
```

In a second terminal:

```powershell
$env:PYTHONPATH="src"
python -m protoagi status
python -m protoagi chat --prompt "Create a short plan for improving this repository."
```

For an interactive session:

```powershell
$env:PYTHONPATH="src"
python -m protoagi chat --allow-write --allow-shell
```

`--allow-shell` gives the local agent permission to run PowerShell commands
inside this workspace. Destructive command patterns are still blocked unless
`--allow-unsafe-shell` is also used.

## Telegram mode

Telegram mode uses a profile selected in `.env`. Start with `mykola` or switch
to `solomiya` for a separate identity, user model, and conversation style.
Telegram memory is shared globally across chats and profiles:

```env
PROTOAGI_TELEGRAM_PERSONA=solomiya
```

Start the model server, create a bot with `@BotFather`, then run:

```powershell
$env:TELEGRAM_BOT_TOKEN="123456:ABC..."
.\scripts\start-telegram.ps1
```

Or start the local model server and Telegram bot together:

```powershell
$env:TELEGRAM_BOT_TOKEN="123456:ABC..."
.\scripts\start-nikola-stack.ps1
```

There is also a root convenience launcher:

```powershell
.\run-nikola.bat
```

`Ctrl+C` stops the Telegram bot and the local model servers that the stack uses.
Pass `-KeepServers` if you intentionally want to leave them warm after the bot
exits.

Stop the whole local stack for this workspace:

```powershell
.\stop-nikola.bat
```

Unexpected Telegram loop crashes are logged to `runs\telegram-errors.log`.

The bot is intentionally conversation-first: profile switching and behavior are
configured through `.env`, not Telegram commands. `/start` only registers the
chat and greets with the active profile.

Profiles can keep a fictional self-memory (`NIKOLA_FICTIONAL_SELF=1`) for
stable tastes, running jokes, and small invented persona details. This memory is
separate from user facts and remains tied to the active persona.

The active profile can also use stickers from `Bocchi_the_Rock_sticker_pack2`,
`SenkoSan`, and `M1ku_Hatsune`. It can send a short burst of up to three
messages plus stickers for natural Telegram pacing, while Telegram reply is
reserved for group/current replies or explicit recent message IDs.
Optional Telegram image recognition is available through
`PROTOAGI_VISION_BASE_URL` and `PROTOAGI_VISION_MODEL`. When the vision model is
configured for localhost, `run-nikola.bat` starts a separate lightweight
`llama-server` on port `8081` and downloads the GGUF on first launch.

More details: [docs/TELEGRAM.md](docs/TELEGRAM.md).

## Git Hygiene

The repository ignores local secrets, model files, downloaded runtimes, logs,
SQLite memory, and Python caches. Keep real tokens in `.env`; commit
`.env.example` only.

Architecture audit and pre-push checklist: [docs/AUDIT.md](docs/AUDIT.md).
Forward-looking plan: [docs/ROADMAP.md](docs/ROADMAP.md).

## Benchmark first

Run llama.cpp benchmark profiles:

```powershell
.\scripts\bench-llama.ps1
```

Benchmark the OpenAI-compatible endpoint after the server is running:

```powershell
$env:PYTHONPATH="src"
python -m protoagi bench --rounds 3
```

## Memory ops

```powershell
# Recall benchmark over the bundled golden corpus
$env:PYTHONPATH="src"
python -m protoagi memory-eval

# Inspect SQLite memory store
python -m protoagi memory-stats

# Forget low-value old items
python -m protoagi memory-prune --dry-run

# Local admin dashboard at http://127.0.0.1:8765
python -m protoagi admin
```

## Project layout

- `tools/llama.cpp/` - downloaded llama.cpp CUDA runtime
- `scripts/` - launch and benchmark helpers
- `src/protoagi/` - agent, memory, tools, runtime, CLI
- `data/` - SQLite memory database
- `runs/` - benchmark output
- `docs/` - architecture and runtime notes

## Design stance

The first milestone is not a theatrical "one prompt AGI". The first milestone is
a measurable system that can:

1. run the local model reliably,
2. use tools safely,
3. retain useful memory,
4. inspect its environment,
5. execute multi-step tasks,
6. leave logs and metrics that let us improve it.
