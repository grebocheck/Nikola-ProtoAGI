# ProtoAGI

ProtoAGI is a local agentic research harness for a self-hosted
OpenAI-compatible model endpoint, currently tuned around
`models/gpt-oss-20b-MXFP4.gguf`. It is not a claim of artificial general
intelligence. It is a practical system for iterating toward stronger autonomy on
limited local hardware:

- llama.cpp CUDA runtime for the local model
- OpenAI-compatible chat client
- tool-using agent loop
- SQLite long-term memory with FTS search
- workspace, shell, web, and GPU inspection tools
- Telegram conversation mode with deep profiles for `mykola` or `solomiya`
- benchmark scripts and repeatable launch profiles

The public repository contains code, scripts, tests, configuration examples, and
documentation. Model weights, Hugging Face caches, downloaded llama.cpp
runtimes, local databases, Telegram media, logs, and real secrets are kept out
of git.

## Hardware target

This workspace was prepared for:

- RTX 5070 Ti with 16 GB VRAM
- 32 GB system RAM
- Ryzen 7 7800X3D
- Windows + CUDA 13.1
- model file: `models/gpt-oss-20b-MXFP4.gguf`

The model is close to the VRAM limit. Start with 8k context and partial MoE CPU
offload, then increase context only after benchmarking.

Download or place model weights yourself under `models/` according to their
upstream licenses. The default vision and embedding launchers also redirect
Hugging Face cache files into `models/hf-cache/`, which remains local-only.

## Quick start

Copy local environment defaults:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set `TELEGRAM_BOT_TOKEN`. `.env` is local-only and ignored
by git.

Start the local model server:

```powershell
.\scripts\start-server.ps1 -CtxSize 8192 -CpuMoE 4
```

Then in another terminal, run the bot or the admin UI:

```powershell
$env:PYTHONPATH = "src"
python -m protoagi telegram          # Telegram conversation bot
python -m protoagi admin             # Local admin dashboard on :8765
```

There are intentionally only two CLI subcommands now (`telegram` and
`admin`). Experimental utilities (bench / chat / eval / memory ops /
backup / federation) lived on the old single-shot agent loop and have
been retired; their replacements live inside the admin UI or have
moved to PowerShell scripts under `scripts/`.

## Telegram mode

Telegram mode uses a profile selected in `.env`. Start with `mykola` or switch
to `solomiya` for a separate identity, user model, and conversation style.
Telegram memory is shared globally across chats and profiles:

```env
PROTOAGI_TELEGRAM_PERSONA=solomiya
```

For multi-user bots, disable global Telegram memory so user facts are stored
and recalled with per-user isolation:

```env
PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0
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

Run multiple personas as separate instances by giving each process a distinct
token and database:

```powershell
$env:PYTHONPATH="src"
python -m protoagi telegram --persona mykola --db data/mykola.sqlite3 --token "123:AAA"
python -m protoagi telegram --persona solomiya --db data/solomiya.sqlite3 --token "456:BBB"
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
Incoming image bytes and captions are persisted in SQLite media memory so later
recall can refer back to old photos. When the embedding endpoint supports a
joint image/text model, media-linked memories can use image embeddings for
photo-oriented recall; otherwise they fall back to caption/text recall.

Optional voice transcription and TTS are configured through `.env` with
OpenAI-compatible `/audio/transcriptions` and `/audio/speech` endpoints. The
project does not bundle voice or TTS model weights.

More details: [docs/TELEGRAM.md](docs/TELEGRAM.md).

## License and model weights

The project source code, scripts, tests, and documentation are licensed under
the Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

Third-party model weights are not distributed by this repository and are not
relicensed by ProtoAGI. Keep them local under `models/` and follow each
upstream license and usage policy before downloading, modifying, hosting, or
redistributing them. The default model references are tracked in
[THIRD_PARTY_MODELS.md](THIRD_PARTY_MODELS.md).

## Git Hygiene

The repository ignores local secrets, model files, downloaded runtimes, logs,
SQLite memory, Hugging Face caches, and Python caches. Keep real tokens in
`.env`; commit `.env.example` only. Before publishing or tagging, check:

```powershell
git status --short
git check-ignore -v .env models/gpt-oss-20b-MXFP4.gguf tools/llama.cpp/llama-server.exe data/protoagi.sqlite3
```

Architecture audit and pre-push checklist: [docs/AUDIT.md](docs/AUDIT.md).
Forward-looking plan: [docs/ROADMAP.md](docs/ROADMAP.md).

## Benchmark first

Run llama.cpp benchmark profiles:

## Admin UI

The React/Tailwind admin lives under [src/protoagi/admin_panel/web/](src/protoagi/admin_panel/web/).
Production bundle is served by the Python admin server.

`run-nikola.bat` (and `scripts/start-nikola-stack.ps1`) bootstrap the
admin automatically — first run installs npm deps and builds the SPA,
subsequent runs reuse the cached `dist/`. Open <http://127.0.0.1:8765>
once the bot logs `Admin ready: ...`. Pages:

- **Огляд** — counts (active/superseded memories, open goals, unresolved conflicts, user models).
- **Память** — search, filter by kind/scope/persona/pinned, edit/pin/delete.
- **Цілі** — open / completed / abandoned, manual close/reopen.
- **Суперечності** — review pairs the system flagged as semantically close
  but not auto-merged; resolve as superseded / kept_both / dismissed.
- **Чати** — Telegram chats + per-chat reasoning log.

If you don't want the admin spun up alongside the bot, pass `-NoAdmin`
to `start-nikola-stack.ps1`. To force a fresh SPA build pass
`-ForceAdminRebuild`. Standalone:

```powershell
.\scripts\start-admin-server.ps1                  # auto-build + start
.\scripts\start-admin-server.ps1 -Stop            # kill the running admin
.\scripts\start-admin-server.ps1 -Logs            # tail stderr
.\scripts\start-admin-server.ps1 -ForceRebuild    # re-run npm install + build
```

For development with hot reload run Vite separately:

```powershell
cd src\protoagi\admin_panel\web
npm run dev   # http://127.0.0.1:5173, proxies /api/* to the Python server
```

Requires Node 20+ on `PATH`. When missing, the admin step warns and the
bot still runs.

Set `PROTOAGI_LLM_IMPORTANCE=1` to let the chat model score new memory writes
for importance/kind with a SHA256 cache. The deterministic heuristic remains
the default.

Embedding recall uses exact flat cosine by default. For larger stores, set
`PROTOAGI_EMBED_BACKEND=lsh` to use the dependency-free approximate backend.

Developer checks:

```powershell
python -m pip install -e ".[dev]"
python -m ruff check src/
python -m mypy --strict src/protoagi/
```

Live smoke testing is optional and expects a local GGUF model:

```powershell
.\scripts\smoke-test.ps1 -ModelPath C:\models\tiny.gguf -Port 8090
```

## Project layout

- `models/` - local GGUF/model weights and HF caches, ignored except `.gitkeep`
- `tools/llama.cpp/` - downloaded llama.cpp CUDA runtime
- `scripts/` - launch and benchmark helpers
- `src/protoagi/` - agent, runtime, CLI, and domain packages
- `src/protoagi/agent_tools/` - tool registry implementation
- `src/protoagi/admin_panel/` - admin HTTP UI and dashboard data shaping
- `src/protoagi/evals/` - endpoint benchmarks and memory recall evaluation
- `src/protoagi/storage/` - SQLite storage, typed models, backups, federation, and recall service
- `src/protoagi/telegram/` - Telegram transport, prompts, style, media, stickers, and orchestration
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
