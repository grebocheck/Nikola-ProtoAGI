# Architecture Audit

Last audit: 2026-05-01

## Current Shape

ProtoAGI has four main layers:

- `llama.cpp` runtime served through an OpenAI-compatible endpoint
- Python agent core for local tool use and memory
- SQLite state store for messages, facts, Telegram chats, and key-value state
- Telegram personality layer named Микола

The repository is now arranged so source code and documentation can be pushed to
git without committing local model files, downloaded runtimes, logs, databases,
or secrets.

## Fixed During Audit

- Added `.gitignore` for `.env`, local config, GGUF models, downloaded runtime
  binaries, SQLite state, logs, and Python caches.
- Added `.env.example` for shareable settings and `.env` for local-only values.
- Added a dependency-free `.env` loader.
- Added Windows and POSIX convenience launch scripts.
- Fixed chat-scoped memory search to require exact tags and avoid accidental
  cross-chat matches such as `telegram_chat_12` vs `telegram_chat_123`.
- Hardened Telegram env parsing so invalid integers or reply modes fall back to
  safe defaults.
- Made the Telegram polling loop resilient to transient Telegram/network errors.

## Remaining Risks

- Real Telegram integration still needs a live `TELEGRAM_BOT_TOKEN` smoke test.
- The repository is not initialized as git yet unless you run `git init`.
- `llama.cpp` binaries are intentionally ignored. A new machine must download or
  build the runtime again.
- The local SQLite database is ignored. This is correct for git, but production
  deployments need backup/export if memory matters.
- The model file is ignored. Document or script model acquisition before sharing
  the repo with another machine.
- The agent has a safe shell policy, but enabling `--allow-unsafe-shell` remains
  inherently risky.

## Pre-Push Checklist

```powershell
.\run-tests.bat
git init
git status --short
git add .gitignore .gitattributes .env.example README.md pyproject.toml docs examples scripts src tests run-nikola.bat run-nikola.sh run-tests.bat run-tests.sh data/.gitkeep runs/.gitkeep tools/.gitkeep
git status --short
```

Before committing, confirm that these do not appear in `git status`:

- `.env`
- `gpt-oss-20b-MXFP4.gguf`
- `tools/llama.cpp/*`
- `tools/downloads/*`
- `data/protoagi.sqlite3`
- `runs/*.log`
- `__pycache__/`
