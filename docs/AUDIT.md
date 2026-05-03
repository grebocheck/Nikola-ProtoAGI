# Architecture Audit

Last audit: 2026-05-03

## Current Shape

ProtoAGI has four main layers:

- `llama.cpp` runtime served through an OpenAI-compatible endpoint
- Python agent core for local tool use and memory
- SQLite state store for messages, facts, Telegram chats, and key-value state
- Telegram personality layer with `.env`-selected profiles

The repository is now arranged so source code and documentation can be pushed to
git without committing local model files, downloaded runtimes, logs, databases,
or secrets.

## 2026-05-03 modernization

- Replaced the flat `facts` table with a typed `memory_items` schema: `kind`
  (semantic / episodic / procedural / persona_self / fact), `scope`
  (global / user / chat / persona), `importance`, `confidence`,
  `supersedes_id` / `superseded_by`, access metadata, and a normalized
  `memory_tags` table that fixed the substring-matching bug in legacy
  `search_tagged`.
- Added an optional embedding pipeline (`/v1/embeddings`) plus a pure-Python
  cosine index in `protoagi.embedding`.
- Added `protoagi.memory_service` as the canonical recall facade with a
  blended FTS + cosine + recency + importance score and a heuristic
  consolidation pass.
- Added a `users` table and a `reminders` table; the agent gained
  `remind_me` / `list_reminders` tools.
- Refactored the 1.5k-line `telegram_bot.py` monolith into the
  `protoagi.telegram` package (`api`, `config`, `text`, `json_io`,
  `stickers`, `vision`, `identity`, `prompts`, `bot`). The legacy module is
  preserved as a compatibility shim.
- Personas moved out of Python into `config/personas/*.json` with built-in
  fallbacks for fresh checkouts.
- Hardened `web_get` against SSRF (loopback / private / link-local /
  multicast / reserved IPs and `localhost` are blocked).
- Tightened the PowerShell command blocklist with anchored regex patterns to
  remove the substring-evasion gap (e.g. `del.exe`, `reg delete`, `shutdown`).
- Switched SQLite to WAL mode with a single long-lived connection.
- Added size-based rotation for `runs/telegram-errors.log`.
- Wrapped agent user prompts in `<user_input>` markers and instructed the
  system prompt to treat them as data rather than instructions.

## Fixed During Audit

- Added `.gitignore` for `.env`, local config, GGUF models, downloaded runtime
  binaries, SQLite state, logs, and Python caches.
- Added `.env.example` for shareable settings and `.env` for local-only values.
- Added a dependency-free `.env` loader.
- Added Windows and POSIX convenience launch scripts.
- Fixed legacy chat-scoped memory search to require exact tags and avoid accidental
  cross-chat matches such as `telegram_chat_12` vs `telegram_chat_123`.
- Hardened Telegram env parsing so invalid integers or reply modes fall back to
  safe defaults.
- Made the Telegram polling loop resilient to transient Telegram/network errors.
- Changed Telegram replies from always-on to explicit `reply_to` decisions.
- Added sticker support through sticker set discovery and cached file IDs.
- Added deep Telegram profiles for `mykola` and `solomiya`, with profile-specific
  prompts and aliases over shared Telegram memory and thread history.
- Removed Telegram command-based control from the conversational surface; runtime
  behavior and persona are configured through `.env`.

## Remaining Risks

- Real Telegram integration still needs a live `TELEGRAM_BOT_TOKEN` smoke test.
- Sticker packs are fetched lazily; if a pack is removed or renamed, sticker
  sending silently degrades to text-only interaction.
- `llama.cpp` binaries are intentionally ignored. A new machine must download or
  build the runtime again.
- The local SQLite database is ignored. This is correct for git, but production
  deployments need backup/export if memory matters.
- Telegram remembered facts and compact model dialogue history are shared
  globally. This is intentional for a single-owner bot, but multi-user
  deployments need an explicit privacy policy before enabling broad access.
- The model file is ignored. Document or script model acquisition before sharing
  the repo with another machine.
- The agent has a safe shell policy, but enabling `--allow-unsafe-shell` remains
  inherently risky.

## Pre-Push Checklist

```powershell
.\run-tests.bat
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
