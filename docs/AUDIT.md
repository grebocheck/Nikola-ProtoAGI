# Architecture Audit

Last audit: 2026-05-03

For the forward-looking plan and prioritized backlog, see
[ROADMAP.md](ROADMAP.md).

## Current Shape

ProtoAGI has four main layers:

- `llama.cpp` runtime served through an OpenAI-compatible endpoint
- Python agent core for local tool use and memory
- SQLite state store for messages, facts, Telegram chats, and key-value state
- Telegram personality layer with `.env`-selected profiles

The repository is now arranged so source code and documentation can be pushed to
git without committing local model files, downloaded runtimes, logs, databases,
or secrets.

## 2026-05-03 phase 9 (P3 research baselines)

- Self-tuning reply style: `ReplyStyleTuner` records per-chat engagement
  signals from replies, reaction updates, and edited messages, then passes an
  `adaptive_reply_style` hint into Telegram decision/reply/initiative prompts.
- Memory federation: `protoagi memory-export` and `protoagi memory-import`
  create and verify HMAC-signed JSON bundles. Imports are idempotent through
  federation-id tags.
- Admin graph: `GET /api/memory-graph` returns memory/tag nodes plus tag and
  supersession edges; the dashboard renders them in a dependency-free canvas
  force layout.
- Voice: Telegram voice/audio messages can be transcribed through an
  OpenAI-compatible `/audio/transcriptions` endpoint and stored as episodic
  voice memory. Optional TTS uses `/audio/speech` and sends a Telegram voice
  reply after text.

Test count: 130 -> 138.

## 2026-05-03 phase 8 (series A follow-up cleared)

- Privacy-mode migrations: `protoagi memory-rescope --to user` moves legacy
  Telegram rows from `scope=global` to `scope=user` based on existing
  `user:<id>` / `source_chat:<id>` tags.
- Telegram tool decisions now inline trivial `recall` results and record
  LLM-call histograms / averages in admin stats, so simple memory questions no
  longer require a merge completion.
- `protoagi bench-tools` measures whether the local model emits native
  `tool_calls`, JSON `tool_request`, both, or neither when both `tools` and
  `response_format` are set.
- `media_blobs` and LLM importance scores now have garbage collection:
  reflection prunes old orphan media plus old `importance_cache` rows, and
  admin stats report both row counts separately.
- `web_get` resolves and validates DNS once, then opens the HTTP socket to the
  validated IP to avoid DNS rebinding between validation and fetch.
- `AsyncBotRunner.poll_once` no longer acknowledges failed updates past the
  Telegram offset cursor, so transient failures can replay on the next poll.
- `EmbeddingClient` now uses an `OrderedDict` LRU cache instead of FIFO
  insertion-order eviction.

Test count: 121 -> 130.

## 2026-05-03 phase 6 (P2 backlog cleared)

- Embedding recall now goes through an `EmbeddingBackend` boundary. The exact
  flat cosine backend remains the default, while `PROTOAGI_EMBED_BACKEND=lsh`
  enables a pure-Python random-hyperplane LSH backend for larger stores without
  adding binary dependencies.
- Multimodal memory persists Telegram image bytes in `media_blobs` and links
  image-derived memory items through `memory_items.media_id`. The admin API can
  serve stored media at `GET /api/media/<file_id>`.
- `ProtoAgent` now makes a short JSON execution plan before tool use and can
  update it once after a tool observation by default. `PROTOAGI_PLAN_REFLECT`
  and `PROTOAGI_PLAN_CALL_LIMIT` control the extra calls.
- Telegram multi-instance deployment is explicit: `protoagi telegram --db ...`
  selects a per-persona SQLite file and `--persona ...` overrides the env
  persona for that process.

Test count: 116 → 121.

## 2026-05-03 phase 5 (P1 backlog cleared)

- `MemoryService.score_importance_llm` adds opt-in model scoring for memory
  importance/kind via `PROTOAGI_LLM_IMPORTANCE=1`, with SHA256 cache entries
  (moved from `kv` to `importance_cache` in Phase 8) and deterministic
  heuristic fallback.
- Telegram memory can now be per-user isolated with
  `PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0`; user facts use `scope=user`, recall
  passes the current Telegram `user_id`, and global behavior remains the
  default for single-owner bots.
- `protoagi backup` / `protoagi restore` use SQLite's online backup API,
  validate backups with `PRAGMA integrity_check`, and restore via an atomic
  replacement that cleans stale WAL/SHM sidecars.
- `memory-prune` and `memory-consolidate` expose dry-run JSON plans with
  per-item kept/dropped reasons. Admin has preview endpoints for both passes.
- Telegram decisions can request bounded tools (`recall`, `remind_me`) via
  `tool_request` or OpenAI-style `tool_calls`; tool results are merged into a
  final decision JSON before sending.
- `AsyncBotRunner` provides opt-in concurrent polling through `asyncio`,
  `asyncio.to_thread`, and a semaphore around update handling. CLI flag:
  `protoagi telegram --async`.

Test count: 107 → 116.

## 2026-05-03 phase 4 (P0 backlog cleared)

- Embedding llama-server is part of the stack:
  [scripts/start-embed-server.ps1](../scripts/start-embed-server.ps1)
  serves `bge-m3-Q4_K_M` in `--embedding` mode on port 8082.
  [scripts/start-nikola-stack.ps1](../scripts/start-nikola-stack.ps1)
  launches and tears it down alongside the chat / vision servers.
  ``.env.example`` documents `PROTOAGI_EMBED_HF_REPO`.
- ``NikolaBot.run_reflection_pass`` now prunes low-value items in the
  global and active-persona scopes after consolidation. Counters
  ``pruned_global`` / ``pruned_persona`` are surfaced.
- Admin dashboard supports inline curation: ``set_pinned`` and
  ``update_memory`` in [memory.py](../src/protoagi/memory.py); endpoints
  ``POST /api/memories/<id>/pin`` and ``POST /api/memories/<id>/edit``.
  The HTML view ships save / pin / delete buttons with a small JS shim.
- [scripts/eval-memory.ps1](../scripts/eval-memory.ps1) runs the eval
  harness, writes a timestamped report to ``runs/``, and compares
  against [runs/memory-eval-baseline.json](../runs/memory-eval-baseline.json).
  Recall@k drops > 5 pp fail the script.
- Schema fix: FTS5 is now self-contained (no ``content='memory_items'``)
  so ``update_memory`` can DELETE/INSERT rows with plain SQL. Existing
  experiment DBs keep their old FTS table; fresh checkouts use the new
  one.

Test count: 99 → 107.

## 2026-05-03 phase 3 (operations & evaluation)

- ``MemoryService.prune`` forgets low-value items by score
  (``0.5*importance + 0.3*recency + 0.2*access``) with a configurable
  threshold and a default 30-day grace window. Pinned items, items in
  ``protect_kinds`` (defaults to ``persona_self``), and superseded rows are
  skipped.
- Reminders are first-class in the decision payload now: ``Decision`` and
  ``InitiativeDecision`` carry a ``reminders`` list, the JSON schemas advertise
  it, and ``NikolaBot._persist_reminder_requests`` persists them with
  ``trigger_at`` resolution from either ``in_minutes`` or an explicit ISO
  timestamp. The bot's existing dispatcher delivers them at the next worker
  tick.
- Memory recall harness in ``protoagi.memory_eval``: a JSON corpus
  (``config/memory_eval/golden.json``), a ``protoagi memory-eval`` CLI that
  reports recall@k and MRR, and ``--with-embeddings`` to include the cosine
  index. The bundled corpus surfaces the FTS-only blind spot for synonyms
  (characters / phrases) without the embedding layer.
- Operational CLIs: ``protoagi memory-stats``, ``memory-prune``,
  ``memory-consolidate``, and ``admin`` (a tiny stdlib ``http.server``
  dashboard with HTML view and JSON endpoints for stats, memories,
  reminders, chats, plus POSTable delete / prune / consolidate actions).

## 2026-05-03 phase 2 (autonomy)

- Constrained JSON output: ``OpenAICompatibleClient.chat_completion`` now
  forwards ``response_format`` to llama-server, and the Telegram decision
  / initiative paths pass JSON schemas (``DECISION_JSON_SCHEMA``,
  ``INITIATIVE_JSON_SCHEMA``) so the model emits well-formed JSON instead
  of relying on best-effort regex extraction.
- Streaming: ``OpenAICompatibleClient.chat_completion_stream`` parses
  ``text/event-stream`` deltas. ``protoagi chat --prompt ... --stream``
  prints chunks live for quick sanity checks.
- Reminder dispatcher: ``NikolaBot.dispatch_due_reminders`` delivers
  pending reminders into Telegram chats, marking unrecoverable rows as
  ``cancelled``. The ``remind_me`` tool can now actually surface its
  output.
- Reflection loop: every ~6 hours ``NikolaBot.run_reflection_pass``
  consolidates near-duplicate memories and (when ``fictional_self`` is
  enabled) asks the model for one or two short first-person reflections
  that are stored as ``persona_self`` memories.
- ``BotRunner`` runs polling on the main thread and a worker thread for
  initiative / reminders / reflection, so a just-due reminder fires within
  ~1 s instead of blocking on the long-poll. ``--single-thread`` keeps the
  legacy behavior for debugging.

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
- Telegram remembered facts are global by default for the single-owner bot.
  Multi-user deployments should set `PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0` and
  still define an explicit privacy policy before broad access.
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
