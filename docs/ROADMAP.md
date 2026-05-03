# ProtoAGI Roadmap

Living plan of what to build next. Last updated: 2026-05-03.

This document is the source of truth for "what's queued and why". When a
work item lands, move it from **Backlog** to **Done** and link the relevant
PR / commit. Do not delete completed items — they are reference for future
audits.

Effort scale: **S** = under a day, **M** = 1-3 days, **L** = a week,
**XL** = multiple weeks.

---

## Done so far (recap)

### Phase 1 — modernization (2026-05-03)
- Memory v2 schema with `kind`, `scope`, `importance`, supersession,
  normalized `memory_tags`, optional embeddings BLOB.
- `MemoryService` facade with hybrid recall.
- `protoagi.telegram/` package; `telegram_bot.py` is a thin compat shim.
- Personas in `config/personas/*.json` with built-in fallbacks.
- SSRF guard in `web_get`; tightened PowerShell blocklist; prompt-injection
  isolation via `<user_input>` markers; size-based log rotation.
- `remind_me` / `list_reminders` tools; `users` and `reminders` tables.

### Phase 2 — autonomy (2026-05-03)
- Constrained JSON schemas (`DECISION_JSON_SCHEMA`,
  `INITIATIVE_JSON_SCHEMA`).
- SSE streaming in `OpenAICompatibleClient.chat_completion_stream` and
  `protoagi chat --stream`.
- Reminder dispatcher (`NikolaBot.dispatch_due_reminders`).
- Reflection loop (`run_reflection_pass`) with consolidation +
  optional self-memory writes.
- `BotRunner` runs polling on the main thread, periodic tasks on a worker
  thread (reminders fire within ~1 s instead of waiting for long-poll).

### Phase 3 — operations & evaluation (2026-05-03)
- `MemoryService.prune()` with importance × recency × access decay.
- Reminders as first-class part of decision JSON; bot persists them.
- `protoagi.memory_eval` harness + `config/memory_eval/golden.json` corpus.
- CLIs: `memory-eval`, `memory-stats`, `memory-prune`, `memory-consolidate`.
- Local admin web dashboard (`protoagi admin`) — stdlib `http.server`,
  HTML view + JSON API for memory / reminders / chats.

Test count grew 53 → 99.

### Phase 4 — P0 finished (2026-05-03)
- **P0-1** Embedding server in the stack: new
  [scripts/start-embed-server.ps1](../scripts/start-embed-server.ps1)
  spins up llama-server in `--embedding` mode for `bge-m3`.
  [scripts/start-nikola-stack.ps1](../scripts/start-nikola-stack.ps1)
  launches it whenever `PROTOAGI_EMBED_MODEL` points at localhost,
  stops it on exit. `.env.example` documents
  `PROTOAGI_EMBED_HF_REPO`.
- **P0-2** `NikolaBot.run_reflection_pass` now also prunes low-value
  global / persona items (`score_threshold=0.10`,
  `keep_newer_than_days=60`). Counters surface as `pruned_global` /
  `pruned_persona`.
- **P0-3** Admin dashboard supports inline editing: `set_pinned`,
  `update_memory(text/importance/tags)` in
  [memory.py](../src/protoagi/memory.py); endpoints
  `POST /api/memories/<id>/pin` (toggle or explicit) and
  `POST /api/memories/<id>/edit`. The HTML view ships save / pin /
  delete buttons + a tiny JS shim with toast feedback.
- **P0-4** Regression gate:
  [scripts/eval-memory.ps1](../scripts/eval-memory.ps1) runs
  `protoagi memory-eval --json`, writes a timestamped report under
  `runs/`, compares against
  [runs/memory-eval-baseline.json](../runs/memory-eval-baseline.json),
  and exits non-zero when recall@k drops more than 5 pp.
  `-UpdateBaseline` regenerates the baseline.
- FTS5 schema switched from external-content to self-contained so
  `update_memory` can DELETE/INSERT into `memory_items_fts` with plain
  SQL. See decision log entry below.

Test count: 99 → 107.

---

## Backlog

Ordered by priority. Reorder freely as new evidence arrives.

### P0 — done (see Phase 4 above)

All four P0 items shipped on 2026-05-03:
- ✅ #1 Embedding server in stack
- ✅ #2 Auto-prune in `run_reflection_pass`
- ✅ #3 Admin pin/unpin + edit
- ✅ #4 Memory eval regression gate

### P1 — meaningful work that needs design before code

#### 5. LLM-driven importance scoring with cache — **M**
**Why:** the current heuristic in
[memory_service.py:score_importance](../src/protoagi/memory_service.py)
is intentionally simple. A small LLM call per write could rate facts much
better, but cost and latency must stay bounded.

**How:** add `MemoryService.score_importance_llm(text, context)` that
sends a tiny prompt (~150 tokens) requesting `{importance: 0..1, kind:
..., reasoning: ...}`. Cache responses by SHA256 of normalized text.
Make it opt-in via `PROTOAGI_LLM_IMPORTANCE=1`. Heuristic remains the
fallback so tests stay deterministic.

**Files:** [src/protoagi/memory_service.py](../src/protoagi/memory_service.py),
[src/protoagi/config.py](../src/protoagi/config.py),
new tests in `tests/test_memory_service.py`.

**Acceptance:** with `PROTOAGI_LLM_IMPORTANCE=1` and a real model, a
critical fact ("користувач має алергію на горіхи") gets importance > 0.85
and a one-shot mood note gets < 0.3. Cache hit rate ≥ 90 % on repeated
writes.

---

#### 6. Per-user privacy scopes — **M**
**Why:** today Telegram facts are global by design. Multi-user bots leak
user A's preferences into chats with user B. The dimension exists in the
schema (`user_id`, `scope=user`) but the recall path doesn't use it.

**How:** add a `PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0` env flag. When off,
`NikolaBot._search_chat_memory` passes `user_id` to `RecallQuery` and
`MemoryService.recall` enforces user isolation (already supported by
`_scope_matches`). Keep current behavior as default for the single-owner
use case; document the flag.

**Files:** [src/protoagi/telegram/bot.py](../src/protoagi/telegram/bot.py),
[src/protoagi/telegram/config.py](../src/protoagi/telegram/config.py),
[.env.example](../.env.example), [docs/TELEGRAM.md](TELEGRAM.md), tests.

**Acceptance:** with the flag off, two test users in the same group only
recall their own facts; with the flag on, behavior matches today's
golden tests.

---

#### 7. Backup / restore CLI — **S**
**Why:** the SQLite database is single-point-of-failure. WAL is great for
concurrency but does not protect against corruption or a `data/` wipe.

**How:** `protoagi backup --to data/backups/<timestamp>.sqlite3` calls
SQLite's online `.backup` API (`Connection.backup`). `protoagi restore
--from <path>` validates and atomically swaps. Document a 7-day rolling
retention recipe.

**Files:** [src/protoagi/cli.py](../src/protoagi/cli.py),
new `src/protoagi/backup.py`, tests, README "Memory ops" section.

**Acceptance:** backup → restore round-trip preserves all rows including
embeddings BLOBs.

---

#### 8. Memory diff for `consolidate` / `prune` — **S**
**Why:** today running prune/consolidate does the work and reports counts.
A "what would change" view with per-item before/after would build trust
in the heuristic.

**How:** extend both methods with a `dry_run=True` mode (prune already
has it; add to consolidate). When set, return a list of `{kept, dropped,
reason}` entries. Add `--json` output to the CLIs and a "preview"
endpoint to the admin server.

**Files:** [src/protoagi/memory_service.py](../src/protoagi/memory_service.py),
[src/protoagi/cli.py](../src/protoagi/cli.py),
[src/protoagi/admin.py](../src/protoagi/admin.py), tests.

**Acceptance:** `protoagi memory-consolidate --dry-run --json` prints the
exact supersession plan without touching the DB.

---

#### 9. Tool-use inside Telegram persona — **L**
**Why:** the workspace agent has tools (`recall`, `web_get`, `remind_me`,
…). The Telegram persona is a separate ad-hoc decision loop and cannot
call them. That asymmetry blocks turning the bot into something genuinely
"agentic".

**How:** introduce a two-mode model. Default Telegram path stays as is
(decision-only). When the model adds `tool_calls` to its response or the
decision JSON includes a `tool_request` field, run a bounded
`ProtoAgent`-style loop (max 4 steps), then merge the result back into
the decision (e.g. `reply` becomes the tool-augmented answer). Recall and
remind would be the first wired tools.

**Files:** [src/protoagi/telegram/bot.py](../src/protoagi/telegram/bot.py),
[src/protoagi/telegram/json_io.py](../src/protoagi/telegram/json_io.py),
new `src/protoagi/telegram/tool_runner.py`, tests.

**Acceptance:** "Соломіє, що ти памʼятаєш про мене?" triggers a `recall`
tool call and the reply quotes a real fact; the loop respects a 4-step
budget.

---

#### 10. Async polling — **L**
**Why:** today the long-poll blocks the main thread; the worker thread
mitigates reminders but a slow LLM call still blocks new updates being
read. Async unlocks parallel handling.

**How:** option A — keep stdlib only and use `asyncio` + `urllib` via
`asyncio.to_thread`. Option B — accept `httpx` (the only external dep
we'd add). Pipeline: `asyncio.TaskGroup` with separate tasks for
polling, initiative, reminders, reflection; `asyncio.Semaphore` to bound
concurrent LLM calls.

Decision pending: keep zero-deps or accept httpx? Current consensus is
to defer until Tool-use (#9) lands, since both will benefit from async
together.

**Files:** new `src/protoagi/telegram/async_runner.py`,
[src/protoagi/openai_compat.py](../src/protoagi/openai_compat.py)
(async variant), tests.

**Acceptance:** with a model that takes 8 s per reply, two messages
arriving 1 s apart finish their replies in roughly 8 s total instead of
16.

---

### P2 — bigger investments, defer until P0/P1 settle

#### 11. HNSW or sqlite-vec for embeddings — **M**
**Why:** the in-memory cosine index is fine for ≤ 10 k items. Past that
recall latency creeps up linearly per query (Python loop over BLOBs).

**How:** option A — bind `sqlite-vec` (single C extension, fast). Option
B — pure-Python HNSW (more code, no extension). Either way `MemoryStore`
gains an `EmbeddingBackend` interface and the cosine path becomes one
implementation among several.

**Acceptance:** benchmark recall over 50 k items completes in < 50 ms
end-to-end (today's flat scan needs ~250 ms at that size).

---

#### 12. Multimodal memory — **L**
**Why:** today image content is stored as a text caption; the original
bytes vanish after the vision call. Re-asking about an old image fails.

**How:** new table `media_blobs(file_id PK, mime, sha256, bytes BLOB,
caption TEXT, created_at)`. Vision module writes here. Memory items
get an optional `media_id` link. Embedding path can call CLIP-style
joint encoders later.

**Acceptance:** "що було на тій фотці тиждень тому?" returns the
correct caption and (optionally) re-renders the image in the admin
dashboard.

---

#### 13. Plan-and-Reflect agent loop — **L**
**Why:** [agent.py](../src/protoagi/agent.py) is a flat tool-use loop
without an explicit planning step. Long tasks tend to wander.

**How:** insert a planning prompt before tool execution that returns
`{plan: [...], step: 1}`; after each tool result, a tiny reflection
prompt updates the plan. Cap planning calls at 2 per run to control
cost.

**Acceptance:** a multi-step task ("read README, propose 3 changes,
write a draft to runs/draft.md") reliably completes within the 8-step
budget instead of looping.

---

#### 14. Multi-instance deployment — **L**
**Why:** running two personas against the same Telegram token is illegal
(409 conflict). Running them with separate tokens but a shared SQLite
file works but couples them.

**How:** add `--db` and `--persona` flags to `protoagi telegram` (#7
backup CLI dovetails). Document a per-persona deployment recipe with
separate `data/<persona>.sqlite3` files. Optional: cross-instance memory
mirror through admin API.

**Acceptance:** two personas (Mykola + Solomiya) run side-by-side on
distinct tokens with distinct memory and never conflict on poll offsets.

---

### P3 — long-term / research

#### 15. Self-tuning reply style — **XL**
The bot watches engagement signals (reply rate, reaction emojis, edits)
and incrementally adjusts reply length / formality / sticker frequency
per chat. Requires a lightweight bandit or Thompson-sampling mechanism
plus a feedback collection step.

#### 16. Memory federation across machines — **XL**
A second ProtoAGI box should be able to subscribe to a curated subset of
memories from the first. Requires an export format, signing, and a sync
protocol (probably Merkle-tree based).

#### 17. Memory graph visualization in admin — **L**
Render `supersedes`/`superseded_by` chains plus tag clusters as an
interactive graph. Force-directed layout in the admin HTML, no
external libraries.

#### 18. Voice — **XL**
Whisper-based transcription for incoming voice messages, TTS for
outgoing replies. Doubles as a multimodal memory feeder.

---

## Decision log

Brief notes on choices made to date so future work can revisit them with
context.

- **Why per-call SQLite connections instead of a single long-lived one?**
  The persistent variant broke `tempfile.TemporaryDirectory` cleanup on
  Windows because the WAL file kept the handle open. Per-call connections
  with WAL pragma set once at init keeps every other property and makes
  tests trivially clean. Revisit only if profiling shows the connect
  overhead matters.
- **Why JSON instead of YAML for personas?** Zero dependencies. Loss is
  minor — persona configs are short and not edited live.
- **Why pure Python cosine instead of `sqlite-vec`?** Single binary
  dependency we'd need to ship per-platform. The current implementation
  is fast enough up to ~10 k facts, which is the experiment-scale we
  target. Item #11 covers the upgrade path when scale demands it.
- **Why a thread-based `BotRunner` instead of asyncio?** Smaller
  surface change, no new dependencies, immediate win for reminder
  latency. Async is item #10 once we have a concrete reason to migrate
  (most likely Tool-use #9).
- **Why heuristic importance instead of LLM-scored?** Determinism in
  tests + zero per-write latency. LLM scoring is item #5 and can be
  toggled when accuracy matters more than latency.
- **Why self-contained FTS5 instead of `content='memory_items'`?** With
  external content, plain `DELETE FROM memory_items_fts WHERE rowid = ?`
  is a no-op against the index — you have to use the
  `INSERT INTO fts(fts, 'delete', ...)` magic command and supply the
  *old* text/tags. `update_memory` would have needed to fetch and pass
  those each call. Self-contained FTS5 costs a small amount of duplicated
  storage but lets DELETE/INSERT work normally. Schema is recreated on
  fresh checkouts; existing experiment DBs keep their old FTS but the
  `try/except` in `_init_db` is a no-op for them.

---

## How to use this file

- Pick the highest-priority unblocked item.
- If it spans multiple files or > 1 day, draft the design in a comment
  on the matching item before coding.
- When done: move the entry to **Done so far**, link the commit, update
  test counts in `AUDIT.md`.
- If a new idea appears mid-flight, add it to the right priority bucket
  immediately rather than relying on memory.
