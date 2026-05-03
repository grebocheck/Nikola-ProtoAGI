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

### Phase 5 — P1 finished (2026-05-03)
- **P1-5** LLM-driven importance scoring is opt-in via
  `PROTOAGI_LLM_IMPORTANCE=1`. `MemoryService.score_importance_llm`
  asks for `{importance, kind, reasoning}`, caches by SHA256 of normalized
  text in `kv`, and falls back to the deterministic heuristic.
- **P1-6** Telegram memory can be isolated per user with
  `PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0`; chat facts are stored under
  `scope=user` when a Telegram sender is known, and recall passes the
  current `user_id`.
- **P1-7** `protoagi backup` / `protoagi restore` use SQLite's online
  backup API, validate with `PRAGMA integrity_check`, and restore through
  an atomic swap plus WAL/SHM cleanup.
- **P1-8** `memory-prune` and `memory-consolidate` can emit dry-run JSON
  plans with per-item kept/dropped reasons. Admin exposes preview endpoints
  for both operations.
- **P1-9** Telegram decisions can request bounded tools (`recall`,
  `remind_me`) via `tool_request` or OpenAI-style `tool_calls`; results are
  merged back into the final decision.
- **P1-10** `AsyncBotRunner` adds opt-in concurrent Telegram update
  handling through `asyncio.to_thread` and a semaphore; CLI flag:
  `protoagi telegram --async`.

Test count: 107 → 116.

### Phase 6 — P2 finished (2026-05-03)
- **P2-11** Embedding search now has a backend boundary:
  `EmbeddingBackend`, exact `FlatEmbeddingBackend`, and pure-Python
  `LSHEmbeddingBackend` selectable with `PROTOAGI_EMBED_BACKEND=lsh`
  (or `hnsw`/`auto` alias). Exact cosine remains the default.
- **P2-12** Multimodal memory landed: `media_blobs(file_id, mime, sha256,
  bytes, caption, created_at)`, optional `memory_items.media_id`, admin
  `GET /api/media/<id>`, and Telegram image messages persist bytes +
  caption-linked memory items.
- **P2-13** `ProtoAgent` now runs a bounded Plan-and-Reflect loop:
  an initial JSON plan plus at most one post-tool plan update by default
  (`PROTOAGI_PLAN_CALL_LIMIT=2`). Tool execution remains capped by the
  normal `max_steps`.
- **P2-14** Telegram supports multi-instance deployment flags:
  `protoagi telegram --db data/<persona>.sqlite3 --persona <key> --token ...`.
  The startup line prints the active database path.

Test count: 116 → 121.

### Phase 7 — follow-up audit cleanup (2026-05-03)
Tightened a few things uncovered by re-reading P1+P2 work:

- ``MemoryService._scope_matches`` had a dead branch that re-checked
  ``item.scope == SCOPE_USER`` after an early return — removed for
  clarity. Behavior unchanged.
- ``VisionDescriber.describe`` had unreachable English-fallback text and
  a confusing double-fallback chain; collapsed to a single Ukrainian
  fallback so logs and recall stay in one language.
- ``VisionDescriber._store_media`` was silently swallowing every
  exception. It now narrows to ``sqlite3.Error / OSError / ValueError``
  and prints the error to the runtime log so persistence failures stop
  being invisible.
- ``embedding.py`` carried an unused ``import sqlite3``; removed.
- ``MemoryStore`` docstring said "single long-lived connection" but the
  module switched to per-call connections with WAL-set-once-at-init for
  Windows compatibility — docstring corrected.
- ``AsyncBotRunner.poll_once`` now passes
  ``return_exceptions=True`` to ``asyncio.gather`` and logs per-update
  failures instead of cancelling the rest of the batch. Permanent
  failures still don't busy-loop; transient replay semantics shipped later
  in Phase 8 (item A6).

No behavior tests changed. 121 tests still green.

Bugs found by this audit became items A1–A8 and are fixed in Phase 8:

- **A1** Privacy-mode flag silently hides legacy global rows.
- **A2** Tool-augmented decisions cost up to 3 LLM calls.
- **A3** ``tools`` + ``response_format`` interaction not measured.
- **A4** ``media_blobs`` has no garbage collection.
- **A5** SSRF guard is vulnerable to DNS rebinding.
- **A6** Async runner advances offset on transient failure.
- **A7** Importance cache in ``kv`` has no eviction.
- **A8** ``EmbeddingClient`` cache evicts FIFO instead of LRU.

### Phase 8 — series A audit fixes (2026-05-03)
Closed the A1–A8 follow-up cohort:

- **A1** added `protoagi memory-rescope --to user`, backed by
  `MemoryStore.rescope_telegram_memories`, and documented the privacy-mode
  migration in [docs/TELEGRAM.md](TELEGRAM.md).
- **A2** inlines trivial `recall` tool answers, avoiding the merge LLM call
  on simple "what do you remember?" paths, and records per-decision LLM-call
  histograms / averages in admin stats.
- **A3** added `protoagi bench-tools`, which reports whether a local model
  emitted `tool_calls`, `tool_request`, both, or neither when both
  `tools` and `response_format` are set.
- **A4** added `MemoryStore.prune_orphan_media(older_than_days=...)` and
  runs it during reflection.
- **A5** hardened `web_get` against DNS rebinding by resolving once,
  validating the resolved IP, then opening the HTTP socket to that same IP.
- **A6** changed async polling offset handling so failed updates are not
  acknowledged past Telegram's offset cursor and can be replayed.
- **A7** moved LLM importance scoring cache out of `kv` into
  `importance_cache` with pruning and admin row-count stats.
- **A8** replaced FIFO embedding cache eviction with true LRU via
  `OrderedDict`.

### Phase 9 — P3 research baselines (2026-05-03)
Turned each P3 research direction into a dependency-free first working layer:

- **P3-15** self-tuning reply style: a per-chat UCB-style tuner stores
  engagement signals in SQLite `kv`, learns from replies/reactions/edits,
  and injects an `adaptive_reply_style` hint into decision/reply/initiative
  prompts.
- **P3-16** memory federation: `protoagi memory-export` and
  `protoagi memory-import` move curated memory bundles between machines as
  HMAC-signed JSON, with idempotent imports through federation-id tags.
- **P3-17** admin memory graph: `/api/memory-graph` exposes memory/tag
  nodes plus tag and supersession edges; the admin dashboard renders a small
  no-dependency force-directed canvas graph.
- **P3-18** voice: Telegram voice/audio messages can be transcribed through
  an OpenAI-compatible `/audio/transcriptions` endpoint and stored as
  episodic voice memory. Optional TTS sends a generated Telegram voice reply
  after the normal text reply.

---

## Backlog

Ordered by priority. Reorder freely as new evidence arrives.

### Original P0 — done (see Phase 4 above)

All four original P0 items shipped on 2026-05-03:
- ✅ #1 Embedding server in stack
- ✅ #2 Auto-prune in `run_reflection_pass`
- ✅ #3 Admin pin/unpin + edit
- ✅ #4 Memory eval regression gate

The follow-up audit added a fresh P0 cohort (A1–A8); all shipped in Phase 8.

### P1 — done (see Phase 5 above)

All six P1 items shipped on 2026-05-03:
- ✅ #5 LLM-driven importance scoring with cache
- ✅ #6 Per-user privacy scopes
- ✅ #7 Backup / restore CLI
- ✅ #8 Memory diff for `consolidate` / `prune`
- ✅ #9 Tool-use inside Telegram persona
- ✅ #10 Async polling

### P2 — done (see Phase 6 above)

All four P2 items shipped on 2026-05-03:
- ✅ #11 HNSW/approximate embedding backend path
- ✅ #12 Multimodal memory
- ✅ #13 Plan-and-Reflect agent loop
- ✅ #14 Multi-instance deployment

### P3 — baseline shipped (see Phase 9 above)

All four P3 research baselines shipped on 2026-05-03:
- ✅ #15 Self-tuning reply style
- ✅ #16 Memory federation across machines
- ✅ #17 Memory graph visualization in admin
- ✅ #18 Voice

### P0 — series A shipped from the 2026-05-03 follow-up audit

All A-series items below shipped in Phase 8; details remain here as the
historical acceptance record.

#### A1. Privacy mode loses access to legacy global memories — **S**
**What:** when `PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0`, `_search_chat_memory`
sets `include_global=False`. Existing rows (written before the flag flipped)
have `scope=SCOPE_GLOBAL`, so they become invisible. SCOPE_USER rows
written during privacy mode also become invisible to non-private callers
(``query.user_id=None``).

**Why:** silent recall regression for users who toggle the flag mid-stream;
hard to debug without reading the scope-matcher.

**How:** add a one-shot ``protoagi memory-rescope --to user`` command that
reassigns ``scope`` and ``user_id`` based on existing
``user:<id>``/``source_chat:<id>`` tags. Document the trade-off in
[docs/TELEGRAM.md](TELEGRAM.md). Optionally support a "permissive
private" mode that still recalls SCOPE_GLOBAL items but never writes new
ones.

**Acceptance:** existing test still passes; new rescoping CLI test
verifies a global → user migration.

---

#### A2. Tool-augmented decisions cost three LLM calls — **M**
**What:** when the model emits a `tool_request` or `tool_calls`,
`decide_incoming` runs (call #1), the tool runner executes, and
`_merge_decision_tool_results` runs another full chat completion (#2).
If `compose_reply` then triggers (no `reply` text yet), that's #3. On a
20B local model this is ~30 s end-to-end.

**Why:** noticeable latency for what should feel like a quick chat
reaction.

**How:** option A — make the merge call optional when the tool result is
trivially formattable (skip merge for `recall` with one strong hit;
inline the text directly). Option B — short-circuit `compose_reply` when
the merge call already produced `reply`/`replies`. Item already partially
mitigated; needs a measurement: log per-decision call counts and surface
in admin stats.

**Acceptance:** P95 latency for a simple "що ти памʼятаєш?" path drops
below 2× a no-tool reply on the local model.

---

#### A3. `tools` + `response_format` combined behavior is unverified — **S**
**What:** `decide_incoming` passes both `tools=TelegramToolRunner.schemas()`
and `response_format=DECISION_JSON_SCHEMA`. llama.cpp's behavior with both
set is model-specific: the gpt-oss-20b harness usually picks one path
(text content matching the schema) and ignores the tool calls. This means
the carefully-typed `tool_calls` branch in `_merge_decision_tool_results`
may be reached far less often than the `tool_request` branch.

**Why:** silently degraded tool coverage; tests cover both branches but
real-world ratios are unknown.

**How:** add a small bench that issues a "use the recall tool to find …"
prompt and reports whether the model emitted `tool_calls`, `tool_request`,
both, or neither. Pin the choice we trust as the canonical path; consider
dropping the unused branch.

**Acceptance:** `protoagi bench-tools` writes a counts report; ROADMAP
decision log records which branch is the production path.

---

#### A4. Vision blob persistence has no garbage collection — **S**
**What:** `media_blobs` rows accumulate forever; the only deletion is via
`memory_items` cascade, but `_remember_media_fact` only writes the memory
when the description is non-empty. Images received without a vision model
configured persist as blobs with no linked memory item, so they are never
cleaned up.

**Why:** SQLite database can grow unbounded with binary content even
during ordinary use.

**How:** add a `MemoryStore.prune_orphan_media(older_than_days=…)`
method. Wire it into `run_reflection_pass` next to the existing prune
step. Surface counts.

**Acceptance:** unit test verifies a 60-day-old orphan blob is removed
while a recently-linked one survives.

---

#### A5. `_validate_public_url` is vulnerable to DNS rebinding — **S**
**What:** the SSRF guard resolves DNS once, validates IPs, then
`urlopen` resolves DNS again. A malicious server can advertise a public
IP for the validation lookup and a private IP for the actual fetch.

**Why:** the experimental risk is low (no privileged services on the
host except llama-server on 127.0.0.1, which is the very thing we are
trying to protect). Still, the guard reads as if it were robust.

**How:** open a TCP socket to one of the validated IPs ourselves
(``socket.create_connection`` with explicit address), then issue the
HTTP request through that socket. Or document the limitation and
restrict `web_get` to an allowlist of hosts.

**Acceptance:** new test exercises the rebinding scenario with a fake
DNS hook.

---

#### A6. Async runner advances offset on transient failures — **S**
**What:** ``poll_once`` now uses ``return_exceptions=True`` and advances
the offset to ``max_update_id + 1`` regardless of failure. Permanent
failures are skipped, but transient failures (network blip, llama-server
restart) can still lose updates because we never retry the same id.

**Why:** synchronous polling reprocesses on transient failure (offset
advances per id in a try/finally inside the loop).

**How:** track failed update_ids; if all updates in a batch failed, do
not advance the offset. If some succeeded, only acknowledge the contiguous
successful prefix before the first failed id; advancing past a failed id
would ask Telegram to drop it. Later successful ids can replay once, which
is safer than losing an update.

**Acceptance:** new async test simulates a one-shot transient error and
confirms the failed update is replayed on the next poll.

---

#### A7. Importance cache is forever-pinned in `kv` — **S**
**What:** `score_importance_llm` writes scores into the `kv` table keyed
by SHA256. There is no eviction policy, so a long-running bot accumulates
millions of entries.

**Why:** unbounded growth in a table that was meant for small singleton
state (offsets, last-reflection-at, sticker pack caches).

**How:** move importance cache to a dedicated `importance_cache` table
with `created_at` and a periodic prune (e.g. drop entries older than
30 days, or LRU on count). Or store it in `memory_items.metadata` so the
cache lives with the memory itself.

**Acceptance:** stats endpoint reports importance-cache row count
separately; reflection pass evicts old entries.

---

#### A8. Embedding cache evicts on insertion order, not LRU — **XS**
**What:** ``EmbeddingClient._cache`` is keyed by text; eviction pops the
oldest *inserted* key, not the least *recently used*. Hot facts can drop
out while cold ones linger.

**Why:** small but real perf hit when the workload reuses the same query
strings.

**How:** swap the list-based eviction for an `OrderedDict` with
`move_to_end` on hit.

**Acceptance:** unit test inserts N+1 items with one repeat hit and
verifies the repeated text is preserved.

---

### P3 — shipped research baselines

These are no longer blank research items; each now has a working baseline.
Future work should extend the baseline rather than start from scratch.

#### 15. Self-tuning reply style — **XL**
The bot watches engagement signals (reply rate, reaction emojis, edits)
and incrementally adjusts reply length / formality / sticker frequency
per chat. Requires a lightweight bandit or Thompson-sampling mechanism
plus a feedback collection step.

**Shipped baseline:** `ReplyStyleTuner` stores per-chat arm stats in `kv`,
records replies/reactions/edits, and passes `adaptive_reply_style` into the
Telegram prompts.

#### 16. Memory federation across machines — **XL**
A second ProtoAGI box should be able to subscribe to a curated subset of
memories from the first. Requires an export format, signing, and a sync
protocol (probably Merkle-tree based).

**Shipped baseline:** signed JSON bundles through `memory-export` /
`memory-import`. Merkle subscriptions remain future work.

#### 17. Memory graph visualization in admin — **L**
Render `supersedes`/`superseded_by` chains plus tag clusters as an
interactive graph. Force-directed layout in the admin HTML, no
external libraries.

**Shipped baseline:** `/api/memory-graph` plus a force-directed canvas in the
admin dashboard.

#### 18. Voice — **XL**
Whisper-based transcription for incoming voice messages, TTS for
outgoing replies. Doubles as a multimodal memory feeder.

**Shipped baseline:** optional OpenAI-compatible transcription + optional
`/audio/speech` TTS-to-Telegram voice replies.

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
- **Why flat cosine by default even after the LSH backend landed?** Exact
  ranking is simplest and deterministic for small stores. `lsh` is now an
  opt-in pure-Python backend for larger stores without taking a binary
  `sqlite-vec` dependency.
- **Why keep thread-based `BotRunner` after async landed?** It remains the
  conservative default with the smallest operational surface. `--async` is
  available when concurrent update handling matters more than simplicity.
- **Why heuristic importance by default after LLM scoring landed?**
  Determinism and zero per-write latency still make sense for local dev.
  `PROTOAGI_LLM_IMPORTANCE=1` flips on the cached model scorer when quality
  matters more.
- **Why self-contained FTS5 instead of `content='memory_items'`?** With
  external content, plain `DELETE FROM memory_items_fts WHERE rowid = ?`
  is a no-op against the index — you have to use the
  `INSERT INTO fts(fts, 'delete', ...)` magic command and supply the
  *old* text/tags. `update_memory` would have needed to fetch and pass
  those each call. Self-contained FTS5 costs a small amount of duplicated
  storage but lets DELETE/INSERT work normally. Schema is recreated on
  fresh checkouts; existing experiment DBs keep their old FTS but the
  `try/except` in `_init_db` is a no-op for them.
- **Why keep two parallel tool-call paths in `decide_incoming` instead of
  picking one?** The model emits either `tool_calls` (OpenAI-style) or
  `tool_request` inside the JSON body, depending on whether llama.cpp
  honored `tools=` or only `response_format=`. We don't yet know which
  path fires in production with gpt-oss-20b. Phase 8 added
  `protoagi bench-tools` so the branch split can be measured before
  dropping either path.
- **After `bench-tools`, which tool path is canonical?** Keep
  `tool_request` as the production-compatible path for now because it works
  inside the schema content even when llama.cpp ignores native
  OpenAI-style tool calls. `protoagi bench-tools` now reports the real
  split for the current local model; if `tool_calls` dominates in a
  measured run, revisit the branch choice with data.
- **Why log via `print` from the async runner / vision module instead of
  using the runs/telegram-errors.log file?** Those modules are imported
  from places without easy access to `NikolaBot.error_log_path`. The
  current pattern is: catch known exception types, print to stdout/stderr
  so the launcher captures it, never re-raise unless the failure should
  abort the loop. A small `protoagi.log` shim with rotation could
  centralize this — currently out of scope.

---

## How to use this file

- Pick the highest-priority unblocked item.
- If it spans multiple files or > 1 day, draft the design in a comment
  on the matching item before coding.
- When done: move the entry to **Done so far**, link the commit, update
  test counts in `AUDIT.md`.
- If a new idea appears mid-flight, add it to the right priority bucket
  immediately rather than relying on memory.
