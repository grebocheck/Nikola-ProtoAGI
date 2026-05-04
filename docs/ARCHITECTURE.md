# Architecture

ProtoAGI is a local agent harness with explicit layers.

## 1. Inference layer

`llama-server.exe` serves `models/gpt-oss-20b-MXFP4.gguf` through an
OpenAI-compatible API at `http://127.0.0.1:8080/v1`.

The default runtime profile uses:

- 8k context
- Flash Attention on
- Jinja chat templates on
- partial MoE CPU offload with `--n-cpu-moe 4`

This is chosen because the model and compute buffers are close to the 16 GB VRAM
limit.

An optional second `llama-server` instance can host an embedding model
(`bge-m3`, `nomic-embed-text`, ...) and is consumed through the standard
`/v1/embeddings` endpoint. Memory recall degrades gracefully when no
embedding server is configured.

## 2. Client layer

`protoagi.openai_compat` is a small dependency-free HTTP client. It supports:

- `/v1/models`
- `/v1/chat/completions`
- OpenAI-style tool definitions

`protoagi.embedding` is a small dependency-free embedding client over
`/v1/embeddings`, with a tiny LRU cache and pure-Python exact/LSH indexes. It
can also ask a joint image/text embedding endpoint for media vectors by sending
stored image bytes as base64; unsupported endpoints simply fall back to text
embeddings.

## 3. Agent loop

`ProtoAgent` runs:

1. build context from system prompt, hybrid memory recall, and recent thread
   messages
2. wrap user input in `<user_input>...</user_input>` markers so the model
   treats it as data, not as instructions
3. call model with the registered tool schemas
4. execute tool calls
5. feed observations back to the model
6. stop on a final answer or step limit

## 4. Tools

The first tool set is intentionally practical:

- memory: `remember`, `recall`
- workspace: `list_dir`, `read_file`, `write_file`, `append_file`, `search_workspace`
- environment: `now`, `gpu_status`
- execution: `run_powershell`
- network: `web_get` (with a public-URL SSRF filter)
- reminders: `remind_me`, `list_reminders`

The shell tool is policy-gated, blocks common destructive patterns by default,
and refuses any URL that resolves to loopback / private / link-local /
multicast / reserved IP space.

The registry implementation lives in `protoagi.tools_core`; `protoagi.tools`
is kept as a small compatibility facade for existing imports and tests.

## 5. Memory

SQLite is used for durable memory. WAL mode is enabled once at init, then
operations use short-lived per-call connections. The schema is typed:

- `users`: known principals (Telegram user, agent caller).
- `memory_items`: typed entries with `kind` (semantic, episodic, procedural,
  persona_self, fact), `scope` (global, user, chat, persona), `importance`,
  `confidence`, supersession, and access metadata.
- `memory_tags`: normalized tag table indexed for exact matching (no more
  `LIKE '%tag%'` substring confusion).
- `memory_embeddings`: optional float32 BLOBs for semantic recall.
- `media_blobs`: Telegram image/voice bytes linked from `memory_items.media_id`.
- `importance_cache`: bounded cache for optional LLM importance scoring.
- `memory_items_fts`: FTS5 over text+tags.
- `messages` / `tool_events` / `kv`: agent loop logs and small KV state.
- `telegram_chats` / `telegram_messages`: Telegram-specific state.
- `reminders`: scheduled prompts the bot should surface later.

Storage types and vector helpers live in `protoagi.storage.models`; the SQLite
store implementation lives in `protoagi.storage.memory`. `protoagi.memory`
remains a compatibility facade.

`MemoryService` is the high-level facade: it scores importance heuristically,
performs hybrid recall (FTS + cosine + recency + importance + pinned bonus,
plus a small media-aware cosine boost for image-linked items), exposes a
heuristic consolidation pass that supersedes near-duplicate items, and a
``prune()`` pass that forgets low-value items by a blended
``importance × recency × access`` score. Embeddings are optional; when no
embedding endpoint is configured, recall falls back to FTS only.

A small evaluation harness lives in ``protoagi.memory_eval``: it loads a
JSON corpus (``config/memory_eval/golden.json`` by default), plays probe
queries through ``MemoryService.recall``, and reports recall@k, MRR, and
per-section subscores for friendly, contradiction, negative, paraphrase, and
media-caption probes. ``protoagi memory-eval [--with-embeddings]`` runs it
end-to-end.

`protoagi.memory_federation` exports curated active memories as HMAC-signed
JSON bundles and imports them idempotently on another machine with
`protoagi memory-export` / `protoagi memory-import`. Full exports store a
source/filter manifest in `kv`; `memory-export --since <iso>` emits only
new/changed rows plus deletion tombstones keyed by `federation_id`.

## 6. Evaluation

The system includes two benchmark paths:

- `llama-bench` for raw runtime profiles
- endpoint benchmark for real chat latency

Future evals should add task suites for:

- file editing
- planning
- tool use accuracy
- memory retention
- long-context summarization
- self-correction

## 7. Telegram Layer

The Telegram bot lives in the `protoagi.telegram` package and decomposes the
old monolith into focused units:

- `api.py` — Telegram Bot API transport
- `config.py` — env-loaded configuration
- `text.py` / `json_io.py` — text and decision payload helpers (with
  `DECISION_JSON_SCHEMA` / `INITIATIVE_JSON_SCHEMA` forwarded to the model
  as `response_format` so decisions arrive as well-formed JSON)
- `stickers.py` / `vision.py` / `identity.py` — narrow concerns extracted
  for testing and iteration
- `voice.py` — optional Telegram voice transcription and TTS helpers
- `style.py` — per-chat reply-style tuning from lightweight engagement
  signals
- `attachments.py` / `sticker_ops.py` - incoming media extraction and sticker
  pack caching/selection
- `prompts.py` — system prompt templates
- `orchestrator.py` — `NikolaBot` orchestration with `dispatch_due_reminders`
  and `run_reflection_pass` hooks
- `bot.py` — compatibility facade for the historical import path
- `runner.py` — `BotRunner` runs the long-poll on the main thread and a
  worker thread for initiative, reminder dispatch, and reflection so a
  just-due reminder fires within ~1 s

`protoagi.telegram_bot` remains a thin compatibility shim re-exporting the
public surface so existing imports keep working.

`NikolaBot` now lives in `protoagi.telegram.orchestrator`; `protoagi.telegram.bot`
is a compatibility facade. Incoming attachment extraction and sticker-pack
selection/caching are split into `attachments.py` and `sticker_ops.py`.

The bot uses a profile selected through `.env`:

- `mykola` — calm, grounded, practical
- `solomiya` — warmer, self-possessed, more relational

Profiles are loaded from `config/personas/*.json` (with hard-coded fallbacks)
so new identities can be added without touching Python.

The bot deliberately uses a narrower surface than the workspace agent:

- official Bot API long polling with `getUpdates`
- text sending with `sendMessage`
- typing indicator with `sendChatAction`
- sticker set discovery with `getStickerSet`
- sticker sending with `sendSticker`
- optional voice/audio transcription and optional TTS voice replies
- optional persistence of incoming voice/audio bytes alongside transcripts
- shared Telegram long-term facts in SQLite, recalled through `MemoryService`
- per-chat recent thread history for local dialogue context
- Telegram message ID history for intentional replies
- reply policy: `smart`, `always`, `mention`, or `silent`
- bounded proactive messages for chats that already know the bot

Telegram does not allow a bot to open a brand-new private chat by itself. The
initiative loop therefore only works for chats that previously contacted the bot
or added it.

Profiles are intentionally deeper than a display name. The active profile
changes the system prompt, aliases, self-model, user model, relationship stance,
and memory policy. Telegram facts are shared through the global Telegram memory,
while compact dialogue history and Telegram message IDs stay per chat so style
and reply targeting do not bleed between conversations.

## 8. Admin and source organization

The local admin dashboard is split between `protoagi.admin_server` for HTTP
handling and HTML, and `protoagi.admin_data` for stats, style reports, memory
serialization, and graph payloads. `protoagi.admin` remains a compatibility
facade.

Model weights are stored under `models/` and ignored by git except for
`models/.gitkeep`. Source modules that moved during cleanup keep lightweight
facades (`protoagi.memory`, `protoagi.tools`, `protoagi.admin`,
`protoagi.telegram.bot`) so external callers do not need an immediate import
migration.
