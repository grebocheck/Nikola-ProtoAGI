# Architecture

ProtoAGI is a local agent harness with explicit layers.

## 1. Inference layer

`llama-server.exe` serves `gpt-oss-20b-MXFP4.gguf` through an
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
`/v1/embeddings`, with a tiny LRU cache and a pure-Python cosine index.

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

## 5. Memory

SQLite is used for durable memory and runs in WAL mode with a single
long-lived connection. The schema is typed:

- `users`: known principals (Telegram user, agent caller).
- `memory_items`: typed entries with `kind` (semantic, episodic, procedural,
  persona_self, fact), `scope` (global, user, chat, persona), `importance`,
  `confidence`, supersession, and access metadata.
- `memory_tags`: normalized tag table indexed for exact matching (no more
  `LIKE '%tag%'` substring confusion).
- `memory_embeddings`: optional float32 BLOBs for semantic recall.
- `memory_items_fts`: FTS5 over text+tags.
- `messages` / `tool_events` / `kv`: agent loop logs and small KV state.
- `telegram_chats` / `telegram_messages`: Telegram-specific state.
- `reminders`: scheduled prompts the bot should surface later.

`MemoryService` is the high-level facade: it scores importance heuristically,
performs hybrid recall (FTS + cosine + recency + importance + pinned bonus),
exposes a heuristic consolidation pass that supersedes near-duplicate
items, and a ``prune()`` pass that forgets low-value items by a blended
``importance × recency × access`` score. Embeddings are optional; when no
embedding endpoint is configured, recall falls back to FTS only.

A small evaluation harness lives in ``protoagi.memory_eval``: it loads a
JSON corpus (``config/memory_eval/golden.json`` by default), plays probe
queries through ``MemoryService.recall``, and reports recall@k and MRR.
``protoagi memory-eval [--with-embeddings]`` runs it end-to-end.

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
- `prompts.py` — system prompt templates
- `bot.py` — `NikolaBot` orchestration with `dispatch_due_reminders` and
  `run_reflection_pass` hooks
- `runner.py` — `BotRunner` runs the long-poll on the main thread and a
  worker thread for initiative, reminder dispatch, and reflection so a
  just-due reminder fires within ~1 s

`protoagi.telegram_bot` remains a thin compatibility shim re-exporting the
public surface so existing imports keep working.

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
