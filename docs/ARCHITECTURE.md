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

## 2. Client layer

`protoagi.openai_compat` is a small dependency-free HTTP client. It supports:

- `/v1/models`
- `/v1/chat/completions`
- OpenAI-style tool definitions

## 3. Agent loop

`ProtoAgent` runs:

1. build context from system prompt, durable memory, and recent thread messages
2. call model
3. execute tool calls
4. feed observations back to the model
5. stop on a final answer or step limit

## 4. Tools

The first tool set is intentionally practical:

- memory: `remember`, `recall`
- workspace: `list_dir`, `read_file`, `write_file`, `append_file`, `search_workspace`
- environment: `now`, `gpu_status`
- execution: `run_powershell`
- network: `web_get`

The shell tool is policy-gated and blocks common destructive patterns by default.

## 5. Memory

SQLite is used for durable memory:

- `facts` for long-term facts
- `messages` for thread history
- `tool_events` for auditability
- optional FTS5 table for fast recall

Embeddings are intentionally deferred. SQLite FTS gives us a transparent and
cheap baseline that works on limited hardware.

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

`protoagi.telegram_bot` runs a Telegram personality named Микола.

It deliberately uses a narrower surface than the workspace agent:

- official Bot API long polling with `getUpdates`
- text sending with `sendMessage`
- typing indicator with `sendChatAction`
- chat-scoped memory in SQLite
- reply policy: `smart`, `always`, `mention`, or `silent`
- bounded proactive messages for chats that already know the bot

Telegram does not allow a bot to open a brand-new private chat by itself. The
initiative loop therefore only works for chats that previously contacted the bot
or added it.
