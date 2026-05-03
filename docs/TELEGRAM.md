# Telegram Mode

Telegram mode runs ProtoAGI as a conversation bot with a profile selected from
`.env`. The default profile is `mykola`; `solomiya` is available as a more
self-possessed, warmer profile over the same shared Telegram memory.

## Telegram limits

Telegram bots cannot start a brand-new private chat with a user who has never
contacted the bot. The bot can write first only in chats that are already known
to the bot: a user opened `/start`, sent a message, or added the bot to a group.

The implementation uses the official Bot API:

- `getUpdates` long polling
- `sendMessage`
- `sendChatAction`
- `getStickerSet`
- `sendSticker`

## Setup

Create a bot with `@BotFather`, then set the token:

```powershell
$env:TELEGRAM_BOT_TOKEN="123456:ABC..."
```

Or put it in local `.env`:

```env
TELEGRAM_BOT_TOKEN=123456:ABC...
PROTOAGI_TELEGRAM_PERSONA=solomiya
```

Start the local model server first:

```powershell
.\scripts\start-server.ps1 -CtxSize 8192 -CpuMoE 4
```

Run the Telegram bot:

```powershell
.\scripts\start-telegram.ps1
```

Or start the model server and Telegram bot together:

```powershell
.\scripts\start-nikola-stack.ps1
```

When the stack is running, `Ctrl+C` stops the Telegram bot and local
`llama-server` processes. Use `-KeepServers` to leave the model servers loaded
after the bot exits.

Only one `getUpdates` polling process can run for a Telegram bot token. If
Telegram returns a 409 conflict, stop the existing local instance:

```powershell
.\stop-nikola.bat
```

Unexpected Telegram loop exceptions are appended to
`runs\telegram-errors.log`; normal llama.cpp server logs stay in
`runs\llama-server.stderr.log` and `runs\llama-vision.stderr.log`.

If the bot previously used webhooks:

```powershell
.\scripts\start-telegram.ps1 -DeleteWebhook
```

## Access control

For private experiments, restrict the bot to one chat ID:

```powershell
$env:TELEGRAM_ALLOWED_CHAT_IDS="123456789"
.\scripts\start-telegram.ps1
```

You can also pass it directly:

```powershell
.\scripts\start-telegram.ps1 -AllowedChatId 123456789
```

## Profiles

Profiles are not Telegram commands. Set them in `.env` and restart the bot:

```env
PROTOAGI_TELEGRAM_PERSONA=mykola
```

or:

```env
PROTOAGI_TELEGRAM_PERSONA=solomiya
```

The active profile affects:

- displayed identity and name
- aliases used to detect direct address in groups
- system prompts and decision style
- how the bot models the user and the relationship
- how it writes into the shared Telegram memory
- fictional self-memory used to keep a stable persona voice
- per-chat recent conversation thread history
- Telegram message history exposed to the model for intentional replies

## Fictional Self-Memory

For experiments with a more human-feeling chat presence, profiles can keep a
small fictional self-memory: tastes, running jokes, tiny invented habits, and
other harmless details that make the persona consistent over time.

```env
NIKOLA_FICTIONAL_SELF=1
```

When enabled, the decision JSON can include `self_memories`. Those memories are
stored separately from user/chat facts under the active persona, then exposed
back to the model as `known_persona_self_memory`. This lets Соломія invent and
remember small stable details about herself without mixing them into user
memory. The prompt still tells the bot not to claim real physical presence or
offline life; direct questions about being human/a bot get a short honest
answer rather than a support-style disclaimer.

`/start` is still accepted because Telegram uses it to register a chat, but the
bot does not expose a command menu. Ongoing behavior is configured through
`.env` and normal conversation.

## Reply policy

Default mode is `smart`.

- In private chats, the active profile usually responds.
- In groups, the bot responds when addressed by profile alias, username, reply,
  or command-like direct message.
- The model can decide to stay silent when a message does not need a reply.
- Normal answers are sent as plain messages, not Telegram replies.
- In private chats, `reply_to="current"` is ignored because plain messages
  already answer the latest user message.
- The bot uses Telegram reply for group/current replies or explicit recent
  message IDs.

## Message Rhythm

The bot can send one to three short Telegram messages for a single turn when
that feels more natural than one polished paragraph. The decision JSON supports
both a legacy `reply` string and a `replies` array.

The bot can also pair text with stickers more often in light private chats. The
current sticker sets are:

- `Bocchi_the_Rock_sticker_pack2`
- `SenkoSan`
- `M1ku_Hatsune`

Sticker file IDs are fetched lazily with `getStickerSet` and cached in SQLite.
The model chooses a pack and optional emoji; the bot picks a matching sticker
from that pack.
Short generic sticker captions such as "hope this lifted your mood" are
suppressed when a sticker is already being sent.

```env
NIKOLA_STICKER_FREQUENCY=normal
NIKOLA_STICKER_COOLDOWN_MESSAGES=3
TELEGRAM_MAX_REPLY_MESSAGES=3
```

`NIKOLA_STICKER_FREQUENCY` accepts `off`, `low`, `normal`, `high`, or `always`.
The automatic sticker nudge is skipped for serious or heavy topics and only
fires on clear emotional triggers, with a small per-chat cooldown.

## Images

Photo messages and image documents are accepted. If `PROTOAGI_VISION_MODEL` is
configured, the bot downloads the Telegram file, sends it to an
OpenAI-compatible vision endpoint, and includes a short image description in
the conversation context. If vision is not configured, the bot still receives
the image with a neutral `опис недоступний` marker instead of prompting itself
to say "I can't see it."

```env
PROTOAGI_VISION_BASE_URL=http://127.0.0.1:8081/v1
PROTOAGI_VISION_MODEL=smolvlm2-2.2b-instruct
PROTOAGI_VISION_HF_REPO=ggml-org/SmolVLM2-2.2B-Instruct-GGUF:Q4_K_M
```

`run-nikola.bat` starts this local vision server automatically when
`PROTOAGI_VISION_MODEL` points to a localhost URL. The first start downloads the
model into the llama.cpp/Hugging Face cache. Use `scripts\start-vision-server.ps1`
directly if you want to warm the cache before starting Telegram.
Vision requests include the llama.cpp multimodal marker internally so uploaded
Telegram images are paired with the image bytes instead of failing tokenization.

Incoming stickers are also treated as conversational messages with emoji and
pack metadata, so the model can react to them instead of ignoring them.

## Initiative

The active profile periodically reviews known chats and may send a message first
if there is a good reason. A cooldown prevents spam. Defaults:

- check interval: 300 seconds
- initiative cooldown: 6 hours
- proactive messages are sent silently by default

## Reminders

The agent has a `remind_me` tool that creates rows in the `reminders` table.
When the Telegram bot is running, `BotRunner`'s worker thread checks for due
reminders every minute and delivers them into the originating chat with a
``⏰`` prefix. Reminders without a deliverable chat are marked ``cancelled``
to avoid retry loops.

## Reflection

Every ~6 hours the bot runs a maintenance pass:

- consolidates near-duplicate memories in the `global` and active `persona`
  scopes (older items are marked `superseded_by` the winner);
- when fictional self-memory is enabled, asks the model for 1-2 short
  first-person reflection notes built from recent user facts and existing
  self-memories. They are stored as ordinary `persona_self` memories so they
  surface again in normal recall.

The reflection cadence is tracked in the `kv` table under
``telegram:last_reflection_at`` so a freshly restarted bot doesn't run the
pass twice in a row.

## Background workers

`BotRunner` is the default runtime: it runs the long-poll on the main thread
and a small worker thread for periodic tasks (initiative, reminders,
reflection). Pass ``--single-thread`` to ``protoagi telegram`` to fall back
to the legacy interleaved loop, mostly useful for step-through debugging.
