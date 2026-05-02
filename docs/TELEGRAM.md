# Telegram Mode

Telegram mode runs ProtoAGI as a conversation bot with a profile selected from
`.env`. The default profile is `mykola`; `solomiya` is available as a more
self-possessed, warmer profile with separate memory.

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

Only one `getUpdates` polling process can run for a Telegram bot token. If
Telegram returns a 409 conflict, stop the existing local instance:

```powershell
.\stop-nikola.bat
```

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
- long-term memory namespace
- recent conversation thread history
- Telegram message history exposed to the model for intentional replies

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
- The bot uses Telegram reply only when it explicitly refers to the current or
  a recent message.

## Stickers

The bot can use stickers sparingly when they fit the mood. The current sticker
sets are:

- `Bocchi_the_Rock_sticker_pack2`
- `SenkoSan`
- `M1ku_Hatsune`

Sticker file IDs are fetched lazily with `getStickerSet` and cached in SQLite.
The model chooses a pack and optional emoji; the bot picks a matching sticker
from that pack.

## Initiative

The active profile periodically reviews known chats and may send a message first
if there is a good reason. A cooldown prevents spam. Defaults:

- check interval: 300 seconds
- initiative cooldown: 6 hours
- proactive messages are sent silently by default
