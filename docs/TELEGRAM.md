# Telegram Mode

Telegram mode runs ProtoAGI as a conversation bot named Микола.

## Telegram limits

Telegram bots cannot start a brand-new private chat with a user who has never
contacted the bot. Микола can write first only in chats that are already known
to the bot: a user opened `/start`, sent a message, or added the bot to a group.

The implementation uses the official Bot API:

- `getUpdates` long polling
- `sendMessage`
- `sendChatAction`

## Setup

Create a bot with `@BotFather`, then set the token:

```powershell
$env:TELEGRAM_BOT_TOKEN="123456:ABC..."
```

Or put it in local `.env`:

```env
TELEGRAM_BOT_TOKEN=123456:ABC...
```

Start the local model server first:

```powershell
.\scripts\start-server.ps1 -CtxSize 8192 -CpuMoE 4
```

Run Микола:

```powershell
.\scripts\start-telegram.ps1
```

Or start the model server and Telegram bot together:

```powershell
.\scripts\start-nikola-stack.ps1
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

## Commands

- `/start` - register the chat
- `/help` - show commands
- `/remember text` - store a chat-scoped memory
- `/recall query` - search chat memory
- `/quiet` - disable proactive messages
- `/wake` - enable proactive messages
- `/mode smart|always|mention|silent` - change reply policy
- `/status` - show chat state

## Reply policy

Default mode is `smart`.

- In private chats, Микола usually responds.
- In groups, he responds when addressed by name, username, reply, or command.
- The model can decide to stay silent when a message does not need a reply.

## Initiative

Микола periodically reviews known chats and may send a message first if there is
a good reason. A cooldown prevents spam. Defaults:

- check interval: 300 seconds
- initiative cooldown: 6 hours
- proactive messages are sent silently by default
