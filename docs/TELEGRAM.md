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

For side-by-side personas, run separate Telegram bot tokens and separate
SQLite databases:

```powershell
$env:PYTHONPATH="src"
python -m protoagi telegram --persona mykola --db data/mykola.sqlite3 --token "123:AAA"
python -m protoagi telegram --persona solomiya --db data/solomiya.sqlite3 --token "456:BBB"
```

Each database keeps its own poll offset, chat table, reminders, and memory, so
instances do not fight over Telegram `getUpdates` state.

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

By default Telegram memory is global, which is convenient for a single-owner
bot that moves between private chats, groups, and personas. For multi-user
deployments, switch to per-user memory isolation:

```env
PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0
```

With that flag off, remembered Telegram facts are stored in `scope=user` when
the sender is known, and recall only returns facts for the current Telegram
user. Chat-scoped system facts remain available inside the originating chat.

If you previously ran the bot with global Telegram memory and then switch
`PROTOAGI_TELEGRAM_GLOBAL_MEMORY=0`, legacy rows can be rescoped by
calling `MemoryStore.rescope_telegram_memories(to_scope='user')` from a
Python shell. The migration only touches rows that still have
`scope=global` and a `user:<id>` tag — it copies `source_chat:<id>`
into `chat_id` when available, so future privacy-mode recall keeps the
original Telegram context.

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

The bot uses stickers sparingly: most replies are text-only. The current
sticker sets are:

- `Bocchi_the_Rock_sticker_pack2`
- `SenkoSan`
- `M1ku_Hatsune`

Sticker file IDs are fetched lazily with `getStickerSet` and cached in SQLite.
The model chooses a pack and optional emoji; the bot picks a matching sticker
from that pack.
Short generic sticker captions such as "hope this lifted your mood" are
suppressed when a sticker is already being sent.

```env
NIKOLA_STICKER_FREQUENCY=low
NIKOLA_STICKER_COOLDOWN_MESSAGES=6
NIKOLA_STICKER_MAX_REPLY_CHARS=180
NIKOLA_STICKER_INITIATIVE=0
TELEGRAM_MAX_REPLY_MESSAGES=3
```

`NIKOLA_STICKER_FREQUENCY` accepts `off`, `low`, `normal`, `high`, or
`always`. The default is `low`. Beyond the percentage cap, stickers are
filtered out — for both LLM-emitted and auto-reaction paths — when:

- the topic is serious / heavy (legal, medical, conflict, anxiety);
- the reply text is longer than `NIKOLA_STICKER_MAX_REPLY_CHARS`
  (a sticker rarely fits a paragraph-shaped human thought);
- the bot already sent a sticker within the last
  `NIKOLA_STICKER_COOLDOWN_MESSAGES` user messages
  (no two stickers in a row);
- the message is a proactive *initiative*, unless
  `NIKOLA_STICKER_INITIATIVE=1` is set;
- the per-chat style tuner currently leans toward the `concise` arm.

The auto-reaction trigger words are intentionally narrow: explicit
laughter (`ахах`, `lol`, `🤣`/`😂`), explicit warmth (`❤`, `🤗`,
`обнімаю`, `дякую тобі`), or unambiguous gameplay context. Single
neutral nouns like `чай`, `кава`, or `грати` no longer fire on their
own.

The bot also keeps a small per-chat style tuner in SQLite. Replies, reaction
updates, and edited messages are treated as lightweight engagement signals; the
next decision prompt receives a soft `adaptive_reply_style` hint for reply
length, formality, and sticker frequency. The hint is advisory only and is not
shown to the chat.

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
Animated GIFs use Telegram's still-frame thumbnail when one is available. If a
GIF arrives as a raw `image/gif` document without a thumbnail, the bot stores
the original media and, when local `ffmpeg` is available, extracts the first
JPEG frame for the vision LLM. If no frame can be extracted, the current turn
gets a neutral GIF marker but no long-term media memory is written from an
`опис недоступний` placeholder.

Incoming stickers are also treated as conversational messages with emoji and
pack metadata, so the model can react to them instead of ignoring them.

Image bytes are stored in `media_blobs` with their caption and linked from a
memory item via `media_id`. This lets later recall surface old photo captions;
the admin API can serve the original bytes at `/api/media/<file_id>`.

## Voice

Voice messages and Telegram audio messages are accepted. If
`PROTOAGI_VOICE_MODEL` is configured, the bot downloads the Telegram audio file,
sends it to an OpenAI-compatible `/audio/transcriptions` endpoint, and includes
the transcript in the conversation context. Successful transcripts are stored
as episodic voice memory. By default the original OGG/audio bytes are also
stored in `media_blobs` and linked from the voice memory via `media_id`, so a
future transcription model can reprocess the same audio without depending on
Telegram file retention. Set `PROTOAGI_STORE_VOICE=0` to keep transcript-only
behavior.

```env
PROTOAGI_VOICE_BASE_URL=http://127.0.0.1:8083/v1
PROTOAGI_VOICE_MODEL=whisper-large-v3
PROTOAGI_STORE_VOICE=1
```

Outgoing TTS is opt-in. By default, when enabled, `PROTOAGI_TTS_DELIVERY=auto`
still prefers normal text. The model can request a rare short voice reply with
`voice_reply=true`, and local guards only allow it in private chats when the
reply is short, sticker-free, and outside the cooldown window. If synthesis or
Telegram delivery fails, it falls back to the normal text reply. Set
`PROTOAGI_TTS_DELIVERY=voice` to force audio instead of text, or
`PROTOAGI_TTS_DELIVERY=text_and_voice` only when you explicitly want both a
text transcript and a voice/audio attachment. Set `PROTOAGI_TTS_DELIVERY=text`
to keep replies text-only even when TTS is enabled. Each persona can pick its
own voice through the `tts_voice` field in
`config/personas/*.json` (built-ins: `solomiya`→`solomiya`, `mykola`→`mykola`);
`PROTOAGI_TTS_VOICE` is only the fallback when a persona does not set one.

```env
PROTOAGI_TTS_ENABLED=1
PROTOAGI_TTS_BASE_URL=http://127.0.0.1:8084/v1
PROTOAGI_TTS_MODEL=tts-1-hd
PROTOAGI_TTS_VOICE=nova
PROTOAGI_TTS_RESPONSE_FORMAT=opus
PROTOAGI_TTS_SPEED=1.0
PROTOAGI_TTS_DELIVERY=auto
PROTOAGI_TTS_AUTO_COOLDOWN_SECONDS=21600
PROTOAGI_TTS_AUTO_MAX_CHARS=280
```

### Recommended Ukrainian setup: Piper UA bridge

```powershell
./scripts/start-tts-server.ps1                # start (default port 8084)
./scripts/start-tts-server.ps1 -Logs          # tail server log
./scripts/start-tts-server.ps1 -Stop          # stop the server
./scripts/start-tts-server.ps1 -Reinstall     # wipe venv and reinstall deps
```

The script bootstraps a local Python venv under `runs\tts-venv`, installs
[piper-tts](https://github.com/rhasspy/piper) + FastAPI/uvicorn, downloads
the `uk_UA-ukrainian_tts-medium` model (~63 MB) into `config\tts\models\`,
and starts the OpenAI-compatible TTS bridge [scripts/tts-server-uk.py](../scripts/tts-server-uk.py).

The bridge transcodes Piper's WAV output through `ffmpeg` into OGG/Opus so
Telegram receives a proper voice waveform via `sendVoice`. On Windows the
startup scripts first look for `ffmpeg` in `PATH`, then bootstrap a local copy
under `runs\ffmpeg` using `scripts\ensure-ffmpeg.ps1`. Override the download
source with `PROTOAGI_FFMPEG_URL` if needed. If ffmpeg is still unavailable,
`run-nikola.bat` falls back to `PROTOAGI_TTS_RESPONSE_FORMAT=wav`, which skips
transcoding and delivers the result via `sendAudio` instead.

[config/tts/voice_map.json](../config/tts/voice_map.json) maps persona
voice keys to Piper UA speakers (the model ships with three speakers
from the `robinhad/ukrainian-tts` dataset):

- `solomiya` → `lada` — warm female timbre, slightly slowed (length_scale 1.05)
- `mykola` → `mykyta` — the model's only male voice
- standard OpenAI names (`alloy`/`echo`/`fable`/`onyx`/`nova`/`shimmer`)
  remap to the three UA speakers
- raw speaker keys (`mykyta`/`lada`/`tetiana`) also work

Resource footprint: Piper runs on CPU through onnxruntime, ~150 MB RAM,
no VRAM at all — gpt-oss-20b keeps its full GPU budget. First synthesis
of a turn takes ~0.5 s; subsequent generations are faster.

### Why not XTTS-v2 / Coqui

The previous Docker setup wired
[openedai-speech](https://github.com/matatonic/openedai-speech) with
XTTS-v2 (`language=uk`). XTTS-v2 lists Ukrainian but tokenizes it
through a Russian-leaning phoneme set; even with a Ukrainian reference
WAV the output had heavy Russian accent and mis-stressed words. This is
a property of XTTS-v2's text frontend, not something cloning can fix.
The old `config/tts/voice_to_speaker.json` is kept only as reference; the
PowerShell start script no longer launches Docker.

### Cloning your own Ukrainian voice (later)

Piper does not do zero-shot cloning — to use a personal voice you have
to fine-tune the model on a small dataset. The community workflow:

1. Record 10–30 minutes of clean Ukrainian speech (mono WAV, 22 050 Hz, ~10 s clips with transcripts).
2. Use [piper-recording-studio](https://github.com/rhasspy/piper-recording-studio) to capture sentences.
3. Fine-tune with [piper-train](https://github.com/rhasspy/piper#training) starting from the existing `uk_UA-ukrainian_tts-medium` checkpoint.
4. Drop the resulting `.onnx` + `.onnx.json` into `config/tts/models/` and pass `--model <name>` to `start-tts-server.ps1`.

If you'd rather avoid training, the cloud option is Microsoft Edge
Neural TTS (`uk-UA-OstapNeural`, `uk-UA-PolinaNeural`) or OpenAI's
`gpt-4o-mini-tts` — both speak natural Ukrainian and plug into the same
`/v1/audio/speech` API (point `PROTOAGI_TTS_BASE_URL` at a small
[openai-edge-tts](https://github.com/travisvn/openai-edge-tts) wrapper
or OpenAI's API directly).

### Telegram audio plumbing

Telegram `sendVoice` requires OGG/Opus. If your TTS server cannot emit
opus, set `PROTOAGI_TTS_RESPONSE_FORMAT=mp3` (or `aac`/`flac`/`wav`) and
the bot will route through `sendAudio` instead — the user sees an audio
bubble instead of a voice waveform, but the message goes through.

When the TTS request fails (server down, format mismatch, JSON error
blob instead of audio) the bot prints a one-line `[tts]` reason to
stdout so the operator notices. In `auto` and `voice` delivery modes, the
text reply is sent as a fallback when audio was supposed to replace text.
In `text_and_voice` mode, the text has already been sent and the
voice/audio attachment remains best-effort.

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

The Telegram decision loop can also request bounded tools directly. The first
wired tools are `recall` and `remind_me`: if the model returns a
`tool_request` inside the constrained decision JSON, the bot executes up to
four tool steps and asks the profile to revise the final decision with the tool
results. Telegram production decisions pin the schema-native
`tool_request` path for local llama.cpp compatibility.
This lets questions like "what do you remember about me?" quote real memory
instead of guessing from the current prompt.

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

An opt-in asyncio supervisor is available for concurrent update handling:

```powershell
$env:PYTHONPATH="src"
python -m protoagi telegram --async --max-concurrent-updates 2
```

It keeps the synchronous bot internals and wraps blocking Telegram/LLM work in
`asyncio.to_thread`, with a semaphore bounding concurrent LLM calls.
