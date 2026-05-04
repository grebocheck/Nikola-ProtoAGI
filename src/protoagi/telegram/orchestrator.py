"""Telegram conversation bot orchestration.

This module composes the small focused modules in :mod:`protoagi.telegram`
into the public ``NikolaBot`` class. The bot is intentionally synchronous to
match the local llama.cpp deployment, which generates one reply at a time.
"""

from __future__ import annotations

import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import AgentConfig, PROJECT_ROOT
from ..embedding import EmbeddingClient, EmbeddingConfig
from ..harmony import clean_model_content
from ..memory import (
    KIND_EPISODIC,
    KIND_FACT,
    KIND_PERSONA_SELF,
    MemoryStore,
    SCOPE_CHAT,
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    SCOPE_USER,
    TelegramChat,
    utc_now,
)
from ..memory_service import MemoryService, RecallQuery
from ..openai_compat import OpenAICompatError, OpenAICompatibleClient
from ..persona import PersonaProfile, get_persona

from .api import TelegramApi, TelegramApiError, is_telegram_polling_conflict
from .attachments import TelegramAttachmentMixin
from .config import TelegramConfig
from .constants import (
    OFFSET_KEY,
    TELEGRAM_CHAT_THREAD_PREFIX,
    TELEGRAM_GLOBAL_MEMORY_TAG,
    TELEGRAM_PERSONA_SELF_MEMORY_TAG,
)
from .identity import (
    honest_identity_reply,
    is_deceptive_identity_reply,
    is_identity_question,
    is_image_blind_reply,
)
from .json_io import (
    DECISION_JSON_SCHEMA,
    Decision,
    INITIATIVE_JSON_SCHEMA,
    ImageAttachment,
    InitiativeDecision,
    StickerAttachment,
    decision_from_payload,
    decision_reply_texts,
    extract_json_object,
    image_to_payload,
    initiative_from_payload,
    sticker_to_payload,
)
from .prompts import decision_system_prompt, initiative_system_prompt, reply_system_prompt
from .stickers import (
    STICKER_PACKS,
    normalize_sticker_pack,
)
from .sticker_ops import TelegramStickerMixin
from .text import (
    GENERIC_STICKER_FILLER_RE,
    display_sender,
    parse_command,
    split_telegram_message,
    strip_assistanty_phrases,
    strip_speaker_prefixes,
)
from .tool_runner import TelegramToolEvent, TelegramToolRunner
from .vision import VisionDescriber
from .style import ReplyStyleTuner
from .voice import (
    VoiceAttachment,
    VoiceSynthesisConfig,
    VoiceSynthesizer,
    VoiceTranscriptionConfig,
    VoiceTranscriptionResult,
    VoiceTranscriber,
)


_TELEGRAM_LOG_TAIL_BYTES = 1_000_000
_REMINDER_CHECK_KV = "telegram:last_reminder_check"
_REFLECTION_LAST_KV = "telegram:last_reflection_at"
_DECISION_METRICS_KV = "telegram:decision_metrics"
_REMINDER_CHECK_SECONDS = 60
_REFLECTION_INTERVAL_SECONDS = 6 * 60 * 60


def sqlite3_error_types() -> tuple[type[BaseException], ...]:
    """Return the runtime sqlite3 exception types as a tuple."""

    import sqlite3

    return (sqlite3.Error,)


class NikolaBot(TelegramAttachmentMixin, TelegramStickerMixin):
    def __init__(
        self,
        *,
        telegram: TelegramApi,
        llm: OpenAICompatibleClient,
        memory: MemoryStore,
        telegram_config: TelegramConfig,
        agent_config: AgentConfig,
        memory_service: MemoryService | None = None,
    ) -> None:
        self.telegram = telegram
        self.llm = llm
        self.memory = memory
        self.telegram_config = telegram_config
        self.agent_config = agent_config
        self.persona: PersonaProfile = get_persona(telegram_config.persona_key)
        self.bot_username = self.memory.get_kv("telegram:bot_username") or ""
        self._last_proactive_check = 0.0
        self._last_reminder_check = 0.0
        self._last_reflection_check = 0.0
        self._sticker_cache: dict[str, list[dict[str, str]]] = {}
        self.error_log_path = PROJECT_ROOT / "runs" / "telegram-errors.log"
        self._style_tuner = ReplyStyleTuner(memory)
        initial_vision_llm = (
            OpenAICompatibleClient(
                telegram_config.vision_base_url or agent_config.base_url,
                telegram_config.vision_model,
                timeout_seconds=telegram_config.vision_timeout_seconds,
            )
            if telegram_config.vision_model
            else None
        )
        self._vision = VisionDescriber(
            telegram,
            initial_vision_llm,
            max_bytes=telegram_config.vision_max_bytes,
            memory=memory,
        )
        voice_base_url = telegram_config.voice_base_url or agent_config.base_url
        self._voice = VoiceTranscriber(
            telegram,
            VoiceTranscriptionConfig(
                base_url=voice_base_url if telegram_config.voice_model else "",
                model=telegram_config.voice_model,
                timeout_seconds=telegram_config.voice_timeout_seconds,
                max_bytes=telegram_config.voice_max_bytes,
            ),
        )
        tts_base_url = telegram_config.tts_base_url or agent_config.base_url
        self._tts = VoiceSynthesizer(
            VoiceSynthesisConfig(
                base_url=tts_base_url if telegram_config.tts_enabled else "",
                model=telegram_config.tts_model,
                voice=telegram_config.tts_voice,
                timeout_seconds=telegram_config.voice_timeout_seconds,
                max_chars=telegram_config.tts_max_chars,
                enabled=telegram_config.tts_enabled,
            )
        )
        if memory_service is None:
            embedding_config = EmbeddingConfig(
                base_url=agent_config.embedding.base_url,
                model=agent_config.embedding.model,
                timeout_seconds=agent_config.embedding.timeout_seconds,
                request_dimensions=agent_config.embedding.request_dimensions,
            )
            embedding_client = EmbeddingClient(embedding_config) if embedding_config.enabled else None
            memory_service = MemoryService(
                memory,
                embedding_client=embedding_client,
                embedding_backend=agent_config.embedding.backend,
                importance_client=llm if agent_config.llm_importance else None,
                llm_importance=agent_config.llm_importance,
            )
        self.memory_service = memory_service

    # ------------------------------------------------------------------
    # Vision LLM proxy. Tests and callers occasionally swap the vision
    # client at runtime; mirror the assignment into the VisionDescriber so
    # the change actually takes effect.

    @property
    def vision_llm(self) -> OpenAICompatibleClient | None:
        return self._vision.vision_llm

    @vision_llm.setter
    def vision_llm(self, value: OpenAICompatibleClient | None) -> None:
        self._vision.vision_llm = value
        # Cached marker becomes stale when the underlying client changes.
        self._vision._media_marker = None  # type: ignore[attr-defined]

    # ------------------------------------------------------------------
    # Lifecycle

    def bootstrap(self, *, delete_webhook: bool = False, drop_pending_updates: bool = False) -> dict[str, Any]:
        if delete_webhook:
            self.telegram.delete_webhook(drop_pending_updates=drop_pending_updates)
        me = self.telegram.get_me()
        self.bot_username = str(me.get("username", ""))
        self.memory.set_kv("telegram:bot_username", self.bot_username)
        self.memory.set_kv("telegram:bot_id", str(me.get("id", "")))
        return me

    def run_forever(self) -> None:
        while True:
            try:
                self.poll_once()
                self.maybe_run_initiative()
                self.maybe_dispatch_reminders()
                self.maybe_run_reflection()
            except TelegramApiError as exc:
                if is_telegram_polling_conflict(exc):
                    raise
                print(f"Telegram loop transient error: {exc}", flush=True)
                time.sleep(5)
            except (OpenAICompatError, OSError) as exc:
                print(f"Telegram loop transient error: {exc}", flush=True)
                time.sleep(5)
            except Exception as exc:
                self._log_loop_exception(exc)
                print(f"Telegram loop unexpected error: {exc}; see {self.error_log_path}", flush=True)
                time.sleep(5)

    def poll_once(self) -> int:
        offset_text = self.memory.get_kv(OFFSET_KEY)
        offset = int(offset_text) if offset_text else None
        updates = self.telegram.get_updates(
            offset=offset,
            timeout_seconds=self.telegram_config.poll_timeout_seconds,
            allowed_updates=["message", "edited_message", "message_reaction"],
        )
        processed = 0
        for update in updates:
            update_id = int(update.get("update_id", 0))
            try:
                if self.process_update(update):
                    processed += 1
            finally:
                self.memory.set_kv(OFFSET_KEY, str(update_id + 1))
        return processed

    def maybe_run_initiative(self) -> int:
        if not self.telegram_config.proactive_enabled:
            return 0
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_proactive_check < self.telegram_config.proactive_check_seconds:
            return 0
        self._last_proactive_check = now_monotonic
        return self.run_initiative_once()

    def maybe_dispatch_reminders(self) -> int:
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_reminder_check < _REMINDER_CHECK_SECONDS:
            return 0
        self._last_reminder_check = now_monotonic
        return self.dispatch_due_reminders()

    def dispatch_due_reminders(self, *, limit: int = 20) -> int:
        delivered = 0
        now = utc_now()
        for reminder in self.memory.due_reminders(now, limit=limit):
            chat_id = reminder.chat_id
            if not chat_id or not self._chat_allowed(str(chat_id)):
                # No chat to deliver into; mark as cancelled so we stop
                # retrying. The text is still recorded in the reminders
                # table for auditing.
                self.memory.mark_reminder(reminder.id, "cancelled")
                continue
            chat = self.memory.get_telegram_chat(chat_id)
            if chat is None:
                self.memory.mark_reminder(reminder.id, "cancelled")
                continue
            if reminder.persona_key and reminder.persona_key != self.persona.key:
                # Reminder targets a different persona; another worker can
                # pick it up. Skip without marking sent.
                continue
            try:
                text = f"⏰ {reminder.text}"
                self._send_reply(
                    chat,
                    text,
                    initiative=True,
                    disable_notification=False,
                )
                self.memory.mark_reminder(reminder.id, "sent")
                delivered += 1
            except TelegramApiError:
                # Leave the reminder pending for retry next tick.
                continue
        return delivered

    def maybe_run_reflection(self) -> bool:
        now_monotonic = time.monotonic()
        if now_monotonic - self._last_reflection_check < _REFLECTION_INTERVAL_SECONDS:
            return False
        last_at = self.memory.get_kv(_REFLECTION_LAST_KV)
        if last_at:
            try:
                last_dt = datetime.fromisoformat(last_at)
                if datetime.now(timezone.utc) - last_dt < timedelta(seconds=_REFLECTION_INTERVAL_SECONDS):
                    self._last_reflection_check = now_monotonic
                    return False
            except ValueError:
                pass
        self._last_reflection_check = now_monotonic
        self.run_reflection_pass()
        return True

    def run_reflection_pass(self) -> dict[str, int]:
        """Periodic memory hygiene + meta-memory write.

        The pass:

        1. Consolidates near-identical memories in the global + active
           persona scopes (older items get ``superseded_by`` set).
        2. Prunes low-value items past the 60-day grace window using a
           conservative score threshold. Pinned and ``persona_self`` items
           are protected by ``MemoryService.prune`` defaults.
        3. When fictional self-memory is enabled, asks the model for one
           or two short first-person reflection notes that are stored as
           ``persona_self`` memories.
        """

        result = {
            "consolidated_global": 0,
            "consolidated_persona": 0,
            "pruned_global": 0,
            "pruned_persona": 0,
            "pruned_media": 0,
            "pruned_importance_cache": 0,
            "reflections_written": 0,
        }
        try:
            result["consolidated_global"] = self.memory_service.consolidate(
                scope=SCOPE_GLOBAL, max_items=300
            )
        except sqlite3_error_types() as exc:  # pragma: no cover - defensive
            print(f"reflection consolidate(global) failed: {exc}", flush=True)
        try:
            result["consolidated_persona"] = self.memory_service.consolidate(
                scope=SCOPE_PERSONA, persona_key=self.persona.key, max_items=200
            )
        except sqlite3_error_types() as exc:  # pragma: no cover - defensive
            print(f"reflection consolidate(persona) failed: {exc}", flush=True)
        try:
            result["pruned_global"] = self.memory_service.prune(
                scope=SCOPE_GLOBAL,
                score_threshold=0.10,
                keep_newer_than_days=60.0,
                max_items=500,
            )["deleted"]
        except sqlite3_error_types() as exc:  # pragma: no cover - defensive
            print(f"reflection prune(global) failed: {exc}", flush=True)
        try:
            result["pruned_persona"] = self.memory_service.prune(
                scope=SCOPE_PERSONA,
                persona_key=self.persona.key,
                score_threshold=0.10,
                keep_newer_than_days=60.0,
                max_items=300,
            )["deleted"]
        except sqlite3_error_types() as exc:  # pragma: no cover - defensive
            print(f"reflection prune(persona) failed: {exc}", flush=True)
        try:
            result["pruned_media"] = self.memory.prune_orphan_media(older_than_days=60.0)
        except sqlite3_error_types() as exc:  # pragma: no cover - defensive
            print(f"reflection prune(media) failed: {exc}", flush=True)
        try:
            cache_prune = self.memory.prune_importance_cache(older_than_days=60.0)
            result["pruned_importance_cache"] = int(cache_prune["deleted"])
        except sqlite3_error_types() as exc:  # pragma: no cover - defensive
            print(f"reflection prune(importance_cache) failed: {exc}", flush=True)
        if self.telegram_config.fictional_self_enabled:
            try:
                result["reflections_written"] = self._write_reflection_memory()
            except (OpenAICompatError, OSError) as exc:
                print(f"reflection memory write failed: {exc}", flush=True)
        self.memory.set_kv(_REFLECTION_LAST_KV, utc_now())
        return result

    def _write_reflection_memory(self) -> int:
        """Ask the model to write up to two short self-reflection notes.

        We collect recent memories the bot already knows about, give them as
        context, and ask for terse first-person reflections. Each reflection
        is stored as a ``persona_self`` memory.
        """

        recent_facts = self.memory_service.recent(
            persona_key=self.persona.key,
            limit=8,
        )
        recent_self = self.memory_service.recent(
            persona_key=self.persona.key,
            kind=KIND_PERSONA_SELF,
            limit=8,
        )
        # Skip when there is essentially nothing new to reflect on.
        if not recent_facts and not recent_self:
            return 0
        prompt = (
            f"Ти — {self.persona.display_name}. Це внутрішня нічна нотатка для себе, не для користувача. "
            "На основі останніх памʼяток коротко запиши 1-2 факти або висновки про себе у форматі першої особи "
            "(наприклад: \"Я помітила, що мені краще даються розмови вранці\"). "
            "Не повторюй уже відомі факти; не вигадуй контактні дані чи офлайн-події. "
            "Поверни JSON: {\"reflections\": [string, ...]}."
        )
        context_payload = {
            "recent_user_facts": [
                {"text": item.text, "tags": item.tags}
                for item in recent_facts
            ],
            "recent_self_memories": [
                {"text": item.text, "tags": item.tags}
                for item in recent_self
            ],
        }
        schema = {
            "type": "json_schema",
            "json_schema": {
                "name": "reflection",
                "strict": False,
                "schema": {
                    "type": "object",
                    "properties": {
                        "reflections": {
                            "type": "array",
                            "items": {"type": "string"},
                        }
                    },
                    "required": ["reflections"],
                    "additionalProperties": False,
                },
            },
        }
        response = self.llm.chat_completion(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json.dumps(context_payload, ensure_ascii=False)},
            ],
            temperature=0.4,
            top_p=0.95,
            max_tokens=320,
            response_format=schema,
        )
        content = clean_model_content(
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        payload = extract_json_object(content)
        reflections = payload.get("reflections") or []
        written = 0
        for raw in reflections:
            text = str(raw or "").strip()
            if not text:
                continue
            self._remember_persona_self_fact(text)
            written += 1
        return written

    # ------------------------------------------------------------------
    # Update handling

    def process_update(self, update: dict[str, Any]) -> bool:
        if isinstance(update.get("message_reaction"), dict):
            return self._handle_message_reaction(update["message_reaction"])
        if isinstance(update.get("edited_message"), dict):
            return self._handle_edited_message(update["edited_message"])
        message = update.get("message")
        if not isinstance(message, dict):
            return False
        text = str(message.get("text") or message.get("caption") or "").strip()
        image = self._extract_image_attachment(message)
        voice = self._extract_voice_attachment(message)
        incoming_sticker = self._extract_sticker_attachment(message)
        if not text and image is None and incoming_sticker is None and voice is None:
            return False
        chat = message.get("chat") or {}
        if "id" not in chat:
            return False
        chat_id = str(chat["id"])
        if not self._chat_allowed(chat_id):
            return False

        user = message.get("from") or {}
        user_id = self._user_id_for(user)
        chat_state = self.memory.upsert_telegram_chat(
            chat,
            user,
            reply_mode=self.telegram_config.reply_mode,
        )
        self.memory.mark_telegram_user_message(chat_id)
        self._upsert_user_profile(user)
        self._style_tuner.record_incoming_reply(chat_id)

        current_message_id = int(message.get("message_id", 0))
        thread_id = self.thread_id(chat_id)
        display = display_sender(user)
        image_description = self._vision.describe(image, caption=text) if image is not None else ""
        voice_result = self._transcribe_voice(voice) if voice is not None else VoiceTranscriptionResult()
        voice_transcript = voice_result.text
        incoming_text = self._incoming_text_with_media(
            text,
            image,
            image_description,
            incoming_sticker,
            voice,
            voice_transcript,
        )
        content = self._history_user_content(chat_state, display, incoming_text)
        self.memory.log_message(thread_id, "user", content)
        if image is not None:
            self._remember_media_fact(
                chat_state,
                image,
                image_description,
                user_id=user_id,
            )
        if voice is not None:
            self._remember_voice_fact(
                chat_state,
                voice,
                voice_transcript,
                user_id=user_id,
                media_bytes=voice_result.data if self.telegram_config.store_voice else b"",
            )
        if current_message_id:
            self.memory.log_telegram_message(
                chat_id=chat_id,
                message_id=current_message_id,
                persona_key=self.persona.key,
                role="user",
                sender_id=user.get("id"),
                sender_name=display,
                text=incoming_text,
                metadata={
                    "from": user,
                    "chat": chat,
                    "image": image_to_payload(image),
                    "sticker": sticker_to_payload(incoming_sticker),
                    "voice": self._voice_to_payload(voice),
                    "voice_transcript": voice_transcript,
                },
            )

        command_reply = self._handle_command(text, chat_state)
        if command_reply is not None:
            self._send_reply(chat_state, command_reply)
            self._schedule_from_minutes(
                chat_state.chat_id,
                self.telegram_config.proactive_cooldown_seconds // 60,
            )
            return True

        addressed = self._is_addressed(text, message)
        if self._should_skip_without_llm(chat_state, addressed):
            return True

        decision = self.decide_incoming(
            chat_state,
            incoming_text,
            display,
            addressed,
            user_id=user_id,
        )
        for fact in decision.memories:
            self._remember_chat_fact(chat_state, fact, user_id=user_id)
        for fact in decision.self_memories:
            self._remember_persona_self_fact(fact)
        self._persist_reminder_requests(
            decision.reminders,
            chat_state=chat_state,
            user_id=user_id,
        )
        self._maybe_add_reaction_sticker(chat_state, incoming_text, decision)
        if not decision.should_reply and not decision.stickers:
            self._schedule_from_minutes(chat_state.chat_id, decision.next_check_minutes)
            return True
        reply_to_message_id = self._resolve_reply_target(
            decision.reply_to,
            current_message_id=current_message_id,
            chat=chat_state,
        )
        replies = [self._limit_reply(self._clean_reply_text(item)) for item in decision_reply_texts(decision)]
        self._send_reply(
            chat_state,
            replies,
            message_id=reply_to_message_id,
            stickers=decision.stickers,
        )
        self._schedule_from_minutes(chat_state.chat_id, decision.next_check_minutes)
        return True

    def _handle_message_reaction(self, reaction: dict[str, Any]) -> bool:
        chat = reaction.get("chat") or {}
        if "id" not in chat:
            return False
        chat_id = str(chat["id"])
        if not self._chat_allowed(chat_id):
            return False
        emoji = ""
        new_reaction = reaction.get("new_reaction") or []
        if isinstance(new_reaction, list) and new_reaction:
            first = new_reaction[0]
            if isinstance(first, dict):
                emoji = str(first.get("emoji") or "")
        self._style_tuner.record_reaction(chat_id, emoji)
        return True

    def _handle_edited_message(self, message: dict[str, Any]) -> bool:
        chat = message.get("chat") or {}
        if "id" not in chat:
            return False
        chat_id = str(chat["id"])
        if not self._chat_allowed(chat_id):
            return False
        self._style_tuner.record_edit(chat_id)
        if message.get("message_id"):
            self.memory.log_telegram_message(
                chat_id=chat_id,
                message_id=int(message["message_id"]),
                persona_key=self.persona.key,
                role="user",
                sender_id=(message.get("from") or {}).get("id"),
                sender_name=display_sender(message.get("from") or {}),
                text=str(message.get("text") or message.get("caption") or ""),
                metadata={"edited": True, "message": message},
            )
        return True

    # ------------------------------------------------------------------
    # Decision pipeline

    def decide_incoming(
        self,
        chat: TelegramChat,
        incoming_text: str,
        sender: str,
        addressed: bool,
        user_id: str | None = None,
    ) -> Decision:
        started = time.monotonic()
        llm_calls = 0
        used_tools = False
        recent = self._recent_compact_messages(chat)
        recent_telegram = self._recent_telegram_messages(chat.chat_id)
        facts = self._search_chat_memory(chat, incoming_text, user_id=user_id)
        persona_self_memory = self._persona_self_context(incoming_text)
        context_payload = {
            "persona": self.persona.payload(),
            "fictional_self_enabled": self.telegram_config.fictional_self_enabled,
            "persona_self_lore": list(self.persona.self_lore)
            if self.telegram_config.fictional_self_enabled
            else [],
            "chat": {
                "id": chat.chat_id,
                "type": chat.chat_type,
                "display_name": chat.display_name,
                "reply_mode": chat.reply_mode,
                "addressed_to_bot": addressed,
            },
            "sender": sender,
            "incoming_text": incoming_text,
            "recent_messages": recent,
            "recent_telegram_messages": recent_telegram,
            "relevant_memory": [
                {"text": fact.text, "tags": fact.tags, "created_at": fact.created_at}
                for fact in facts
            ],
            "known_persona_self_memory": [
                {"text": fact.text, "created_at": fact.created_at}
                for fact in persona_self_memory
            ],
            "adaptive_reply_style": self._style_payload(chat),
            "available_sticker_packs": STICKER_PACKS,
        }
        messages = [
            {"role": "system", "content": self._decision_system_prompt()},
            {
                "role": "user",
                "content": json.dumps(context_payload, ensure_ascii=False),
            },
        ]
        response = self.llm.chat_completion(
            messages,
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
            response_format=DECISION_JSON_SCHEMA,
        )
        llm_calls += 1
        message = response.get("choices", [{}])[0].get("message", {})
        content = clean_model_content(message.get("content", ""))
        payload = extract_json_object(content)
        decision = decision_from_payload(payload)
        if decision.tool_request:
            used_tools = True
            decision, merge_llm_calls = self._merge_decision_tool_results(
                chat,
                incoming_text,
                sender,
                addressed,
                user_id=user_id,
                context_payload=context_payload,
                initial_decision=decision,
            )
            llm_calls += merge_llm_calls
        if self.telegram_config.reply_mode == "always":
            decision.should_reply = True
        if chat.chat_type != "private" and self.telegram_config.reply_mode == "mention" and not addressed:
            decision.should_reply = False
        if self.telegram_config.reply_mode == "silent":
            decision.should_reply = False
        if is_identity_question(incoming_text):
            decision.should_reply = True
            replies = decision_reply_texts(decision)
            if not replies or any(is_deceptive_identity_reply(item) for item in replies):
                decision.reply = honest_identity_reply(self.persona)
                decision.replies = []
                decision.stickers = []
        if decision.should_reply and not decision.reply and not decision.replies:
            decision.reply = self.compose_reply(chat, incoming_text, sender)
            llm_calls += 1
        if "[зображення:" in incoming_text:
            decision.reply = "" if is_image_blind_reply(decision.reply) else decision.reply
            decision.replies = [item for item in decision.replies if not is_image_blind_reply(item)]
            if not decision_reply_texts(decision) and not decision.stickers:
                decision.should_reply = False
        if decision.should_reply and not decision_reply_texts(decision) and not decision.stickers:
            decision.should_reply = False
        self._record_decision_metrics(
            llm_calls=llm_calls,
            used_tools=used_tools,
            elapsed_seconds=time.monotonic() - started,
        )
        return decision

    def _merge_decision_tool_results(
        self,
        chat: TelegramChat,
        incoming_text: str,
        sender: str,
        addressed: bool,
        *,
        user_id: str | None,
        context_payload: dict[str, Any],
        initial_decision: Decision,
    ) -> tuple[Decision, int]:
        runner = TelegramToolRunner(
            memory_service=self.memory_service,
            chat=chat,
            persona_key=self.persona.key,
            user_id=user_id,
            global_memory=self.telegram_config.global_memory,
            max_steps=4,
        )
        events = runner.run(
            tool_request=initial_decision.tool_request,
        )
        if not events:
            return initial_decision, 0
        inline = self._inline_tool_decision(initial_decision, events)
        if inline is not None:
            return inline, 0
        merge_payload = {
            "original_context": context_payload,
            "initial_decision": _decision_to_payload(initial_decision),
            "tool_results": [_tool_event_payload(event) for event in events],
            "instructions": (
                "Revise the Telegram decision using tool_results. Return final decision JSON only. "
                "Do not request another tool."
            ),
        }
        response = self.llm.chat_completion(
            [
                {"role": "system", "content": self._decision_system_prompt()},
                {"role": "user", "content": json.dumps(merge_payload, ensure_ascii=False)},
            ],
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
            response_format=DECISION_JSON_SCHEMA,
        )
        message = response.get("choices", [{}])[0].get("message", {})
        content = clean_model_content(message.get("content", ""))
        revised = decision_from_payload(extract_json_object(content))
        if not revised.should_reply and not decision_reply_texts(revised) and not revised.stickers:
            revised = initial_decision
        if revised.should_reply and not decision_reply_texts(revised):
            fallback = self._tool_result_reply(events)
            if fallback:
                revised.reply = fallback
        revised.tool_request = None
        return revised, 1

    def _inline_tool_decision(
        self,
        initial_decision: Decision,
        events: list[TelegramToolEvent],
    ) -> Decision | None:
        if decision_reply_texts(initial_decision) or initial_decision.stickers:
            return None
        if len(events) != 1:
            return None
        fallback = self._tool_result_reply(events)
        if not fallback:
            return None
        return Decision(
            should_reply=True,
            reply=fallback,
            memories=list(initial_decision.memories),
            self_memories=list(initial_decision.self_memories),
            replies=[],
            reply_to=initial_decision.reply_to,
            stickers=list(initial_decision.stickers),
            reminders=list(initial_decision.reminders),
            tool_request=None,
            next_check_minutes=initial_decision.next_check_minutes,
        )

    def _record_decision_metrics(
        self,
        *,
        llm_calls: int,
        used_tools: bool,
        elapsed_seconds: float,
    ) -> None:
        try:
            raw = self.memory.get_kv(_DECISION_METRICS_KV)
            metrics = json.loads(raw) if raw else {}
            if not isinstance(metrics, dict):
                metrics = {}
            decision_count = int(metrics.get("decisions", 0)) + 1
            total_llm_calls = int(metrics.get("llm_calls", 0)) + max(0, int(llm_calls))
            elapsed_ms = max(0.0, elapsed_seconds * 1000.0)
            histogram = metrics.get("llm_call_histogram")
            if not isinstance(histogram, dict):
                histogram = {}
            bucket = str(max(0, int(llm_calls)))
            histogram[bucket] = int(histogram.get(bucket, 0)) + 1
            metrics.update(
                {
                    "decisions": decision_count,
                    "llm_calls": total_llm_calls,
                    "latency_ms": float(metrics.get("latency_ms", 0.0)) + elapsed_ms,
                    "max_llm_calls": max(
                        int(metrics.get("max_llm_calls", 0)),
                        max(0, int(llm_calls)),
                    ),
                    "max_latency_ms": max(
                        float(metrics.get("max_latency_ms", 0.0)),
                        elapsed_ms,
                    ),
                    "llm_call_histogram": histogram,
                    "updated_at": utc_now(),
                }
            )
            if used_tools:
                metrics["tool_decisions"] = int(metrics.get("tool_decisions", 0)) + 1
                metrics["tool_llm_calls"] = int(metrics.get("tool_llm_calls", 0)) + max(
                    0,
                    int(llm_calls),
                )
            self.memory.set_kv(_DECISION_METRICS_KV, json.dumps(metrics, ensure_ascii=False))
        except sqlite3_error_types() + (ValueError, TypeError, json.JSONDecodeError):  # pragma: no cover
            return None

    @staticmethod
    def _tool_result_reply(events: list[TelegramToolEvent]) -> str:
        for event in events:
            if event.name != "recall" or not event.result.get("ok"):
                continue
            items = event.result.get("items") or []
            if not items:
                return ""
            first = items[0]
            text = str(first.get("text") or "").strip()
            if text:
                return f"Пам'ятаю: {text}"
        return ""

    def compose_reply(self, chat: TelegramChat, incoming_text: str, sender: str) -> str:
        recent = self._recent_compact_messages(chat)
        facts = self._search_chat_memory(chat, incoming_text)
        persona_self_memory = self._persona_self_context(incoming_text)
        response = self.llm.chat_completion(
            [
                {"role": "system", "content": self._reply_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "persona": self.persona.payload(),
                            "fictional_self_enabled": self.telegram_config.fictional_self_enabled,
                            "persona_self_lore": list(self.persona.self_lore)
                            if self.telegram_config.fictional_self_enabled
                            else [],
                            "chat": {"type": chat.chat_type, "display_name": chat.display_name},
                            "sender": sender,
                            "incoming_text": incoming_text,
                            "recent_messages": recent,
                            "relevant_memory": [fact.text for fact in facts],
                            "known_persona_self_memory": [fact.text for fact in persona_self_memory],
                            "adaptive_reply_style": self._style_payload(chat),
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
        )
        content = clean_model_content(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return self._clean_reply_text(content)

    def run_initiative_once(self) -> int:
        sent = 0
        now = utc_now()
        for chat in self.memory.list_due_telegram_chats(now, limit=10):
            if not self._chat_allowed(chat.chat_id):
                continue
            if not self._initiative_cooldown_elapsed(chat):
                self._schedule_from_minutes(chat.chat_id, self.telegram_config.proactive_cooldown_seconds // 60)
                continue
            decision = self.decide_initiative(chat)
            for fact in decision.memories:
                self._remember_chat_fact(chat, fact)
            for fact in decision.self_memories:
                self._remember_persona_self_fact(fact)
            self._persist_reminder_requests(decision.reminders, chat_state=chat)
            self._schedule_from_minutes(chat.chat_id, decision.next_check_minutes)
            if not decision.send:
                continue
            if not decision.message.strip() and not decision.stickers:
                continue
            try:
                self._send_reply(
                    chat,
                    self._limit_reply(self._clean_reply_text(decision.message)),
                    initiative=True,
                    disable_notification=self.telegram_config.proactive_disable_notification,
                    stickers=decision.stickers,
                )
                sent += 1
            except TelegramApiError:
                self.memory.set_telegram_proactive(chat.chat_id, False)
        return sent

    def decide_initiative(self, chat: TelegramChat) -> InitiativeDecision:
        recent = self._recent_compact_messages(chat)
        recent_telegram = self._recent_telegram_messages(chat.chat_id)
        facts = self._search_chat_memory(chat, chat.display_name)
        persona_self_memory = self._persona_self_context(chat.display_name)
        response = self.llm.chat_completion(
            [
                {"role": "system", "content": self._initiative_system_prompt()},
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "persona": self.persona.payload(),
                            "fictional_self_enabled": self.telegram_config.fictional_self_enabled,
                            "persona_self_lore": list(self.persona.self_lore)
                            if self.telegram_config.fictional_self_enabled
                            else [],
                            "chat": {
                                "id": chat.chat_id,
                                "type": chat.chat_type,
                                "display_name": chat.display_name,
                                "last_user_message_at": chat.last_user_message_at,
                                "last_bot_message_at": chat.last_bot_message_at,
                                "last_initiative_at": chat.last_initiative_at,
                            },
                            "recent_messages": recent,
                            "recent_telegram_messages": recent_telegram,
                            "memory": [fact.text for fact in facts],
                            "known_persona_self_memory": [fact.text for fact in persona_self_memory],
                            "adaptive_reply_style": self._style_payload(chat),
                            "available_sticker_packs": STICKER_PACKS,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
            response_format=INITIATIVE_JSON_SCHEMA,
        )
        content = clean_model_content(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        return initiative_from_payload(extract_json_object(content))

    # ------------------------------------------------------------------
    # Sending

    def _handle_command(self, text: str, chat: TelegramChat) -> str | None:
        command, _rest = parse_command(text, self.bot_username)
        if command is None:
            return None
        if command == "start":
            self.memory.set_telegram_proactive(chat.chat_id, True)
            return self.persona.start_message
        return None

    def _send_reply(
        self,
        chat: TelegramChat,
        text: str | list[str],
        *,
        message_id: int | None = None,
        initiative: bool = False,
        disable_notification: bool = False,
        stickers: list[dict[str, str]] | None = None,
    ) -> None:
        stickers = stickers or []
        style_choice = self._style_tuner.choose(chat.chat_id)
        raw_messages = text if isinstance(text, list) else [text]
        messages: list[str] = []
        for item in raw_messages[: max(1, self.telegram_config.max_reply_messages)]:
            cleaned = self._clean_reply_text(item)
            if cleaned:
                messages.append(cleaned)
        if stickers and len(messages) == 1 and self._is_generic_sticker_filler(messages[0]):
            messages = []
        if messages:
            try:
                self.telegram.send_chat_action(chat.chat_id, "typing")
            except TelegramApiError:
                pass
        sent_text = False
        for message_index, message_text in enumerate(messages):
            chunks = split_telegram_message(message_text, max_chars=self.telegram_config.max_reply_chars)
            for chunk_index, chunk in enumerate(chunks):
                should_reply_to = message_id if message_index == 0 and chunk_index == 0 and not initiative else None
                sent = self.telegram.send_message(
                    chat.chat_id,
                    chunk,
                    reply_to_message_id=should_reply_to,
                    disable_notification=disable_notification,
                )
                sent_text = True
                self._log_sent_telegram_message(
                    chat,
                    sent,
                    "assistant",
                    chunk,
                    reply_to_message_id=should_reply_to,
                )
        if messages and self.telegram_config.tts_enabled:
            voice_data = self._tts.synthesize(messages[0])
            if voice_data:
                try:
                    sent_voice = self.telegram.send_voice_bytes(
                        chat.chat_id,
                        voice_data,
                        reply_to_message_id=message_id if not initiative and not sent_text else None,
                        disable_notification=disable_notification,
                    )
                    self._log_sent_telegram_message(
                        chat,
                        sent_voice,
                        "assistant",
                        "[voice-reply]",
                        reply_to_message_id=message_id if not initiative and not sent_text else None,
                    )
                except (TelegramApiError, OSError):
                    pass
        for sticker_choice in stickers[:2]:
            file_id = self._select_sticker_file_id(sticker_choice, chat.chat_id)
            if not file_id:
                continue
            try:
                self.telegram.send_chat_action(chat.chat_id, "choose_sticker")
            except TelegramApiError:
                pass
            sent = self.telegram.send_sticker(
                chat.chat_id,
                file_id,
                reply_to_message_id=message_id if not initiative and not sent_text else None,
                disable_notification=disable_notification,
            )
            pack = normalize_sticker_pack(sticker_choice.get("pack", ""))
            self._log_sent_telegram_message(
                chat,
                sent,
                "assistant",
                f"[sticker:{pack or 'unknown'}]",
                reply_to_message_id=message_id if not initiative and not sent_text else None,
            )
        if messages or stickers:
            history_text = "\n".join(messages) if messages else "[sticker]"
            self.memory.log_message(
                self.thread_id(chat.chat_id),
                "assistant",
                self._history_assistant_content(chat, history_text),
            )
            self.memory.mark_telegram_bot_message(chat.chat_id, initiative=initiative)
            self._style_tuner.record_sent(
                chat.chat_id,
                arm=style_choice.arm,
                reply_chars=sum(len(item) for item in messages),
                sticker_count=len(stickers),
                message_count=len(messages),
            )

    def _log_sent_telegram_message(
        self,
        chat: TelegramChat,
        sent: dict[str, Any],
        role: str,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> None:
        message_id = sent.get("message_id")
        if message_id is None:
            return
        metadata = dict(sent)
        if reply_to_message_id is not None:
            metadata["protoagi_reply_to_message_id"] = int(reply_to_message_id)
        self.memory.log_telegram_message(
            chat_id=chat.chat_id,
            message_id=int(message_id),
            persona_key=self.persona.key,
            role=role,
            sender_id=None,
            sender_name=self.telegram_config.bot_name,
            text=text,
            metadata=metadata,
        )

    def _resolve_reply_target(
        self,
        reply_to: str | int | None,
        *,
        current_message_id: int,
        chat: TelegramChat,
    ) -> int | None:
        if reply_to is None:
            return None
        if isinstance(reply_to, str):
            value = reply_to.strip().lower()
            if value in {"", "none", "null", "false", "no"}:
                return None
            if value == "current":
                if chat.chat_type == "private":
                    return None
                return current_message_id or None
            if value.isdigit():
                reply_to = int(value)
            else:
                return None
        try:
            target = int(reply_to)
        except (TypeError, ValueError):
            return None
        if target == current_message_id:
            return None if chat.chat_type == "private" else target
        recent_ids = {
            int(item["message_id"])
            for item in self.memory.recent_telegram_messages(
                chat.chat_id,
                limit=self.telegram_config.max_history_messages,
            )
        }
        return target if target in recent_ids else None

    # ------------------------------------------------------------------
    # Memory hooks

    def _remember_chat_fact(
        self,
        chat: TelegramChat,
        text: str,
        *,
        user_id: str | None = None,
    ) -> None:
        text = text.strip()
        if not text:
            return
        tags = [
            "telegram",
            TELEGRAM_GLOBAL_MEMORY_TAG,
            f"chat_type:{chat.chat_type}",
            f"source_chat:{chat.chat_id}",
            f"persona:{self.persona.key}",
        ]
        if user_id:
            tags.append(f"user:{user_id}")
        scope = SCOPE_GLOBAL if self.telegram_config.global_memory else (SCOPE_USER if user_id else SCOPE_CHAT)
        self.memory_service.remember(
            text,
            kind=KIND_FACT,
            scope=scope,
            tags=tags,
            user_id=user_id,
            chat_id=chat.chat_id,
            persona_key=self.persona.key,
            source="telegram_chat",
        )

    def _remember_media_fact(
        self,
        chat: TelegramChat,
        image: ImageAttachment,
        description: str,
        *,
        user_id: str | None = None,
    ) -> None:
        description = str(description or "").strip()
        if not description:
            return
        text = f"Telegram image in chat {chat.chat_id}: {description}"
        tags = [
            "telegram",
            "media",
            "image",
            TELEGRAM_GLOBAL_MEMORY_TAG,
            f"source_chat:{chat.chat_id}",
            f"persona:{self.persona.key}",
        ]
        if user_id:
            tags.append(f"user:{user_id}")
        scope = SCOPE_GLOBAL if self.telegram_config.global_memory else (SCOPE_USER if user_id else SCOPE_CHAT)
        self.memory_service.remember(
            text,
            kind=KIND_EPISODIC,
            scope=scope,
            tags=tags,
            user_id=user_id,
            chat_id=chat.chat_id,
            persona_key=self.persona.key,
            media_id=image.file_id,
            importance=0.55,
            source="telegram_media",
            metadata={
                "file_id": image.file_id,
                "mime_type": image.mime_type,
                "label": image.label,
                "file_name": image.file_name,
            },
        )

    def _remember_voice_fact(
        self,
        chat: TelegramChat,
        voice: VoiceAttachment,
        transcript: str,
        *,
        user_id: str | None = None,
        media_bytes: bytes | None = None,
    ) -> None:
        transcript = str(transcript or "").strip()
        media_id: str | None = None
        if media_bytes:
            try:
                media_id = self.memory.store_media_blob(
                    file_id=voice.file_id,
                    mime=voice.mime_type or "audio/ogg",
                    data=media_bytes,
                    caption=transcript,
                ).file_id
            except ((OSError, ValueError) + sqlite3_error_types()) as exc:
                print(f"voice media persistence failed: {exc}", flush=True)
        if not transcript:
            return
        text = f"Telegram voice in chat {chat.chat_id}: {transcript}"
        tags = [
            "telegram",
            "voice",
            "audio",
            TELEGRAM_GLOBAL_MEMORY_TAG,
            f"source_chat:{chat.chat_id}",
            f"persona:{self.persona.key}",
        ]
        if user_id:
            tags.append(f"user:{user_id}")
        scope = SCOPE_GLOBAL if self.telegram_config.global_memory else (SCOPE_USER if user_id else SCOPE_CHAT)
        self.memory_service.remember(
            text,
            kind=KIND_EPISODIC,
            scope=scope,
            tags=tags,
            user_id=user_id,
            chat_id=chat.chat_id,
            persona_key=self.persona.key,
            media_id=media_id,
            importance=0.5,
            source="telegram_voice",
            metadata={
                "file_id": voice.file_id,
                "mime_type": voice.mime_type,
                "duration": voice.duration,
                "label": voice.label,
                "transcript": transcript,
            },
        )

    def _transcribe_voice(self, voice: VoiceAttachment | None) -> VoiceTranscriptionResult:
        if voice is None:
            return VoiceTranscriptionResult()
        with_bytes = getattr(self._voice, "transcribe_with_bytes", None)
        if callable(with_bytes):
            result = with_bytes(voice)
            if isinstance(result, VoiceTranscriptionResult):
                return result
            if isinstance(result, tuple):
                text = str(result[0] if len(result) > 0 else "").strip()
                data = result[1] if len(result) > 1 and isinstance(result[1], bytes) else b""
                return VoiceTranscriptionResult(
                    text=text,
                    data=data,
                    mime_type=voice.mime_type or "audio/ogg",
                    file_id=voice.file_id,
                )
        text = self._voice.transcribe(voice)
        return VoiceTranscriptionResult(
            text=str(text or "").strip(),
            mime_type=voice.mime_type or "audio/ogg",
            file_id=voice.file_id,
        )

    def _persist_reminder_requests(
        self,
        requests: list[dict[str, Any]],
        *,
        chat_state: TelegramChat,
        user_id: str | None = None,
    ) -> int:
        """Turn ``decision.reminders`` entries into rows in the reminders table.

        The model is allowed to phrase reminders naturally; we resolve the
        target timestamp from either ``in_minutes`` or an ISO ``trigger_at``,
        cap at 365 days, and bind the row to the originating chat / persona
        so the dispatcher knows where to deliver it.
        """

        if not requests:
            return 0
        added = 0
        max_minutes = 60 * 24 * 365
        for entry in requests:
            text = str(entry.get("text") or "").strip()
            if not text:
                continue
            trigger_at = ""
            raw_trigger = entry.get("trigger_at")
            if isinstance(raw_trigger, str) and raw_trigger.strip():
                try:
                    parsed = datetime.fromisoformat(raw_trigger.strip())
                    if parsed.tzinfo is None:
                        parsed = parsed.replace(tzinfo=timezone.utc)
                    trigger_at = parsed.isoformat(timespec="seconds")
                except ValueError:
                    trigger_at = ""
            if not trigger_at:
                in_minutes = entry.get("in_minutes")
                try:
                    minutes = max(1, min(int(in_minutes), max_minutes))
                except (TypeError, ValueError):
                    minutes = 60
                trigger_at = (
                    datetime.now(timezone.utc) + timedelta(minutes=minutes)
                ).isoformat(timespec="seconds")
            self.memory.add_reminder(
                text=text,
                trigger_at=trigger_at,
                chat_id=chat_state.chat_id,
                persona_key=self.persona.key,
                user_id=user_id,
                metadata={"source": "telegram_decision"},
            )
            added += 1
        return added

    def _remember_persona_self_fact(self, text: str) -> None:
        if not self.telegram_config.fictional_self_enabled:
            return
        text = text.strip()
        if not text:
            return
        prefix = f"{self.persona.display_name} про себе:"
        if not text.lower().startswith(prefix.lower()):
            text = f"{prefix} {text}"
        self.memory_service.remember(
            text,
            kind=KIND_PERSONA_SELF,
            scope=SCOPE_PERSONA,
            tags=[
                "telegram",
                TELEGRAM_GLOBAL_MEMORY_TAG,
                TELEGRAM_PERSONA_SELF_MEMORY_TAG,
                f"persona:{self.persona.key}",
            ],
            persona_key=self.persona.key,
            source="telegram_persona_self",
        )

    def _schedule_from_minutes(self, chat_id: str, minutes: int | None) -> None:
        minutes = minutes if minutes is not None else self.telegram_config.proactive_cooldown_seconds // 60
        minutes = max(15, min(minutes, 7 * 24 * 60))
        next_at = datetime.now(timezone.utc) + timedelta(minutes=minutes)
        self.memory.schedule_telegram_initiative(chat_id, next_at.isoformat(timespec="seconds"))

    def _initiative_cooldown_elapsed(self, chat: TelegramChat) -> bool:
        if not chat.last_initiative_at:
            return True
        try:
            last = datetime.fromisoformat(chat.last_initiative_at)
        except ValueError:
            return True
        return datetime.now(timezone.utc) - last >= timedelta(seconds=self.telegram_config.proactive_cooldown_seconds)

    def _limit_reply(self, text: str) -> str:
        text = text.strip()
        if len(text) <= self.telegram_config.max_reply_chars:
            return text
        return text[: self.telegram_config.max_reply_chars - 20].rstrip() + "\n...[скорочено]"

    def _chat_allowed(self, chat_id: str) -> bool:
        allowed = self.telegram_config.allowed_chat_ids
        return not allowed or chat_id in allowed

    def _style_payload(self, chat: TelegramChat) -> dict[str, Any]:
        choice = self._style_tuner.choose(chat.chat_id)
        return choice.payload

    def _should_skip_without_llm(self, chat: TelegramChat, addressed: bool) -> bool:
        mode = chat.reply_mode or self.telegram_config.reply_mode
        if mode == "silent":
            return True
        if chat.chat_type != "private" and mode == "mention" and not addressed:
            return True
        return False

    def _is_addressed(self, text: str, message: dict[str, Any]) -> bool:
        lower = text.lower()
        if any(name in lower for name in self.persona.aliases):
            return True
        if self.bot_username and f"@{self.bot_username.lower()}" in lower:
            return True
        if text.startswith("/"):
            return True
        reply = message.get("reply_to_message") or {}
        reply_from = reply.get("from") or {}
        return bool(self.bot_username and str(reply_from.get("username", "")).lower() == self.bot_username.lower())

    # ------------------------------------------------------------------
    # Memory recall

    def _search_chat_memory(
        self,
        chat: TelegramChat,
        query: str,
        *,
        user_id: str | None = None,
    ):
        # By default Telegram memory is intentionally global across chats and
        # personas. When privacy mode is enabled, user-scoped memories are
        # only recalled for the originating Telegram user.
        private_mode = not self.telegram_config.global_memory
        results = self.memory_service.recall(
            RecallQuery(
                text=query,
                user_id=user_id if private_mode else None,
                chat_id=chat.chat_id if private_mode else None,
                require_tags=(TELEGRAM_GLOBAL_MEMORY_TAG,),
                limit=self.telegram_config.max_memory_facts,
                include_global=not private_mode,
            )
        )
        if len(results) >= self.telegram_config.max_memory_facts:
            return [self._fact_view(result.item) for result in results]

        existing = {result.item.id for result in results}
        fallback = self.memory_service.recall(
            RecallQuery(
                text=query,
                user_id=user_id if private_mode else None,
                chat_id=chat.chat_id if private_mode else None,
                require_tags=("telegram",),
                limit=self.telegram_config.max_memory_facts * 2,
                include_global=not private_mode,
            )
        )
        for result in fallback:
            if result.item.id in existing:
                continue
            results.append(result)
            existing.add(result.item.id)
            if len(results) >= self.telegram_config.max_memory_facts:
                break
        return [self._fact_view(result.item) for result in results]

    def _persona_self_context(self, query: str):
        if not self.telegram_config.fictional_self_enabled:
            return []
        results = self.memory_service.recall(
            RecallQuery(
                text=query,
                persona_key=self.persona.key,
                require_tags=(TELEGRAM_PERSONA_SELF_MEMORY_TAG, f"persona:{self.persona.key}"),
                limit=self.telegram_config.max_memory_facts,
            )
        )
        items = [result.item for result in results]
        seen = {item.id for item in items}
        if len(items) < self.telegram_config.max_memory_facts:
            recent = self.memory.recent_tagged_all(
                [TELEGRAM_PERSONA_SELF_MEMORY_TAG, f"persona:{self.persona.key}"],
                limit=self.telegram_config.max_memory_facts,
            )
            for fact in recent:
                if fact.id in seen:
                    continue
                stored = self.memory.get_memory(fact.id)
                if stored is None:
                    continue
                items.append(stored)
                seen.add(stored.id)
                if len(items) >= self.telegram_config.max_memory_facts:
                    break
        return [self._fact_view(item) for item in items[: self.telegram_config.max_memory_facts]]

    @staticmethod
    def _fact_view(item):
        from ..memory import MemoryFact

        return MemoryFact(
            id=item.id,
            text=item.text,
            tags=list(item.tags),
            created_at=item.created_at,
            importance=item.importance,
            kind=item.kind,
            scope=item.scope,
        )

    # ------------------------------------------------------------------
    # Helpers

    @staticmethod
    def _user_id_for(user: dict[str, Any]) -> str | None:
        identifier = user.get("id") if isinstance(user, dict) else None
        if identifier in (None, ""):
            return None
        return f"telegram:{identifier}"

    def _upsert_user_profile(self, user: dict[str, Any]) -> None:
        if not isinstance(user, dict) or not user.get("id"):
            return
        user_id = self._user_id_for(user)
        if user_id is None:
            return
        self.memory.upsert_user(
            user_id,
            display_name=display_sender(user),
            language=str(user.get("language_code") or "") or None,
            metadata={"telegram_user": user},
        )

    def thread_id(self, chat_id: str | int) -> str:
        return f"{TELEGRAM_CHAT_THREAD_PREFIX}:{chat_id}"

    @staticmethod
    def chat_tag(chat_id: str | int) -> str:
        return f"telegram_chat_{chat_id}"

    def _recent_compact_messages(self, chat: TelegramChat) -> list[dict[str, str]]:
        messages = self.memory.recent_messages(
            self.thread_id(chat.chat_id),
            limit=self.telegram_config.max_history_messages,
        )
        cleaned: list[dict[str, str]] = []
        for item in messages:
            role = str(item.get("role", ""))
            content = str(item.get("content", ""))
            if role == "assistant":
                content = self._clean_reply_text(content)
            if content:
                cleaned.append({"role": role, "content": content})
        return cleaned

    def _recent_telegram_messages(self, chat_id: str | int) -> list[dict[str, Any]]:
        messages = self.memory.recent_telegram_messages(
            chat_id,
            limit=self.telegram_config.max_history_messages,
        )
        cleaned: list[dict[str, Any]] = []
        for item in messages:
            copied = dict(item)
            if copied.get("role") == "assistant":
                copied["text"] = self._clean_reply_text(str(copied.get("text", "")))
            cleaned.append(copied)
        return cleaned

    @staticmethod
    def _history_user_content(chat: TelegramChat, sender: str, text: str) -> str:
        if chat.chat_type == "private":
            return f"{sender}: {text}"
        return f"{sender} in {chat.display_name}: {text}"

    def _history_assistant_content(self, chat: TelegramChat, text: str) -> str:
        return self._clean_reply_text(text)

    def _clean_reply_text(self, text: str) -> str:
        cleaned = strip_speaker_prefixes(
            clean_model_content(str(text or "")).strip(),
            [
                self.persona.display_name,
                self.telegram_config.bot_name,
                self.bot_username,
                *self.persona.aliases,
            ],
        )
        return strip_assistanty_phrases(cleaned)

    @staticmethod
    def _is_generic_sticker_filler(text: str) -> bool:
        value = text.strip()
        return bool(value and len(value) <= 160 and GENERIC_STICKER_FILLER_RE.search(value))

    @staticmethod
    def _incoming_text_with_media(
        text: str,
        image: ImageAttachment | None,
        description: str,
        sticker: StickerAttachment | None,
        voice: VoiceAttachment | None = None,
        voice_transcript: str = "",
    ) -> str:
        text = text.strip()
        parts = [text] if text else []
        if image is not None:
            image_label = image.file_name or image.label
            parts.append(f"[зображення: {description or image_label}]")
        if sticker is not None:
            bits = [sticker.kind]
            if sticker.emoji:
                bits.append(sticker.emoji)
            if sticker.set_name:
                bits.append(f"pack={sticker.set_name}")
            parts.append("[стікер: " + ", ".join(bits) + "]")
        if voice is not None:
            transcript = voice_transcript.strip()
            if transcript:
                parts.append(f"[голос: {transcript}]")
            else:
                parts.append(f"[голос: {voice.label}, {voice.duration}s, transcription unavailable]")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Prompts

    def _decision_system_prompt(self) -> str:
        return decision_system_prompt(
            self.persona,
            fictional_self_enabled=self.telegram_config.fictional_self_enabled,
        )

    def _reply_system_prompt(self) -> str:
        return reply_system_prompt(
            self.persona,
            fictional_self_enabled=self.telegram_config.fictional_self_enabled,
        )

    def _initiative_system_prompt(self) -> str:
        return initiative_system_prompt(
            self.persona,
            fictional_self_enabled=self.telegram_config.fictional_self_enabled,
        )

    # ------------------------------------------------------------------
    # Errors / log rotation

    def _log_loop_exception(self, exc: BaseException) -> None:
        try:
            self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
            self._rotate_error_log()
            with self.error_log_path.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {type(exc).__name__}: {exc}\n")
                handle.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
        except OSError:
            pass

    def _rotate_error_log(self) -> None:
        try:
            if self.error_log_path.exists() and self.error_log_path.stat().st_size > _TELEGRAM_LOG_TAIL_BYTES * 5:
                tail = self.error_log_path.read_bytes()[-_TELEGRAM_LOG_TAIL_BYTES:]
                self.error_log_path.write_bytes(tail)
        except OSError:
            pass


def build_nikola_bot(
    *,
    agent_config: AgentConfig,
    telegram_config: TelegramConfig,
) -> NikolaBot:
    memory = MemoryStore(agent_config.database_path)
    llm = OpenAICompatibleClient(agent_config.base_url, agent_config.model)
    telegram = TelegramApi(telegram_config.token)
    embedding_config = EmbeddingConfig(
        base_url=agent_config.embedding.base_url,
        model=agent_config.embedding.model,
        timeout_seconds=agent_config.embedding.timeout_seconds,
        request_dimensions=agent_config.embedding.request_dimensions,
    )
    embedding_client = EmbeddingClient(embedding_config) if embedding_config.enabled else None
    memory_service = MemoryService(
        memory,
        embedding_client=embedding_client,
        embedding_backend=agent_config.embedding.backend,
        importance_client=llm if agent_config.llm_importance else None,
        llm_importance=agent_config.llm_importance,
    )
    return NikolaBot(
        telegram=telegram,
        llm=llm,
        memory=memory,
        telegram_config=telegram_config,
        agent_config=agent_config,
        memory_service=memory_service,
    )


def _decision_to_payload(decision: Decision) -> dict[str, Any]:
    return {
        "should_reply": decision.should_reply,
        "reply": decision.reply,
        "replies": list(decision.replies),
        "reply_to": decision.reply_to,
        "stickers": list(decision.stickers),
        "memories": list(decision.memories),
        "self_memories": list(decision.self_memories),
        "reminders": list(decision.reminders),
        "tool_request": decision.tool_request,
        "next_check_minutes": decision.next_check_minutes,
    }


def _tool_event_payload(event: TelegramToolEvent) -> dict[str, Any]:
    return {
        "name": event.name,
        "arguments": event.arguments,
        "result": event.result,
    }


__all__ = ["NikolaBot", "build_nikola_bot"]
