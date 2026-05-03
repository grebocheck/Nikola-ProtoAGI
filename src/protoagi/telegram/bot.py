"""Telegram conversation bot orchestration.

This module composes the small focused modules in :mod:`protoagi.telegram`
into the public ``NikolaBot`` class. The bot is intentionally synchronous to
match the local llama.cpp deployment, which generates one reply at a time.
"""

from __future__ import annotations

import hashlib
import json
import time
import traceback
from datetime import datetime, timedelta, timezone
from typing import Any

from ..config import AgentConfig, PROJECT_ROOT
from ..embedding import EmbeddingClient, EmbeddingConfig
from ..harmony import clean_model_content
from ..memory import (
    KIND_FACT,
    KIND_PERSONA_SELF,
    MemoryStore,
    SCOPE_GLOBAL,
    SCOPE_PERSONA,
    TelegramChat,
    utc_now,
)
from ..memory_service import MemoryService, RecallQuery
from ..openai_compat import OpenAICompatError, OpenAICompatibleClient
from ..persona import PersonaProfile, get_persona

from .api import TelegramApi, TelegramApiError, is_telegram_polling_conflict
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
    Decision,
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
    auto_sticker_choice,
    looks_serious_for_sticker,
    normalize_sticker_pack,
)
from .text import (
    GENERIC_STICKER_FILLER_RE,
    display_sender,
    parse_command,
    split_telegram_message,
    strip_assistanty_phrases,
    strip_speaker_prefixes,
)
from .vision import VisionDescriber


_TELEGRAM_LOG_TAIL_BYTES = 1_000_000


class NikolaBot:
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
        self._sticker_cache: dict[str, list[dict[str, str]]] = {}
        self.error_log_path = PROJECT_ROOT / "runs" / "telegram-errors.log"
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
        )
        if memory_service is None:
            embedding_config = EmbeddingConfig(
                base_url=agent_config.embedding.base_url,
                model=agent_config.embedding.model,
                timeout_seconds=agent_config.embedding.timeout_seconds,
                request_dimensions=agent_config.embedding.request_dimensions,
            )
            embedding_client = EmbeddingClient(embedding_config) if embedding_config.enabled else None
            memory_service = MemoryService(memory, embedding_client=embedding_client)
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
            allowed_updates=["message"],
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

    # ------------------------------------------------------------------
    # Update handling

    def process_update(self, update: dict[str, Any]) -> bool:
        message = update.get("message")
        if not isinstance(message, dict):
            return False
        text = str(message.get("text") or message.get("caption") or "").strip()
        image = self._extract_image_attachment(message)
        incoming_sticker = self._extract_sticker_attachment(message)
        if not text and image is None and incoming_sticker is None:
            return False
        chat = message.get("chat") or {}
        if "id" not in chat:
            return False
        chat_id = str(chat["id"])
        if not self._chat_allowed(chat_id):
            return False

        user = message.get("from") or {}
        chat_state = self.memory.upsert_telegram_chat(
            chat,
            user,
            reply_mode=self.telegram_config.reply_mode,
        )
        self.memory.mark_telegram_user_message(chat_id)
        self._upsert_user_profile(user)

        current_message_id = int(message.get("message_id", 0))
        thread_id = self.thread_id(chat_id)
        display = display_sender(user)
        image_description = self._vision.describe(image, caption=text) if image is not None else ""
        incoming_text = self._incoming_text_with_media(text, image, image_description, incoming_sticker)
        content = self._history_user_content(chat_state, display, incoming_text)
        self.memory.log_message(thread_id, "user", content)
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

        decision = self.decide_incoming(chat_state, incoming_text, display, addressed)
        for fact in decision.memories:
            self._remember_chat_fact(chat_state, fact, user_id=self._user_id_for(user))
        for fact in decision.self_memories:
            self._remember_persona_self_fact(fact)
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

    # ------------------------------------------------------------------
    # Decision pipeline

    def decide_incoming(
        self,
        chat: TelegramChat,
        incoming_text: str,
        sender: str,
        addressed: bool,
    ) -> Decision:
        recent = self._recent_compact_messages(chat)
        recent_telegram = self._recent_telegram_messages(chat.chat_id)
        facts = self._search_chat_memory(chat, incoming_text)
        persona_self_memory = self._persona_self_context(incoming_text)
        messages = [
            {"role": "system", "content": self._decision_system_prompt()},
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
                        "available_sticker_packs": STICKER_PACKS,
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        response = self.llm.chat_completion(
            messages,
            temperature=self.agent_config.temperature,
            top_p=self.agent_config.top_p,
            max_tokens=self.telegram_config.decision_max_tokens,
        )
        content = clean_model_content(response.get("choices", [{}])[0].get("message", {}).get("content", ""))
        payload = extract_json_object(content)
        decision = decision_from_payload(payload)
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
        if "[зображення:" in incoming_text:
            decision.reply = "" if is_image_blind_reply(decision.reply) else decision.reply
            decision.replies = [item for item in decision.replies if not is_image_blind_reply(item)]
            if not decision_reply_texts(decision) and not decision.stickers:
                decision.should_reply = False
        if decision.should_reply and not decision_reply_texts(decision) and not decision.stickers:
            decision.should_reply = False
        return decision

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
                            "available_sticker_packs": STICKER_PACKS,
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
    # Stickers

    def _maybe_add_reaction_sticker(self, chat: TelegramChat, incoming_text: str, decision: Decision) -> None:
        if decision.stickers or chat.chat_type != "private":
            return
        if not decision.should_reply or not decision_reply_texts(decision):
            return
        frequency = self.telegram_config.sticker_frequency
        if frequency == "off" or looks_serious_for_sticker(incoming_text):
            return
        choice = auto_sticker_choice(incoming_text, " ".join(decision_reply_texts(decision)))
        if not choice:
            return
        if self._recent_sticker_count(chat.chat_id) >= 1:
            return
        thresholds = {"low": 12, "normal": 25, "high": 40, "always": 100}
        threshold = thresholds.get(frequency, 25)
        seed = f"{chat.chat_id}|{incoming_text}|{decision.reply}|{'|'.join(decision.replies)}"
        bucket = int(hashlib.sha1(seed.encode("utf-8")).hexdigest()[:8], 16) % 100
        if bucket >= threshold:
            return
        decision.stickers.append(choice)

    def _recent_sticker_count(self, chat_id: str | int) -> int:
        limit = max(6, self.telegram_config.sticker_cooldown_messages * 3)
        messages = self.memory.recent_telegram_messages(chat_id, limit=limit)
        user_messages_since_sticker = 0
        for item in reversed(messages):
            if str(item.get("text", "")).startswith("[sticker:"):
                return 1 if user_messages_since_sticker < self.telegram_config.sticker_cooldown_messages else 0
            if item.get("role") == "user":
                user_messages_since_sticker += 1
        return 0

    def _select_sticker_file_id(self, choice: dict[str, str], chat_id: str) -> str | None:
        pack = normalize_sticker_pack(choice.get("pack", ""))
        if not pack:
            return None
        stickers = self._load_sticker_pack(pack)
        if not stickers:
            return None
        emoji = str(choice.get("emoji", "") or "").strip()
        candidates = [item for item in stickers if emoji and emoji in item.get("emoji", "")]
        if not candidates:
            candidates = stickers
        seed = f"{chat_id}:{pack}:{emoji}:{choice.get('reason', '')}".encode("utf-8")
        digest = hashlib.sha256(seed).digest()
        index = int.from_bytes(digest[:4], "big") % len(candidates)
        return candidates[index].get("file_id")

    def _load_sticker_pack(self, pack: str) -> list[dict[str, str]]:
        if pack in self._sticker_cache:
            return self._sticker_cache[pack]
        key = f"telegram:stickers:{pack}"
        cached = self.memory.get_kv(key)
        if cached:
            try:
                stickers = json.loads(cached)
                if isinstance(stickers, list):
                    self._sticker_cache[pack] = stickers
                    return stickers
            except json.JSONDecodeError:
                pass
        try:
            sticker_set = self.telegram.get_sticker_set(pack)
        except TelegramApiError:
            self._sticker_cache[pack] = []
            return []
        stickers = [
            {
                "file_id": str(item.get("file_id", "")),
                "emoji": str(item.get("emoji", "") or ""),
            }
            for item in sticker_set.get("stickers", [])
            if item.get("file_id")
        ]
        self.memory.set_kv(key, json.dumps(stickers, ensure_ascii=False))
        self._sticker_cache[pack] = stickers
        return stickers

    # ------------------------------------------------------------------
    # Image / sticker incoming attachments

    def _extract_image_attachment(self, message: dict[str, Any]) -> ImageAttachment | None:
        photos = message.get("photo")
        if isinstance(photos, list) and photos:
            photo = max(
                (item for item in photos if isinstance(item, dict) and item.get("file_id")),
                key=lambda item: int(item.get("file_size") or item.get("width", 0) * item.get("height", 0) or 0),
                default=None,
            )
            if photo:
                return ImageAttachment(
                    file_id=str(photo["file_id"]),
                    mime_type="image/jpeg",
                    label="photo",
                )

        document = message.get("document")
        if isinstance(document, dict):
            mime_type = str(document.get("mime_type") or "")
            file_id = str(document.get("file_id") or "")
            if file_id and mime_type.startswith("image/"):
                return ImageAttachment(
                    file_id=file_id,
                    mime_type=mime_type,
                    label="image document",
                    file_name=str(document.get("file_name") or ""),
                )
        return None

    @staticmethod
    def _extract_sticker_attachment(message: dict[str, Any]) -> StickerAttachment | None:
        sticker = message.get("sticker")
        if not isinstance(sticker, dict):
            return None
        file_id = str(sticker.get("file_id") or "")
        if not file_id:
            return None
        if sticker.get("is_video"):
            kind = "video sticker"
        elif sticker.get("is_animated"):
            kind = "animated sticker"
        else:
            kind = "sticker"
        return StickerAttachment(
            file_id=file_id,
            emoji=str(sticker.get("emoji") or ""),
            set_name=str(sticker.get("set_name") or ""),
            kind=kind,
        )

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
        self.memory_service.remember(
            text,
            kind=KIND_FACT,
            scope=SCOPE_GLOBAL,
            tags=tags,
            user_id=user_id,
            chat_id=chat.chat_id,
            persona_key=self.persona.key,
            source="telegram_chat",
        )

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

    def _search_chat_memory(self, chat: TelegramChat, query: str):
        # Telegram memory is intentionally global across chats and personas;
        # do not filter by chat_id here.
        results = self.memory_service.recall(
            RecallQuery(
                text=query,
                require_tags=(TELEGRAM_GLOBAL_MEMORY_TAG,),
                limit=self.telegram_config.max_memory_facts,
            )
        )
        if len(results) >= self.telegram_config.max_memory_facts:
            return [self._fact_view(result.item) for result in results]

        existing = {result.item.id for result in results}
        fallback = self.memory_service.recall(
            RecallQuery(
                text=query,
                require_tags=("telegram",),
                limit=self.telegram_config.max_memory_facts * 2,
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
    memory_service = MemoryService(memory, embedding_client=embedding_client)
    return NikolaBot(
        telegram=telegram,
        llm=llm,
        memory=memory,
        telegram_config=telegram_config,
        agent_config=agent_config,
        memory_service=memory_service,
    )


__all__ = ["NikolaBot", "build_nikola_bot"]
