import json
import os
from pathlib import Path
import tempfile
import unittest

# Tests use FakeTelegram + offline endpoints, so suppress the bootstrap probe
# warnings (and their per-target ~1 s timeout) globally for this module.
os.environ.setdefault("PROTOAGI_BOOTSTRAP_PROBE", "0")

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.telegram import (
    NikolaBot,
    TELEGRAM_GLOBAL_MEMORY_TAG,
    TELEGRAM_PERSONA_SELF_MEMORY_TAG,
    TelegramConfig,
    VoiceTranscriptionResult,
    clean_vision_description,
    decision_from_payload,
    decision_reply_texts,
    extract_json_object,
    honest_identity_reply,
    is_image_blind_reply,
    is_identity_question,
    is_telegram_polling_conflict,
    normalize_sticker_pack,
    parse_command,
    split_telegram_message,
)


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.stickers: list[dict] = []
        self.voices: list[dict] = []
        self.actions: list[dict] = []
        self.updates: list[dict] = []
        self.reactions: list[dict] = []
        # When set, ``set_message_reaction`` raises this error once and
        # then resets — used to simulate REACTION_INVALID for denylist tests.
        self.next_reaction_error: str | None = None

    def get_me(self) -> dict:
        return {"id": 42, "username": "NikolaTestBot", "first_name": "Микола"}

    def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        return True

    def get_updates(self, *, offset, timeout_seconds, allowed_updates):
        return self.updates

    def send_chat_action(self, chat_id, action="typing") -> bool:
        self.actions.append({"chat_id": chat_id, "action": action})
        return True

    def send_message(self, chat_id, text, *, reply_to_message_id=None, disable_notification=False):
        self.sent.append(
            {
                "chat_id": str(chat_id),
                "text": text,
                "reply_to_message_id": reply_to_message_id,
                "disable_notification": disable_notification,
            }
        )
        return {"message_id": len(self.sent)}

    def get_sticker_set(self, name):
        return {
            "name": name,
            "stickers": [
                {"file_id": f"{name}:smile", "emoji": "🙂"},
                {"file_id": f"{name}:spark", "emoji": "✨"},
            ],
        }

    def get_file(self, file_id):
        return {"file_id": file_id, "file_path": f"photos/{file_id}.jpg"}

    def download_file(self, file_path, *, max_bytes):
        return b"fake-image-bytes"

    def set_message_reaction(self, chat_id, message_id, emoji, *, is_big=False):
        if self.next_reaction_error is not None:
            err = self.next_reaction_error
            self.next_reaction_error = None
            from protoagi.telegram.api import TelegramApiError

            raise TelegramApiError(err)
        self.reactions.append(
            {
                "chat_id": str(chat_id),
                "message_id": message_id,
                "emoji": emoji,
                "is_big": is_big,
            }
        )
        return True

    def send_sticker(self, chat_id, sticker, *, reply_to_message_id=None, disable_notification=False):
        self.stickers.append(
            {
                "chat_id": str(chat_id),
                "sticker": sticker,
                "reply_to_message_id": reply_to_message_id,
                "disable_notification": disable_notification,
            }
        )
        return {"message_id": 100 + len(self.stickers)}

    def send_voice_bytes(
        self,
        chat_id,
        data,
        *,
        filename="reply.ogg",
        mime_type="audio/ogg",
        reply_to_message_id=None,
        disable_notification=False,
    ):
        self.voices.append(
            {
                "chat_id": str(chat_id),
                "data": data,
                "filename": filename,
                "mime_type": mime_type,
                "reply_to_message_id": reply_to_message_id,
                "disable_notification": disable_notification,
            }
        )
        return {"message_id": 200 + len(self.voices)}

    def send_audio_bytes(
        self,
        chat_id,
        data,
        *,
        filename="reply.mp3",
        mime_type="audio/mpeg",
        reply_to_message_id=None,
        disable_notification=False,
    ):
        if not hasattr(self, "audios"):
            self.audios = []
        self.audios.append(
            {
                "chat_id": str(chat_id),
                "data": data,
                "filename": filename,
                "mime_type": mime_type,
                "reply_to_message_id": reply_to_message_id,
                "disable_notification": disable_notification,
            }
        )
        return {"message_id": 300 + len(self.audios)}


class FakeLLM:
    def __init__(self, content: str | list[str]) -> None:
        self.content = content[-1] if isinstance(content, list) and content else content
        self._queue = list(content) if isinstance(content, list) else None
        self.messages = []
        self.kwargs = []

    def chat_completion(self, messages, **kwargs):
        self.messages.append(messages)
        self.kwargs.append(kwargs)
        content = self._queue.pop(0) if self._queue is not None and self._queue else self.content
        return {"choices": [{"message": {"content": content}}]}


class FakeVisionLLM(FakeLLM):
    def server_props(self):
        return {"media_marker": "<dynamic-media-marker>"}


class FakeVoiceTranscriber:
    def __init__(self, text: str) -> None:
        self.text = text

    def transcribe(self, attachment):
        return self.text


class FakeVoiceTranscriberWithBytes(FakeVoiceTranscriber):
    def __init__(self, text: str, data: bytes) -> None:
        super().__init__(text)
        self.data = data

    def transcribe_with_bytes(self, attachment):
        return VoiceTranscriptionResult(
            text=self.text,
            data=self.data,
            mime_type=attachment.mime_type,
            file_id=attachment.file_id,
        )


class FakeTTS:
    def __init__(self, data: bytes, response_format: str = "opus") -> None:
        self.data = data
        self._response_format = response_format
        self.last_error: str | None = None
        self.last_voice: str | None = None
        self.calls = 0

    def synthesize(self, text: str, *, voice: str | None = None):
        self.calls += 1
        self.last_voice = voice
        return self.data

    @property
    def response_format(self) -> str:
        return self._response_format

    @property
    def expected_mime(self) -> str:
        return {
            "opus": "audio/ogg",
            "ogg": "audio/ogg",
            "mp3": "audio/mpeg",
        }.get(self._response_format, "application/octet-stream")


class FakeFailingTTS(FakeTTS):
    def __init__(self, last_error: str = "boom") -> None:
        super().__init__(b"")
        self.last_error = last_error

    def synthesize(self, text: str, *, voice: str | None = None):
        self.calls += 1
        self.last_voice = voice
        return None


class TelegramBotTests(unittest.TestCase):
    def test_parse_command_with_bot_suffix(self) -> None:
        self.assertEqual(parse_command("/remember@NikolaTestBot hello", "NikolaTestBot"), ("remember", "hello"))
        self.assertEqual(parse_command("/remember@OtherBot hello", "NikolaTestBot"), (None, ""))

    def test_extract_json_object_from_fenced_text(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"should_reply": true}\n```'), {"should_reply": True})

    def test_decision_from_payload(self) -> None:
        decision = decision_from_payload(
            {
                "should_reply": True,
                "reply": "Так",
                "replies": ["ага", "сек"],
                "reply_to": "current",
                "voice_reply": True,
                "stickers": [{"pack": "miku", "emoji": "✨", "reason": "playful"}],
                "memories": ["любить чай"],
                "self_memories": ["люблю лимонний чай"],
            }
        )
        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.replies, ["ага", "сек"])
        self.assertEqual(decision_reply_texts(decision), ["ага", "сек"])
        self.assertEqual(decision.reply_to, "current")
        self.assertEqual(decision.stickers[0]["pack"], "M1ku_Hatsune")
        self.assertTrue(decision.voice_reply)
        self.assertEqual(decision.memories, ["любить чай"])
        self.assertEqual(decision.self_memories, ["люблю лимонний чай"])

    def test_normalize_sticker_pack_alias(self) -> None:
        self.assertEqual(normalize_sticker_pack("senko"), "SenkoSan")

    def test_telegram_polling_conflict_detection(self) -> None:
        self.assertTrue(is_telegram_polling_conflict("Telegram HTTP 409: getUpdates request"))
        self.assertFalse(is_telegram_polling_conflict("Telegram HTTP 429: retry later"))

    def test_image_blind_reply_detection(self) -> None:
        self.assertTrue(is_image_blind_reply("Пробач, але я не бачу зображення. Що там?"))
        self.assertTrue(is_image_blind_reply("I can't see the image"))
        self.assertFalse(is_image_blind_reply("На фото схоже біла чашка."))

    def test_identity_question_guard_replaces_deceptive_human_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Так, я людина, просто пишу з телефону.", '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 9,
                    "message": {
                        "message_id": 19,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "ти людина?",
                    },
                }
            )
            self.assertTrue(is_identity_question("ти людина?"))
            self.assertEqual(telegram.sent[0]["text"], honest_identity_reply(bot.persona))

    def test_clean_vision_description_strips_boilerplate_and_repetition(self) -> None:
        cleaned = clean_vision_description(
            "The image you've provided is a white mug with a red ladybug logo. "
            "If you have any other questions or need further clarification, please let me know!"
        )
        self.assertEqual(cleaned, "a white mug with a red ladybug logo.")
        self.assertEqual(clean_vision_description("що є чело " * 20), "опис недоступний")

    def test_clean_vision_description_drops_runaway_trailing_fragment(self) -> None:
        # SmolVLM2 hit max_tokens mid-word — the previous version
        # stitched the partial fragment into the caption.  We now drop
        # the trailing incomplete sentence entirely.
        text = (
            "На стікері аніме-дівчина зашарілася. "
            "Починає говорити \"Якщо ви бачите цей текст, я виконую його д"
        )
        cleaned = clean_vision_description(text)
        self.assertEqual(cleaned, "На стікері аніме-дівчина зашарілася.")
        self.assertNotIn("викон", cleaned)

    def test_clean_vision_description_completes_missing_period(self) -> None:
        # Single sentence that ends on a full word but lacks a terminator
        # (very common with SmolVLM2 output): treat as "model forgot the
        # period", append ``.`` so downstream validation does not mistake
        # it for a mid-word truncation.
        cleaned = clean_vision_description("Дівчина усміхається і тримає чашку")
        self.assertTrue(cleaned.endswith("."), cleaned)
        self.assertFalse(cleaned.endswith("…"), cleaned)

    def test_clean_vision_description_marks_mid_word_truncation_with_ellipsis(self) -> None:
        # When the last token is short and looks like a cut-off fragment
        # (e.g. two-letter stub after a hyphen), keep the ``…`` marker so
        # consumers can tell it was a real loss.
        cleaned = clean_vision_description("Дівчина тримає чашку з на")
        self.assertTrue(cleaned.endswith("…"), cleaned)

    def test_clean_vision_description_can_keep_three_sticker_sentences(self) -> None:
        cleaned = clean_vision_description(
            "Miku smiles at the viewer. She raises both hands near her face. "
            "Small sparkles surround her.",
            max_sentences=3,
        )
        self.assertEqual(
            cleaned,
            "Miku smiles at the viewer. She raises both hands near her face. "
            "Small sparkles surround her.",
        )

    def test_split_telegram_message(self) -> None:
        chunks = split_telegram_message("a" * 20, max_chars=8)
        self.assertEqual(chunks, ["aaaaaaaa", "aaaaaaaa", "aaaa"])

    def test_process_update_replies_and_remembers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Привіт, я тут.", '
                '"memories": ["Користувач любить спокійні розмови"], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 10,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "Привіт, Миколо",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent[0]["text"], "Привіт, я тут.")
            self.assertIsNone(telegram.sent[0]["reply_to_message_id"])
            hits = memory.search_tagged_all("спокійні", [TELEGRAM_GLOBAL_MEMORY_TAG])
            self.assertEqual(len(hits), 1)
            chat = memory.get_telegram_chat("123")
            self.assertIsNotNone(chat)
            self.assertIsNotNone(chat.next_initiative_at)

    def test_reply_style_feedback_is_recorded_after_user_replies(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "first", "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 10,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "hello",
                    },
                }
            )
            bot.process_update(
                {
                    "update_id": 2,
                    "message": {
                        "message_id": 11,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "again",
                    },
                }
            )
            style = json.loads(memory.get_kv("telegram:style:123") or "{}")
            self.assertEqual(style["signals"]["reply"], 1)
            first_payload = json.loads(llm.messages[0][1]["content"])
            self.assertIn("adaptive_reply_style", first_payload)

    def test_privacy_mode_recalls_only_current_user_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            chat = memory.upsert_telegram_chat(
                {"id": -100, "type": "group", "title": "Lab"},
                {"id": 1, "first_name": "Alice"},
            )
            bot = NikolaBot(
                telegram=FakeTelegram(),
                llm=FakeLLM("{}"),
                memory=memory,
                telegram_config=TelegramConfig(token="token", global_memory=False),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot._remember_chat_fact(chat, "Alice likes espresso", user_id="telegram:1")
            bot._remember_chat_fact(chat, "Bob likes espresso", user_id="telegram:2")

            alice = bot._search_chat_memory(chat, "espresso", user_id="telegram:1")
            self.assertEqual([fact.text for fact in alice], ["Alice likes espresso"])

            global_bot = NikolaBot(
                telegram=FakeTelegram(),
                llm=FakeLLM("{}"),
                memory=MemoryStore(Path(tmp) / "global.sqlite3"),
                telegram_config=TelegramConfig(token="token", global_memory=True),
                agent_config=AgentConfig(database_path=Path(tmp) / "global.sqlite3"),
            )
            global_chat = global_bot.memory.upsert_telegram_chat(
                {"id": -100, "type": "group", "title": "Lab"},
                {"id": 1, "first_name": "Alice"},
            )
            global_bot._remember_chat_fact(global_chat, "Alice likes espresso", user_id="telegram:1")
            global_bot._remember_chat_fact(global_chat, "Bob likes espresso", user_id="telegram:2")
            texts = [fact.text for fact in global_bot._search_chat_memory(global_chat, "espresso")]
            self.assertIn("Alice likes espresso", texts)
            self.assertIn("Bob likes espresso", texts)

    def test_decision_tool_request_recall_merges_tool_result_into_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.remember(
                "User has a dangerous nut allergy",
                ["telegram", TELEGRAM_GLOBAL_MEMORY_TAG, "source_chat:123"],
            )
            telegram = FakeTelegram()
            llm = FakeLLM(
                [
                    '{"should_reply": true, "reply": "", '
                    '"tool_request": {"name": "recall", "arguments": {"query": "allergy", "limit": 3}}, '
                    '"memories": [], "next_check_minutes": 60}',
                    '{"should_reply": true, "reply": "I remember the nut allergy.", '
                    '"memories": [], "next_check_minutes": 60}',
                ]
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 101,
                    "message": {
                        "message_id": 501,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "what do you remember about allergies?",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertIn("nut allergy", telegram.sent[0]["text"])
            self.assertEqual(len(llm.messages), 1)
            self.assertNotIn("tools", llm.kwargs[0])
            metrics = json.loads(memory.get_kv("telegram:decision_metrics") or "{}")
            self.assertEqual(metrics["llm_call_histogram"], {"1": 1})
            self.assertEqual(metrics["tool_decisions"], 1)

    def test_process_update_stores_persona_self_memory_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "я таке люблю, дрібний домашній квест", '
                '"memories": [], "self_memories": ["люблю робити з побутових дрібниць маленькі історії"], '
                '"next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 11,
                    "message": {
                        "message_id": 20,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "йду забирати геймпад",
                    },
                }
            )
            self.assertTrue(processed)
            hits = memory.search_tagged_all(
                "побутових",
                [TELEGRAM_PERSONA_SELF_MEMORY_TAG, "persona:solomiya"],
            )
            self.assertEqual(len(hits), 1)
            self.assertTrue(hits[0].text.startswith("Соломія про себе:"))

    def test_persona_self_memory_is_included_in_decision_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.remember(
                "Соломія про себе: любить лимонний чай після дивних вечорів",
                ["telegram", TELEGRAM_PERSONA_SELF_MEMORY_TAG, "persona:solomiya"],
            )
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "лимонний чай, очевидно", '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 12,
                    "message": {
                        "message_id": 21,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "що ти любиш?",
                    },
                }
            )
            payload = json.loads(llm.messages[0][1]["content"])
            self.assertIn("persona_self_lore", payload)
            self.assertEqual(
                payload["known_persona_self_memory"][0]["text"],
                "Соломія про себе: любить лимонний чай після дивних вечорів",
            )

    def test_private_current_reply_is_ignored_and_self_prefix_is_removed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Соломія: Оце повідомлення прямо в точку.", '
                '"reply_to": "current", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 2,
                    "message": {
                        "message_id": 77,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "Соломіє, глянь сюди",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent[0]["text"], "Оце повідомлення прямо в точку.")
            self.assertIsNone(telegram.sent[0]["reply_to_message_id"])
            history = memory.recent_messages(bot.thread_id("123"), limit=5)
            self.assertEqual(history[-1]["content"], "Оце повідомлення прямо в точку.")

    def test_group_message_can_reply_to_current_and_send_sticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Оце повідомлення прямо в точку.", '
                '"reply_to": "current", '
                '"stickers": [{"pack": "SenkoSan", "emoji": "🙂", "reason": "warm"}], '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 2,
                    "message": {
                        "message_id": 77,
                        "chat": {"id": -100, "type": "group", "title": "Lab"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "Миколо, глянь сюди",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent[0]["reply_to_message_id"], 77)
            self.assertEqual(telegram.stickers[0]["sticker"], "SenkoSan:smile")

    def test_decision_can_send_multiple_messages_then_sticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "", '
                '"replies": ["оце так", "ну все, тепер треба тестити"], '
                '"reply_to": "current", '
                '"stickers": [{"pack": "M1ku_Hatsune", "emoji": "✨", "reason": "gamepad hype"}], '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 25,
                    "message": {
                        "message_id": 78,
                        "chat": {"id": -100, "type": "group", "title": "Lab"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "Соломіє, геймпад приїхав",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual([item["text"] for item in telegram.sent], ["оце так", "ну все, тепер треба тестити"])
            self.assertEqual(telegram.sent[0]["reply_to_message_id"], 78)
            self.assertIsNone(telegram.sent[1]["reply_to_message_id"])
            self.assertEqual(telegram.stickers[0]["sticker"], "M1ku_Hatsune:spark")
            self.assertIsNone(telegram.stickers[0]["reply_to_message_id"])

    def test_private_reply_can_get_auto_reaction_sticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ахах, звучить як план", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya", sticker_frequency="always"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 26,
                    "message": {
                        "message_id": 79,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "ахах ну погнали",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent[0]["text"], "ахах, звучить як план")
            self.assertEqual(telegram.stickers[0]["sticker"], "Bocchi_the_Rock_sticker_pack2:smile")

    def test_auto_reaction_sticker_requires_emotional_trigger(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ну так, звучить нормально", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya", sticker_frequency="always"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 27,
                    "message": {
                        "message_id": 80,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "я вийшов у магазин",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent[0]["text"], "ну так, звучить нормально")
            self.assertEqual(telegram.stickers, [])

    def test_auto_reaction_sticker_respects_recent_sticker_cooldown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.log_telegram_message(
                chat_id=123,
                message_id=78,
                persona_key="solomiya",
                role="assistant",
                sender_id=None,
                sender_name="Соломія",
                text="[sticker:Bocchi_the_Rock_sticker_pack2]",
            )
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ахах, ну от", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    persona_key="solomiya",
                    sticker_frequency="always",
                    sticker_cooldown_messages=3,
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 28,
                    "message": {
                        "message_id": 81,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "ахах ну погнали",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent[0]["text"], "ахах, ну от")
            self.assertEqual(telegram.stickers, [])

    def test_recent_sticker_count_counts_user_messages_since_sticker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.log_telegram_message(
                chat_id=123,
                message_id=70,
                persona_key="solomiya",
                role="assistant",
                sender_id=None,
                sender_name="Соломія",
                text="[sticker:SenkoSan]",
            )
            memory.log_telegram_message(
                chat_id=123,
                message_id=71,
                persona_key="solomiya",
                role="user",
                sender_id=123,
                sender_name="Vadim",
                text="ахах",
            )
            bot = NikolaBot(
                telegram=FakeTelegram(),
                llm=FakeLLM("{}"),
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    persona_key="solomiya",
                    sticker_cooldown_messages=3,
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            self.assertEqual(bot._recent_sticker_count(123), 1)

    def test_sticker_filler_text_is_suppressed_when_sticker_is_sent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Соломія: Ось ще один – сподіваюся, підняв настрій! 🎉", '
                '"reply_to": null, '
                '"stickers": [{"pack": "SenkoSan", "emoji": "🙂", "reason": "requested sticker"}], '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 22,
                    "message": {
                        "message_id": 90,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "кинь стікер",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent, [])
            self.assertEqual(telegram.stickers[0]["sticker"], "SenkoSan:smile")

    def test_photo_message_uses_vision_description(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "О, це той самий геймпад?", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya", vision_model="vision-test"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.vision_llm = FakeVisionLLM("На фото чорний геймпад у коробці.")
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 23,
                    "message": {
                        "message_id": 91,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "caption": "нарешті",
                        "photo": [
                            {"file_id": "small", "width": 90, "height": 90, "file_size": 1000},
                            {"file_id": "big", "width": 1200, "height": 900, "file_size": 2000},
                        ],
                    },
                }
            )
            self.assertTrue(processed)
            payload = llm.messages[0][1]["content"]
            vision_content = bot.vision_llm.messages[0][1]["content"]
            self.assertEqual(vision_content[0]["text"].count("<dynamic-media-marker>"), 1)
            self.assertIn("нарешті", payload)
            self.assertIn("На фото чорний геймпад у коробці.", payload)
            self.assertEqual(telegram.sent[0]["text"], "О, це той самий геймпад?")
            blob = memory.get_media_blob("big")
            self.assertIsNotNone(blob)
            assert blob is not None
            self.assertEqual(blob.bytes, b"fake-image-bytes")
            self.assertIn("геймпад", blob.caption)
            media_items = [item for item in memory.list_memories(limit=10) if item.media_id == "big"]
            self.assertTrue(media_items)
            self.assertIn("media", media_items[0].tags)

    def test_voice_message_uses_transcription_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "почула", "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", voice_model="whisper-stub"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot._voice = FakeVoiceTranscriberWithBytes("тестовий голосовий текст", b"voice-ogg")
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 234,
                    "message": {
                        "message_id": 914,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "voice": {"file_id": "voice-file", "duration": 3, "mime_type": "audio/ogg"},
                    },
                }
            )
            self.assertTrue(processed)
            payload = llm.messages[0][1]["content"]
            self.assertIn("тестовий голосовий текст", payload)
            voice_items = [item for item in memory.list_memories(limit=10) if "voice" in item.tags]
            self.assertTrue(voice_items)
            self.assertIn("тестовий голосовий текст", voice_items[0].text)
            self.assertEqual(voice_items[0].media_id, "voice-file")
            blob = memory.get_media_blob("voice-file")
            self.assertIsNotNone(blob)
            assert blob is not None
            self.assertEqual(blob.bytes, b"voice-ogg")
            self.assertEqual(blob.mime, "audio/ogg")

    def test_tts_auto_prefers_text_unless_voice_requested(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "voice reply", "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-stub",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            fake_tts = FakeTTS(b"voice-bytes")
            bot._tts = fake_tts
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 235,
                    "message": {
                        "message_id": 915,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "say it",
                    },
                }
            )
            self.assertEqual(telegram.sent[0]["text"], "voice reply")
            self.assertEqual(telegram.voices, [])
            self.assertEqual(fake_tts.calls, 0)

    def test_tts_auto_sends_voice_when_decision_requests_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "voice reply", "voice_reply": true, "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-stub",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot._tts = FakeTTS(b"voice-bytes")
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 735,
                    "message": {
                        "message_id": 925,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "say it",
                    },
                }
            )
            self.assertEqual(telegram.voices[0]["data"], b"voice-bytes")
            self.assertEqual(telegram.sent, [])

    def test_tts_auto_cooldown_keeps_later_reply_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                [
                    '{"should_reply": true, "reply": "first voice", "voice_reply": true, "memories": [], "next_check_minutes": 60}',
                    '{"should_reply": true, "reply": "second voice", "voice_reply": true, "memories": [], "next_check_minutes": 60}',
                ]
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-stub",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            fake_tts = FakeTTS(b"voice-bytes")
            bot._tts = fake_tts
            bot.bootstrap()
            for update_id, message_id, text in (
                (736, 926, "say it once"),
                (737, 927, "say it twice"),
            ):
                bot.process_update(
                    {
                        "update_id": update_id,
                        "message": {
                            "message_id": message_id,
                            "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                            "from": {"id": 123, "first_name": "Vadim"},
                            "text": text,
                        },
                    }
                )
            self.assertEqual(len(telegram.voices), 1)
            self.assertEqual(telegram.sent[0]["text"], "second voice")
            self.assertEqual(fake_tts.calls, 1)

    def test_tts_text_and_voice_delivery_keeps_transcript_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "voice reply", "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-stub",
                    tts_delivery="text_and_voice",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot._tts = FakeTTS(b"voice-bytes")
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 735,
                    "message": {
                        "message_id": 925,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "say it",
                    },
                }
            )
            self.assertEqual(telegram.sent[0]["text"], "voice reply")
            self.assertEqual(telegram.voices[0]["data"], b"voice-bytes")

    def test_tts_voice_delivery_falls_back_to_text_when_synthesis_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "fallback text", "voice_reply": true, "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-stub",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot._tts = FakeFailingTTS()
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 736,
                    "message": {
                        "message_id": 926,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "say it",
                    },
                }
            )
            self.assertEqual(telegram.voices, [])
            self.assertEqual(telegram.sent[0]["text"], "fallback text")

    def test_tts_uses_persona_voice_when_set(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ага", "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    persona_key="solomiya",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-1-hd",
                    tts_voice="nova",
                    tts_delivery="voice",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            fake_tts = FakeTTS(b"voice-bytes")
            bot._tts = fake_tts
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 720,
                    "message": {
                        "message_id": 920,
                        "chat": {"id": 130, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 130, "first_name": "Vadim"},
                        "text": "що там",
                    },
                }
            )
            # Persona is solomiya, so the persona's tts_voice ("solomiya") wins
            # over the global TelegramConfig fallback ("nova").
            self.assertEqual(fake_tts.last_voice, "solomiya")

    def test_tts_falls_back_to_config_voice_when_persona_has_none(self) -> None:
        from protoagi.persona import PersonaProfile

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ок", "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-1-hd",
                    tts_voice="nova",
                    tts_delivery="voice",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            # Override the persona instance to one without tts_voice.
            bot.persona = PersonaProfile(
                key="anon",
                display_name="Anon",
                memory_tag="anon",
                aliases=("anon",),
                self_model="",
                user_model="",
                relationship_model="",
                decision_style=(),
                reply_style=(),
                memory_policy=(),
                initiative_policy=(),
                start_message="",
                tts_voice="",
            )
            fake_tts = FakeTTS(b"voice-bytes")
            bot._tts = fake_tts
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 721,
                    "message": {
                        "message_id": 921,
                        "chat": {"id": 131, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 131, "first_name": "Vadim"},
                        "text": "тук-тук",
                    },
                }
            )
            # Synthesize() got voice=None, so it should fall back to config.voice.
            self.assertIsNone(fake_tts.last_voice)

    def test_persona_json_loads_tts_voice_field(self) -> None:
        from protoagi.persona import get_persona

        self.assertEqual(get_persona("solomiya").tts_voice, "solomiya")
        self.assertEqual(get_persona("mykola").tts_voice, "mykola")

    def test_tts_delivery_aliases_are_normalized(self) -> None:
        self.assertEqual(TelegramConfig(token="token").tts_delivery, "auto")
        self.assertEqual(TelegramConfig(token="token", tts_delivery="auto").tts_delivery, "auto")
        self.assertEqual(TelegramConfig(token="token", tts_delivery="both").tts_delivery, "text_and_voice")
        self.assertEqual(TelegramConfig(token="token", tts_delivery="voice-only").tts_delivery, "voice")
        self.assertEqual(TelegramConfig(token="token", tts_delivery="bad-value").tts_delivery, "auto")

    def test_voice_synthesizer_rejects_json_error_blob(self) -> None:
        from protoagi.telegram.voice import VoiceSynthesisConfig, VoiceSynthesizer
        from urllib.request import Request as _Req

        synth = VoiceSynthesizer(
            VoiceSynthesisConfig(
                base_url="http://tts.local/v1",
                model="tts-stub",
                enabled=True,
                response_format="opus",
            )
        )

        captured_payloads: list[dict] = []

        def fake_urlopen(request, timeout=120):  # type: ignore[no-untyped-def]
            assert isinstance(request, _Req)
            captured_payloads.append(json.loads(request.data.decode("utf-8")))

            class _Resp:
                def read(self_inner):
                    return b'{"error": {"message": "voice not found", "code": 400}}'

                def __enter__(self_inner):
                    return self_inner

                def __exit__(self_inner, *exc):
                    return False

            return _Resp()

        import protoagi.telegram.voice as voice_mod

        original = voice_mod.urlopen
        voice_mod.urlopen = fake_urlopen  # type: ignore[assignment]
        try:
            self.assertIsNone(synth.synthesize("hello"))
        finally:
            voice_mod.urlopen = original  # type: ignore[assignment]
        self.assertIsNotNone(synth.last_error)
        self.assertIn("JSON error", synth.last_error or "")
        self.assertEqual(captured_payloads[0]["response_format"], "opus")
        self.assertEqual(captured_payloads[0]["model"], "tts-stub")

    def test_tts_mp3_routes_through_send_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "audio reply", "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    tts_enabled=True,
                    tts_base_url="http://tts.local/v1",
                    tts_model="tts-stub",
                    tts_response_format="mp3",
                    tts_delivery="voice",
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot._tts = FakeTTS(b"mp3-bytes", response_format="mp3")
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 555,
                    "message": {
                        "message_id": 916,
                        "chat": {"id": 124, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 124, "first_name": "Vadim"},
                        "text": "say it",
                    },
                }
            )
            self.assertEqual(telegram.voices, [])
            self.assertEqual(telegram.sent, [])
            self.assertTrue(getattr(telegram, "audios", []))
            audio = telegram.audios[0]
            self.assertEqual(audio["data"], b"mp3-bytes")
            self.assertEqual(audio["mime_type"], "audio/mpeg")
            self.assertTrue(audio["filename"].endswith(".mp3"))

    def test_photo_blind_reply_is_suppressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Пробач, але я не бачу зображення. Що на ньому?", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya", vision_model="vision-test"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.vision_llm = FakeVisionLLM("На фото біла чашка з червоним логотипом.")
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 233,
                    "message": {
                        "message_id": 913,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "photo": [{"file_id": "big", "width": 1200, "height": 900, "file_size": 2000}],
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent, [])

    def test_photo_without_vision_uses_neutral_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "О, фотка прилетіла.", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 231,
                    "message": {
                        "message_id": 911,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "photo": [{"file_id": "big", "width": 1200, "height": 900, "file_size": 2000}],
                    },
                }
            )
            self.assertTrue(processed)
            payload = llm.messages[0][1]["content"]
            self.assertIn("опис недоступний", payload)
            self.assertNotIn("vision model", payload)
            self.assertNotIn("не налаштована", payload)
            media_items = [item for item in memory.list_memories(limit=10) if item.media_id == "big"]
            self.assertEqual(media_items, [])

    def test_raw_gif_document_skips_vision_llm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "GIF accepted", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            vision = FakeVisionLLM("should not be called")
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya", vision_model="vision-test"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.vision_llm = vision
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 237,
                    "message": {
                        "message_id": 917,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "document": {
                            "file_id": "GIF_DOC_RAW",
                            "mime_type": "image/gif",
                            "file_name": "meme.gif",
                        },
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(vision.messages, [])
            payload = llm.messages[0][1]["content"]
            self.assertIn("опис недоступний", payload)
            self.assertIn("не вгадуй", payload)
            self.assertIsNotNone(memory.get_media_blob("GIF_DOC_RAW"))
            media_items = [item for item in memory.list_memories(limit=10) if item.media_id == "GIF_DOC_RAW"]
            self.assertEqual(media_items, [])

    def test_raw_gif_document_uses_extracted_still_frame_when_available(self) -> None:
        import protoagi.telegram.vision as vision_mod

        original = vision_mod._extract_gif_still_frame
        vision_mod._extract_gif_still_frame = lambda data: b"jpeg-frame"  # type: ignore[assignment]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                memory = MemoryStore(Path(tmp) / "memory.sqlite3")
                telegram = FakeTelegram()
                llm = FakeLLM(
                    '{"should_reply": true, "reply": "seen", '
                    '"stickers": [], "memories": [], "next_check_minutes": 60}'
                )
                vision = FakeVisionLLM("animated cat waves")
                bot = NikolaBot(
                    telegram=telegram,
                    llm=llm,
                    memory=memory,
                    telegram_config=TelegramConfig(token="token", persona_key="solomiya", vision_model="vision-test"),
                    agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
                )
                bot.vision_llm = vision
                bot.bootstrap()
                processed = bot.process_update(
                    {
                        "update_id": 239,
                        "message": {
                            "message_id": 919,
                            "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                            "from": {"id": 123, "first_name": "Vadim"},
                            "document": {
                                "file_id": "GIF_DOC_RAW",
                                "mime_type": "image/gif",
                                "file_name": "meme.gif",
                            },
                        },
                    }
                )
                self.assertTrue(processed)
                self.assertTrue(vision.messages)
                vision_content = vision.messages[0][1]["content"]
                self.assertIn("data:image/jpeg;base64", vision_content[1]["image_url"]["url"])
                payload = llm.messages[0][1]["content"]
                self.assertIn("animated cat waves", payload)
                media_items = [item for item in memory.list_memories(limit=10) if item.media_id == "GIF_DOC_RAW"]
                self.assertTrue(media_items)
                self.assertIn("animated cat waves", media_items[0].text)
        finally:
            vision_mod._extract_gif_still_frame = original  # type: ignore[assignment]

    def test_decision_prompt_compacts_large_history_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ok", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    prompt_context_max_chars=2500,
                    max_history_messages=20,
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            thread = bot.thread_id(123)
            for index in range(12):
                memory.log_message(thread, "user", f"old message {index} " + ("x" * 1200))
                memory.log_telegram_message(
                    chat_id=123,
                    message_id=index + 1,
                    persona_key=bot.persona.key,
                    role="user",
                    sender_id=123,
                    sender_name="Vadim",
                    text=f"telegram old {index} " + ("y" * 1200),
                    metadata={"message": {"huge": "z" * 15000}},
                )

            processed = bot.process_update(
                {
                    "update_id": 238,
                    "message": {
                        "message_id": 918,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "hello " + ("q" * 3000),
                    },
                }
            )
            self.assertTrue(processed)
            payload = llm.messages[0][1]["content"]
            self.assertLessEqual(len(payload), 2500)
            self.assertNotIn("metadata", payload)
            self.assertNotIn("self_model", payload)
            self.assertIn("context_truncated", payload)

    def test_incoming_sticker_is_not_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ахах, прийнято", '
                '"stickers": [{"pack": "SenkoSan", "emoji": "🙂", "reason": "mirror mood"}], '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 232,
                    "message": {
                        "message_id": 912,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "sticker": {
                            "file_id": "sticker-file",
                            "emoji": "🙂",
                            "set_name": "SenkoSan",
                            "is_animated": False,
                            "is_video": False,
                        },
                    },
                }
            )
            self.assertTrue(processed)
            payload = llm.messages[0][1]["content"]
            self.assertIn("[стікер: sticker, emoji=🙂", payload)
            self.assertEqual(telegram.sent[0]["text"], "ахах, прийнято")
            self.assertEqual(telegram.stickers[0]["sticker"], "SenkoSan:smile")

    def test_incoming_sticker_uses_visual_description_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "бачу сонний настрій", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            # End the caption with a period so it isn't flagged as
            # truncated by the cleaner — exercising the happy-path
            # passthrough rather than the new ellipsis annotation.
            vision = FakeVisionLLM("sleepy fox wrapped in a blanket.")
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya", vision_model="vision-test"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.vision_llm = vision
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 933,
                    "message": {
                        "message_id": 933,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "sticker": {
                            "file_id": "sticker-file",
                            "emoji": "🧃",
                            "set_name": "RandomPack",
                            "is_animated": False,
                            "is_video": False,
                            "thumbnail": {"file_id": "sticker-thumb"},
                        },
                    },
                }
            )
            self.assertTrue(processed)
            self.assertTrue(vision.messages)
            payload = llm.messages[0][1]["content"]
            self.assertIn("sleepy fox wrapped in a blanket", payload)
            self.assertIn("emoji=🧃", payload)
            recent = memory.recent_telegram_messages(123, limit=1)[0]
            self.assertEqual(recent["metadata"]["sticker"]["visual_description"], "sleepy fox wrapped in a blanket.")

    def test_typing_action_is_sent_while_processing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "я тут", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 934,
                    "message": {
                        "message_id": 934,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "ти тут?",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertTrue(any(item["action"] == "typing" for item in telegram.actions))

    def test_assistanty_phrases_are_removed_from_outgoing_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Я просто тут, готова допомогти, якщо треба. Як твої справи сьогодні?", '
                '"stickers": [], "memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 24,
                    "message": {
                        "message_id": 92,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "тобі пишу",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent, [])

    def test_human_prompt_rejects_assistanty_chat_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot = NikolaBot(
                telegram=FakeTelegram(),
                llm=FakeLLM("{}"),
                memory=MemoryStore(Path(tmp) / "memory.sqlite3"),
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            prompt = bot._decision_system_prompt()
            self.assertIn("не сервіс", prompt.lower())
            self.assertIn("готова допомогти", prompt)
            self.assertIn("не перезапускай розмову", prompt.lower())

    def test_solomiya_profile_changes_identity_and_uses_global_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.remember("Микола знає про ранкову каву", ["telegram", "telegram_chat_123", "nikola"])
            memory.remember("Соломія знає про вечірній чай", ["telegram", "telegram_chat_456", "solomiya"])
            memory.log_telegram_message(
                chat_id=123,
                message_id=70,
                persona_key="mykola",
                role="user",
                sender_id=123,
                sender_name="Vadim",
                text="Старе повідомлення з іншого профілю",
            )
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Я тут, і звучить цікаво.", '
                '"memories": ["Користувач любить живі розмови"], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 3,
                    "message": {
                        "message_id": 88,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "Соломіє, памʼятаєш каву і чай?",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(bot.telegram_config.bot_name, "Соломія")
            self.assertEqual(bot.thread_id("123"), "telegram:chat:123")
            payload = llm.messages[0][1]["content"]
            self.assertIn('"display_name": "Соломія"', payload)
            self.assertIn("вечірній чай", payload)
            self.assertIn("ранкову каву", payload)
            self.assertIn("Старе повідомлення з іншого профілю", payload)
            hits = memory.search_tagged_all("живі", [TELEGRAM_GLOBAL_MEMORY_TAG])
            self.assertEqual(len(hits), 1)

    def test_polluted_compact_history_is_cleaned_before_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "Добре.", '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            memory.log_message(bot.thread_id("123"), "assistant", "Соломія: Соломія: Привіт")
            processed = bot.process_update(
                {
                    "update_id": 33,
                    "message": {
                        "message_id": 91,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "ще раз",
                    },
                }
            )
            self.assertTrue(processed)
            payload = llm.messages[0][1]["content"]
            self.assertIn('"content": "Привіт"', payload)
            self.assertNotIn("Соломія: Соломія", payload)

    def test_solomiya_addressing_uses_her_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bot = NikolaBot(
                telegram=FakeTelegram(),
                llm=FakeLLM("{}"),
                memory=MemoryStore(Path(tmp) / "memory.sqlite3"),
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            self.assertTrue(bot._is_addressed("Соломіє, привіт", {}))
            self.assertFalse(bot._is_addressed("Миколо, привіт", {}))

    def test_group_chat_skips_llm_for_unaddressed_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM('{"should_reply": false, "reply": "", "memories": [], "next_check_minutes": 60}')
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 700,
                    "message": {
                        "message_id": 900,
                        "chat": {"id": -100123, "type": "supergroup", "title": "ChatRoom"},
                        "from": {"id": 7, "first_name": "Olesia"},
                        "text": "ну й погодка сьогодні",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(len(llm.messages), 0)
            self.assertEqual(telegram.sent, [])

    def test_group_chat_runs_llm_when_addressed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ага, погода так собі", '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 701,
                    "message": {
                        "message_id": 901,
                        "chat": {"id": -100123, "type": "supergroup", "title": "ChatRoom"},
                        "from": {"id": 7, "first_name": "Olesia"},
                        "text": "Миколо, як тобі погода?",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertGreaterEqual(len(llm.messages), 1)
            self.assertEqual(len(telegram.sent), 1)

    def test_reasoning_capture_records_when_enabled(self) -> None:
        from protoagi.telegram.config import TelegramConfig
        from protoagi.telegram.reasoning_log import REASONING_KV_PREFIX, ReasoningLogConfig

        class ReasoningLLM:
            def __init__(self, content: str, reasoning: str) -> None:
                self.content = content
                self.reasoning = reasoning
                self.messages: list = []
                self.kwargs: list = []

            def chat_completion(self, messages, **kwargs):
                self.messages.append(messages)
                self.kwargs.append(kwargs)
                return {
                    "choices": [
                        {
                            "message": {
                                "content": self.content,
                                "reasoning_content": self.reasoning,
                            }
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = ReasoningLLM(
                '{"should_reply": true, "reply": "ага", "memories": [], "next_check_minutes": 60}',
                "step 1: parse intent\nstep 2: short ack",
            )
            cfg = TelegramConfig(
                token="token",
                reasoning_log=ReasoningLogConfig(enabled=True, max_entries_per_chat=5),
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=cfg,
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 800,
                    "message": {
                        "message_id": 999,
                        "chat": {"id": 333, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 333, "first_name": "Vadim"},
                        "text": "привіт",
                    },
                }
            )
            raw = memory.get_kv(REASONING_KV_PREFIX + "333")
            self.assertIsNotNone(raw)
            entries = json.loads(raw)["entries"]
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["message_id"], 999)
            self.assertIn("step 1: parse intent", entries[0]["reasoning"])

    def test_reasoning_capture_skipped_when_disabled(self) -> None:
        from protoagi.telegram.config import TelegramConfig
        from protoagi.telegram.reasoning_log import REASONING_KV_PREFIX, ReasoningLogConfig

        class ReasoningLLM:
            def chat_completion(self, messages, **kwargs):
                return {
                    "choices": [
                        {
                            "message": {
                                "content": '{"should_reply": false, "reply": "", "memories": [], "next_check_minutes": 60}',
                                "reasoning_content": "internal monologue",
                            }
                        }
                    ]
                }

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            cfg = TelegramConfig(
                token="token",
                reasoning_log=ReasoningLogConfig(enabled=False),
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=ReasoningLLM(),
                memory=memory,
                telegram_config=cfg,
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 801,
                    "message": {
                        "message_id": 1000,
                        "chat": {"id": 444, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 444, "first_name": "Vadim"},
                        "text": "привіт",
                    },
                }
            )
            self.assertIsNone(memory.get_kv(REASONING_KV_PREFIX + "444"))

    def test_normalize_tool_request_accepts_web_search(self) -> None:
        from protoagi.telegram.json_io import normalize_tool_request

        for name in ("recall", "remind_me", "web_search"):
            self.assertEqual(
                normalize_tool_request({"name": name, "arguments": {"query": "x"}}),
                {"name": name, "arguments": {"query": "x"}},
            )
        self.assertIsNone(normalize_tool_request({"name": "rm -rf /", "arguments": {}}))

    def test_web_search_tool_request_invokes_runner(self) -> None:
        from protoagi.telegram.config import TelegramConfig
        from protoagi.web_search import WebSearchConfig

        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()

            llm = FakeLLM(
                [
                    '{"should_reply": true, "reply": "", '
                    '"tool_request": {"name": "web_search", "arguments": {"query": "погода"}}, '
                    '"memories": [], "next_check_minutes": 60}',
                    '{"should_reply": true, "reply": "Знайшла: weather.example", '
                    '"memories": [], "next_check_minutes": 60}',
                ]
            )
            cfg = TelegramConfig(
                token="token",
                web_search=WebSearchConfig(base_url="https://search.example/api"),
            )

            captured: list[str] = []

            payload_json = json.dumps(
                {
                    "results": [
                        {
                            "title": "Forecast",
                            "url": "https://weather.example",
                            "content": "сонячно",
                        }
                    ]
                }
            ).encode("utf-8")

            def fake_fetch(url: str, max_chars: int) -> tuple[str, bytes]:
                captured.append(url)
                return ("application/json", payload_json)

            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=cfg,
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot._web_search._fetcher = fake_fetch  # type: ignore[attr-defined]
            processed = bot.process_update(
                {
                    "update_id": 900,
                    "message": {
                        "message_id": 909,
                        "chat": {"id": 6001, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 6001, "first_name": "Vadim"},
                        "text": "яка зараз погода?",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(len(captured), 1)
            self.assertIn("q=", captured[0])
            self.assertIn("Знайшла", telegram.sent[0]["text"])

    def test_decision_context_includes_available_tools(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM('{"should_reply": false, "reply": "", "memories": [], "next_check_minutes": 60}')
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 702,
                    "message": {
                        "message_id": 902,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "що там нового",
                    },
                }
            )
            decision_context = json.loads(llm.messages[0][1]["content"])
            self.assertIn("available_tools", decision_context)
            self.assertIn("recall", decision_context["available_tools"])
            self.assertIn("remind_me", decision_context["available_tools"])
            self.assertNotIn("web_search", decision_context["available_tools"])

    def test_start_uses_active_persona_without_command_menu(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            bot = NikolaBot(
                telegram=telegram,
                llm=FakeLLM("{}"),
                memory=memory,
                telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            processed = bot.process_update(
                {
                    "update_id": 4,
                    "message": {
                        "message_id": 89,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "/start",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertIn("Соломія", telegram.sent[0]["text"])
            self.assertNotIn("/remember", telegram.sent[0]["text"])

    # ------------------------------------------------------------------
    # setMessageReaction integration

    def test_decision_from_payload_parses_reactions(self) -> None:
        decision = decision_from_payload(
            {
                "should_reply": True,
                "reply": "ок",
                "reactions": [
                    {"emoji": "👍", "message_id": 42, "big": True, "reason": "agreed"}
                ],
            }
        )
        self.assertEqual(len(decision.reactions), 1)
        self.assertEqual(decision.reactions[0]["emoji"], "👍")
        self.assertEqual(decision.reactions[0]["message_id"], 42)
        self.assertTrue(decision.reactions[0]["big"])
        self.assertEqual(decision.reactions[0]["reason"], "agreed")

    def test_decision_reactions_capped_at_one(self) -> None:
        decision = decision_from_payload(
            {
                "should_reply": False,
                "reactions": [
                    {"emoji": "👍"},
                    {"emoji": "🔥"},
                ],
            }
        )
        self.assertEqual(len(decision.reactions), 1)
        self.assertEqual(decision.reactions[0]["emoji"], "👍")

    def test_reaction_sent_on_incoming_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ага", '
                '"reactions": [{"emoji": "👍"}], '
                '"memories": [], "next_check_minutes": 60}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 55,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "+",
                    },
                }
            )
            self.assertEqual(len(telegram.reactions), 1)
            self.assertEqual(telegram.reactions[0]["emoji"], "👍")
            self.assertEqual(telegram.reactions[0]["message_id"], 55)

    def test_reaction_cooldown_blocks_second_within_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                [
                    '{"should_reply": false, '
                    '"reactions": [{"emoji": "🔥"}]}',
                    '{"should_reply": false, '
                    '"reactions": [{"emoji": "🔥"}]}',
                ]
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    reaction_cooldown_seconds=3600,
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            for i, msg_id in enumerate((71, 72), start=1):
                bot.process_update(
                    {
                        "update_id": i,
                        "message": {
                            "message_id": msg_id,
                            "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                            "from": {"id": 123, "first_name": "Vadim"},
                            "text": f"line {i}",
                        },
                    }
                )
            self.assertEqual(len(telegram.reactions), 1)
            self.assertEqual(telegram.reactions[0]["message_id"], 71)

    def test_reaction_disallowed_emoji_filtered(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": false, '
                '"reactions": [{"emoji": "🤖"}]}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 81,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "hi",
                    },
                }
            )
            self.assertEqual(telegram.reactions, [])
            self.assertEqual(telegram.sent, [])

    def test_reaction_only_path_without_reply(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": false, '
                '"reactions": [{"emoji": "❤️"}]}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 91,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "thank you",
                    },
                }
            )
            self.assertEqual(telegram.sent, [])
            self.assertEqual(len(telegram.reactions), 1)
            self.assertEqual(telegram.reactions[0]["emoji"], "❤️")

    def test_reaction_disabled_via_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            llm = FakeLLM(
                '{"should_reply": true, "reply": "ага", '
                '"reactions": [{"emoji": "👍"}]}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(
                    token="token",
                    reaction_enabled=False,
                ),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 101,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "hi",
                    },
                }
            )
            self.assertEqual(telegram.reactions, [])
            self.assertEqual(telegram.sent[0]["text"], "ага")

    def test_reaction_invalid_emoji_added_to_denylist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            telegram = FakeTelegram()
            telegram.next_reaction_error = "Telegram HTTP 400: REACTION_INVALID"
            llm = FakeLLM(
                '{"should_reply": false, '
                '"reactions": [{"emoji": "👍"}]}'
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="token"),
                agent_config=AgentConfig(database_path=Path(tmp) / "memory.sqlite3"),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 111,
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "hi",
                    },
                }
            )
            self.assertEqual(telegram.reactions, [])
            from protoagi.telegram.reactions import (
                REACTION_DENYLIST_KV_PREFIX,
                parse_denylist,
            )

            denylist = parse_denylist(memory.get_kv(REACTION_DENYLIST_KV_PREFIX + "123"))
            self.assertIn("👍", denylist)


if __name__ == "__main__":
    unittest.main()
