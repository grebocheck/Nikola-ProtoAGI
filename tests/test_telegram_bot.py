import json
from pathlib import Path
import tempfile
import unittest

from protoagi.config import AgentConfig
from protoagi.memory import MemoryStore
from protoagi.telegram_bot import (
    NikolaBot,
    TELEGRAM_GLOBAL_MEMORY_TAG,
    TELEGRAM_PERSONA_SELF_MEMORY_TAG,
    TelegramConfig,
    auto_sticker_choice,
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
        self.actions: list[dict] = []
        self.updates: list[dict] = []

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


class FakeLLM:
    def __init__(self, content: str | list[str]) -> None:
        self.content = content[-1] if isinstance(content, list) and content else content
        self._queue = list(content) if isinstance(content, list) else None
        self.messages = []

    def chat_completion(self, messages, **kwargs):
        self.messages.append(messages)
        content = self._queue.pop(0) if self._queue is not None and self._queue else self.content
        return {"choices": [{"message": {"content": content}}]}


class FakeVisionLLM(FakeLLM):
    def server_props(self):
        return {"media_marker": "<dynamic-media-marker>"}


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
            self.assertEqual(telegram.sent[0]["text"], "I remember the nut allergy.")
            self.assertEqual(len(llm.messages), 2)
            merge_payload = json.loads(llm.messages[1][1]["content"])
            self.assertEqual(merge_payload["tool_results"][0]["name"], "recall")
            self.assertIn("nut allergy", json.dumps(merge_payload["tool_results"], ensure_ascii=False))

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
            self.assertIn("[стікер: sticker, 🙂", payload)
            self.assertEqual(telegram.sent[0]["text"], "ахах, прийнято")
            self.assertEqual(telegram.stickers[0]["sticker"], "SenkoSan:smile")

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


if __name__ == "__main__":
    unittest.main()
