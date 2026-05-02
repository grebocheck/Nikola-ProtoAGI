from pathlib import Path
import tempfile
import unittest

from protoagi.config import AgentConfig
from protoagi.memory import MemoryStore
from protoagi.telegram_bot import (
    NikolaBot,
    TELEGRAM_GLOBAL_MEMORY_TAG,
    TELEGRAM_GLOBAL_THREAD_ID,
    TelegramConfig,
    decision_from_payload,
    extract_json_object,
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
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def chat_completion(self, messages, **kwargs):
        self.messages.append(messages)
        return {"choices": [{"message": {"content": self.content}}]}


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
                "reply_to": "current",
                "stickers": [{"pack": "miku", "emoji": "✨", "reason": "playful"}],
                "memories": ["любить чай"],
            }
        )
        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.reply_to, "current")
        self.assertEqual(decision.stickers[0]["pack"], "M1ku_Hatsune")
        self.assertEqual(decision.memories, ["любить чай"])

    def test_normalize_sticker_pack_alias(self) -> None:
        self.assertEqual(normalize_sticker_pack("senko"), "SenkoSan")

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

    def test_process_update_can_reply_to_current_and_send_sticker(self) -> None:
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
                        "chat": {"id": 123, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 123, "first_name": "Vadim"},
                        "text": "Миколо, глянь сюди",
                    },
                }
            )
            self.assertTrue(processed)
            self.assertEqual(telegram.sent[0]["reply_to_message_id"], 77)
            self.assertEqual(telegram.stickers[0]["sticker"], "SenkoSan:smile")

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
            self.assertEqual(bot.thread_id("123"), TELEGRAM_GLOBAL_THREAD_ID)
            payload = llm.messages[0][1]["content"]
            self.assertIn('"display_name": "Соломія"', payload)
            self.assertIn("вечірній чай", payload)
            self.assertIn("ранкову каву", payload)
            self.assertIn("Старе повідомлення з іншого профілю", payload)
            hits = memory.search_tagged_all("живі", [TELEGRAM_GLOBAL_MEMORY_TAG])
            self.assertEqual(len(hits), 1)

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
