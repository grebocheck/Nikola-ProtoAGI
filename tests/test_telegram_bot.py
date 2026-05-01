from pathlib import Path
import tempfile
import unittest

from protoagi.config import AgentConfig
from protoagi.memory import MemoryStore
from protoagi.telegram_bot import (
    NikolaBot,
    TelegramConfig,
    decision_from_payload,
    extract_json_object,
    parse_command,
    split_telegram_message,
)


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []
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


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self.content}}]}


class TelegramBotTests(unittest.TestCase):
    def test_parse_command_with_bot_suffix(self) -> None:
        self.assertEqual(parse_command("/remember@NikolaTestBot hello", "NikolaTestBot"), ("remember", "hello"))
        self.assertEqual(parse_command("/remember@OtherBot hello", "NikolaTestBot"), (None, ""))

    def test_extract_json_object_from_fenced_text(self) -> None:
        self.assertEqual(extract_json_object('```json\n{"should_reply": true}\n```'), {"should_reply": True})

    def test_decision_from_payload(self) -> None:
        decision = decision_from_payload({"should_reply": True, "reply": "Так", "memories": ["любить чай"]})
        self.assertTrue(decision.should_reply)
        self.assertEqual(decision.memories, ["любить чай"])

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
            hits = memory.search_tagged("спокійні", "telegram_chat_123")
            self.assertEqual(len(hits), 1)
            chat = memory.get_telegram_chat("123")
            self.assertIsNotNone(chat)
            self.assertIsNotNone(chat.next_initiative_at)


if __name__ == "__main__":
    unittest.main()
