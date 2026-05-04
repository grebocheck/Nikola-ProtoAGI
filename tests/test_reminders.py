import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.telegram import NikolaBot, TelegramConfig


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.actions: list[dict] = []

    def get_me(self) -> dict:
        return {"id": 42, "username": "NikolaTestBot", "first_name": "Микола"}

    def delete_webhook(self, *, drop_pending_updates: bool = False) -> bool:
        return True

    def get_updates(self, *, offset, timeout_seconds, allowed_updates):
        return []

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
        return {"stickers": []}

    def get_file(self, file_id):
        return {"file_id": file_id, "file_path": f"photos/{file_id}.jpg"}

    def download_file(self, file_path, *, max_bytes):
        return b""

    def send_sticker(self, chat_id, sticker, *, reply_to_message_id=None, disable_notification=False):
        return {"message_id": 999}


class FakeLLM:
    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": "{}"}}]}


class ReminderDispatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        path = Path(self.tmp.name) / "memory.sqlite3"
        self.memory = MemoryStore(path)
        self.bot = NikolaBot(
            telegram=FakeTelegram(),
            llm=FakeLLM(),
            memory=self.memory,
            telegram_config=TelegramConfig(token="token"),
            agent_config=AgentConfig(database_path=path),
        )
        self.bot.bootstrap()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _register_chat(self) -> None:
        self.memory.upsert_telegram_chat(
            {"id": 123, "type": "private", "first_name": "Vadim"},
            {"id": 123, "first_name": "Vadim"},
        )

    def test_due_reminder_is_delivered(self) -> None:
        self._register_chat()
        past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat(timespec="seconds")
        self.memory.add_reminder(text="випити води", trigger_at=past, chat_id="123")
        delivered = self.bot.dispatch_due_reminders()
        self.assertEqual(delivered, 1)
        self.assertTrue(self.bot.telegram.sent[0]["text"].endswith("випити води"))
        due_after = self.memory.due_reminders(
            datetime.now(timezone.utc).isoformat(timespec="seconds")
        )
        self.assertEqual(due_after, [])

    def test_future_reminder_is_left_alone(self) -> None:
        self._register_chat()
        future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(timespec="seconds")
        self.memory.add_reminder(text="зустріч", trigger_at=future, chat_id="123")
        delivered = self.bot.dispatch_due_reminders()
        self.assertEqual(delivered, 0)
        self.assertEqual(self.bot.telegram.sent, [])

    def test_reminder_for_unknown_chat_is_cancelled(self) -> None:
        past = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(timespec="seconds")
        self.memory.add_reminder(text="нагадування", trigger_at=past, chat_id="999")
        delivered = self.bot.dispatch_due_reminders()
        self.assertEqual(delivered, 0)
        self.assertEqual(self.bot.telegram.sent, [])
        future_now = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat(timespec="seconds")
        self.assertEqual(self.memory.due_reminders(future_now), [])


if __name__ == "__main__":
    unittest.main()
