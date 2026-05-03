import json
import tempfile
import unittest
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.memory import MemoryStore
from protoagi.telegram_bot import (
    NikolaBot,
    TelegramConfig,
    normalize_reminder_requests,
)


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def get_me(self):
        return {"id": 1, "username": "ReminderBot"}

    def delete_webhook(self, *, drop_pending_updates=False):
        return True

    def get_updates(self, **kwargs):
        return []

    def send_chat_action(self, *args, **kwargs):
        return True

    def send_message(self, chat_id, text, *, reply_to_message_id=None, disable_notification=False):
        self.sent.append({"chat_id": str(chat_id), "text": text})
        return {"message_id": len(self.sent)}

    def send_sticker(self, *args, **kwargs):
        return {"message_id": 99}

    def get_sticker_set(self, name):
        return {"stickers": []}

    def get_file(self, file_id):
        return {"file_path": ""}

    def download_file(self, file_path, *, max_bytes):
        return b""


class FakeLLM:
    def __init__(self, content: str) -> None:
        self.content = content

    def chat_completion(self, messages, **kwargs):
        return {"choices": [{"message": {"content": self.content}}]}


class ReminderRequestParseTests(unittest.TestCase):
    def test_normalizes_in_minutes_and_trigger_at(self) -> None:
        result = normalize_reminder_requests(
            [
                {"text": "polити квіти", "in_minutes": 30},
                {"text": "зідзвон", "trigger_at": "2030-01-01T10:00:00"},
                {"text": "  "},  # dropped
                {"text": "fallback default"},  # default 60min
            ]
        )
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0]["in_minutes"], 30)
        self.assertEqual(result[1]["trigger_at"], "2030-01-01T10:00:00")
        self.assertEqual(result[2]["in_minutes"], 60)

    def test_drops_non_dict_entries(self) -> None:
        result = normalize_reminder_requests(["bad", 42, None])
        self.assertEqual(result, [])


class DecisionReminderEndToEndTests(unittest.TestCase):
    def test_decision_reminder_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            telegram = FakeTelegram()
            llm = FakeLLM(
                json.dumps(
                    {
                        "should_reply": True,
                        "reply": "добре, нагадаю",
                        "reminders": [
                            {"text": "випити воду", "in_minutes": 45}
                        ],
                        "memories": [],
                        "next_check_minutes": 120,
                    }
                )
            )
            bot = NikolaBot(
                telegram=telegram,
                llm=llm,
                memory=memory,
                telegram_config=TelegramConfig(token="t", persona_key="solomiya"),
                agent_config=AgentConfig(database_path=path),
            )
            bot.bootstrap()
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 10,
                        "chat": {"id": 555, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 555, "first_name": "Vadim"},
                        "text": "нагадай мені випити воду",
                    },
                }
            )
            future = "9999-12-31T23:59:59+00:00"
            reminders = memory.due_reminders(future)
            self.assertEqual(len(reminders), 1)
            self.assertEqual(reminders[0].text, "випити воду")
            self.assertEqual(reminders[0].chat_id, "555")
            self.assertEqual(reminders[0].persona_key, "solomiya")


if __name__ == "__main__":
    unittest.main()
