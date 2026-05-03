import json
import tempfile
import unittest
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.memory import KIND_PERSONA_SELF, SCOPE_PERSONA, MemoryStore
from protoagi.telegram_bot import (
    NikolaBot,
    TELEGRAM_GLOBAL_MEMORY_TAG,
    TELEGRAM_PERSONA_SELF_MEMORY_TAG,
    TelegramConfig,
)


class ScriptedLLM:
    def __init__(self, payloads: list[str]) -> None:
        self.payloads = list(payloads)
        self.calls: list[list[dict]] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append(messages)
        if not self.payloads:
            return {"choices": [{"message": {"content": "{}"}}]}
        content = self.payloads.pop(0)
        return {"choices": [{"message": {"content": content}}]}


class FakeTelegram:
    def get_me(self):
        return {"id": 1, "username": "ReflectionBot"}

    def delete_webhook(self, *, drop_pending_updates=False):
        return True

    def get_updates(self, **kwargs):
        return []

    def send_chat_action(self, *args, **kwargs):
        return True

    def send_message(self, *args, **kwargs):
        return {"message_id": 1}

    def send_sticker(self, *args, **kwargs):
        return {"message_id": 2}

    def get_sticker_set(self, name):
        return {"stickers": []}

    def get_file(self, file_id):
        return {"file_path": ""}

    def download_file(self, file_path, *, max_bytes):
        return b""


class ReflectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "memory.sqlite3"
        self.memory = MemoryStore(self.path)
        self.memory.store_memory(
            "Соломія про себе: любить чай зранку",
            kind=KIND_PERSONA_SELF,
            scope=SCOPE_PERSONA,
            persona_key="solomiya",
            tags=["telegram", TELEGRAM_PERSONA_SELF_MEMORY_TAG, "persona:solomiya"],
        )
        self.memory.store_memory(
            "Користувач любить філософські розмови вечорами",
            tags=["telegram", TELEGRAM_GLOBAL_MEMORY_TAG, "persona:solomiya"],
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _make_bot(self, llm: ScriptedLLM) -> NikolaBot:
        return NikolaBot(
            telegram=FakeTelegram(),
            llm=llm,
            memory=self.memory,
            telegram_config=TelegramConfig(token="token", persona_key="solomiya"),
            agent_config=AgentConfig(database_path=self.path),
        )

    def test_consolidate_dedupes_persona_self(self) -> None:
        # Add a near-duplicate of the existing self memory.
        self.memory.store_memory(
            "Соломія про себе: любить чай зранку",
            kind=KIND_PERSONA_SELF,
            scope=SCOPE_PERSONA,
            persona_key="solomiya",
            tags=[TELEGRAM_PERSONA_SELF_MEMORY_TAG, "persona:solomiya"],
            importance=0.6,
        )
        bot = self._make_bot(
            ScriptedLLM([json.dumps({"reflections": []})])
        )
        result = bot.run_reflection_pass()
        self.assertGreaterEqual(result["consolidated_persona"], 1)

    def test_reflection_writes_self_memory_when_payload_present(self) -> None:
        bot = self._make_bot(
            ScriptedLLM(
                [
                    json.dumps(
                        {
                            "reflections": [
                                "помічаю, що мені приємніше відповідати на тихі вечірні запитання",
                            ]
                        }
                    )
                ]
            )
        )
        result = bot.run_reflection_pass()
        self.assertEqual(result["reflections_written"], 1)
        recent = self.memory.list_memories(
            scope=SCOPE_PERSONA,
            persona_key="solomiya",
            kind=KIND_PERSONA_SELF,
            limit=10,
        )
        texts = [item.text for item in recent]
        self.assertTrue(any("вечірні" in text for text in texts))

    def test_reflection_prunes_old_low_value_global_facts(self) -> None:
        from datetime import datetime, timedelta, timezone

        stored = self.memory.store_memory(
            "old throwaway global note",
            tags=["telegram", "telegram_global"],
            importance=0.05,
        )
        # 270 days old means recency ≈ exp(-3) ≈ 0.05; combined score is
        # 0.5*0.05 + 0.3*0.05 + 0 ≈ 0.04, well below the 0.10 threshold the
        # reflection pass uses.
        old = (datetime.now(timezone.utc) - timedelta(days=270)).isoformat(timespec="seconds")
        with self.memory.connect() as conn:
            conn.execute(
                "UPDATE memory_items SET created_at = ? WHERE id = ?", (old, stored)
            )
        bot = self._make_bot(ScriptedLLM([json.dumps({"reflections": []})]))
        result = bot.run_reflection_pass()
        self.assertGreaterEqual(result["pruned_global"], 1)
        self.assertIsNone(self.memory.get_memory(stored))

    def test_reflection_skips_when_disabled(self) -> None:
        bot = NikolaBot(
            telegram=FakeTelegram(),
            llm=ScriptedLLM([]),
            memory=self.memory,
            telegram_config=TelegramConfig(
                token="token", persona_key="solomiya", fictional_self_enabled=False
            ),
            agent_config=AgentConfig(database_path=self.path),
        )
        result = bot.run_reflection_pass()
        self.assertEqual(result["reflections_written"], 0)


if __name__ == "__main__":
    unittest.main()
