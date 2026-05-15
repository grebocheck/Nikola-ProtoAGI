"""Provenance tests: memory items remember the message they came from.

Each Telegram fact written by the bot now carries an
``origin_message_id`` like ``telegram:<chat_id>:<message_id>`` so the
model can correlate a recalled memory with a row in
``recent_telegram_messages``. This file tests both the storage primitive
and the orchestrator wiring.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.telegram import NikolaBot, TelegramConfig
from protoagi.telegram.orchestrator import _format_origin_ref


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def get_me(self):
        return {"id": 1, "username": "ProvenanceBot"}

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
        self.calls: list[list[dict]] = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append(list(messages))
        return {"choices": [{"message": {"content": self.content}}]}


def _build_bot(memory: MemoryStore, llm: FakeLLM, db_path: Path) -> NikolaBot:
    bot = NikolaBot(
        telegram=FakeTelegram(),
        llm=llm,
        memory=memory,
        telegram_config=TelegramConfig(token="t", persona_key="solomiya"),
        agent_config=AgentConfig(database_path=db_path),
    )
    bot.bootstrap()
    return bot


class FormatOriginRefTests(unittest.TestCase):
    def test_builds_canonical_string(self) -> None:
        self.assertEqual(_format_origin_ref("555", 42), "telegram:555:42")

    def test_none_message_id_returns_none(self) -> None:
        self.assertIsNone(_format_origin_ref("555", None))

    def test_zero_message_id_returns_none(self) -> None:
        # Telegram message_ids start at 1; 0 is the bot's "no message yet"
        # default and should not pollute the column.
        self.assertIsNone(_format_origin_ref("555", 0))

    def test_invalid_input_returns_none(self) -> None:
        self.assertIsNone(_format_origin_ref("555", "not-an-int"))
        self.assertIsNone(_format_origin_ref("555", -5))


class StorePassesOriginTests(unittest.TestCase):
    def test_store_memory_persists_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite3")
            mid = store.store_memory(
                "людина любить ромашковий чай",
                origin_message_id="telegram:555:10",
            )
            item = store.get_memory(mid)
            assert item is not None
            self.assertEqual(item.origin_message_id, "telegram:555:10")

    def test_default_origin_is_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite3")
            mid = store.store_memory("legacy fact without provenance")
            item = store.get_memory(mid)
            assert item is not None
            self.assertIsNone(item.origin_message_id)

    def test_empty_string_origin_is_normalized_to_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = MemoryStore(Path(tmp) / "memory.sqlite3")
            mid = store.store_memory("text", origin_message_id="   ")
            item = store.get_memory(mid)
            assert item is not None
            self.assertIsNone(item.origin_message_id)


class OrchestratorRecordsOriginTests(unittest.TestCase):
    def test_decision_memory_carries_telegram_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            llm = FakeLLM(
                json.dumps(
                    {
                        "should_reply": True,
                        "reply": "запам'ятала",
                        "memories": ["людина любить ромашковий чай"],
                    }
                )
            )
            bot = _build_bot(memory, llm, path)
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 42,
                        "chat": {"id": 555, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 555, "first_name": "Vadim"},
                        "text": "я люблю ромашковий чай",
                    },
                }
            )
            items = memory.list_memories(limit=10)
            relevant = [item for item in items if "ромашковий" in item.text]
            self.assertEqual(len(relevant), 1)
            self.assertEqual(relevant[0].origin_message_id, "telegram:555:42")

    def test_relevant_memory_payload_includes_origin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            # Seed a memory with origin so the next turn surfaces it.
            memory.store_memory(
                "людина любить ромашковий чай",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
                origin_message_id="telegram:555:10",
            )
            llm = FakeLLM(json.dumps({"should_reply": True, "reply": "ок"}))
            bot = _build_bot(memory, llm, path)
            bot.process_update(
                {
                    "update_id": 1,
                    "message": {
                        "message_id": 11,
                        "chat": {"id": 555, "type": "private", "first_name": "Vadim"},
                        "from": {"id": 555, "first_name": "Vadim"},
                        "text": "поговоримо про чай",
                    },
                }
            )
            self.assertTrue(llm.calls)
            user_payload = json.loads(llm.calls[0][1]["content"])
            relevant = user_payload.get("relevant_memory", [])
            self.assertTrue(relevant, "expected at least one recalled memory")
            chai = next(
                (item for item in relevant if "ромашковий" in item.get("text", "")),
                None,
            )
            self.assertIsNotNone(chai)
            self.assertEqual(chai["origin"], "telegram:555:10")


if __name__ == "__main__":
    unittest.main()
