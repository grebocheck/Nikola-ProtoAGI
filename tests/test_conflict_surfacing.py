"""Tests that conflict pairs surface in the decision prompt context.

When a recalled memory has an unresolved conflict partner, the
``relevant_memory`` array in the user-payload should include a
``tensions`` field with snippets of the partner. This lets the persona
hedge instead of asserting old facts as certain.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from protoagi.config import AgentConfig
from protoagi.storage.memory import MemoryStore
from protoagi.telegram import NikolaBot, TelegramConfig


class FakeTelegram:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    def get_me(self):
        return {"id": 1, "username": "TensionBot"}

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


def _send(bot: NikolaBot, chat_id: int, message_id: int, text: str) -> None:
    bot.process_update(
        {
            "update_id": message_id,
            "message": {
                "message_id": message_id,
                "chat": {"id": chat_id, "type": "private", "first_name": "Vadim"},
                "from": {"id": chat_id, "first_name": "Vadim"},
                "text": text,
            },
        }
    )


class ConflictSurfacingTests(unittest.TestCase):
    def test_relevant_memory_includes_tensions_when_partner_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            # Seed two facts about the same topic.
            id_a = memory.store_memory(
                "людина любить ромашковий чай",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
                origin_message_id="telegram:555:10",
            )
            id_b = memory.store_memory(
                "людина перейшла на каву зранку",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
                origin_message_id="telegram:555:25",
            )
            # Record an unresolved conflict between them.
            memory.record_conflict(id_a, id_b, similarity=0.84, persona_key="solomiya")

            llm = FakeLLM(json.dumps({"should_reply": True, "reply": "ок"}))
            bot = _build_bot(memory, llm, path)
            # Query has to overlap with at least one fact for FTS recall to
            # surface it — embeddings are off in this fake setup.
            _send(bot, 555, 30, "розкажи мені про чай і каву")

            self.assertTrue(llm.calls)
            user_payload = json.loads(llm.calls[0][1]["content"])
            relevant = user_payload.get("relevant_memory", [])
            # Find the entry that has tensions.
            with_tensions = [item for item in relevant if "tensions" in item]
            self.assertTrue(with_tensions, f"no tensions found in {relevant}")
            entry = with_tensions[0]
            self.assertIn("text", entry)
            self.assertIn("origin", entry)
            tension = entry["tensions"][0]
            self.assertIn("with_text", tension)
            self.assertIn("similarity", tension)
            self.assertGreaterEqual(tension["similarity"], 0.8)

    def test_non_conflicted_facts_have_no_tensions_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            memory.store_memory(
                "людина любить ромашковий чай",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
            )
            llm = FakeLLM(json.dumps({"should_reply": True, "reply": "ок"}))
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 30, "поговоримо про чай")
            user_payload = json.loads(llm.calls[0][1]["content"])
            relevant = user_payload.get("relevant_memory", [])
            for item in relevant:
                self.assertNotIn("tensions", item)

    def test_resolved_conflict_does_not_surface(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            id_a = memory.store_memory(
                "людина любить ромашковий чай",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
            )
            id_b = memory.store_memory(
                "людина перейшла на каву",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
            )
            cid = memory.record_conflict(id_a, id_b, similarity=0.85, persona_key="solomiya")
            assert cid is not None
            memory.resolve_conflict(cid, status="dismissed")

            llm = FakeLLM(json.dumps({"should_reply": True, "reply": "ок"}))
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 30, "напої")
            user_payload = json.loads(llm.calls[0][1]["content"])
            relevant = user_payload.get("relevant_memory", [])
            for item in relevant:
                self.assertNotIn("tensions", item)

    def test_superseded_partner_does_not_surface(self) -> None:
        # If one side of the pair was superseded, it should not appear as a
        # live tension (it's no longer an active belief).
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(path)
            id_a = memory.store_memory(
                "людина любить ромашковий чай",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
            )
            id_b = memory.store_memory(
                "людина перейшла на каву",
                tags=["telegram", "telegram_global", "source_chat:555"],
                chat_id="555",
                persona_key="solomiya",
            )
            memory.record_conflict(id_a, id_b, similarity=0.85, persona_key="solomiya")
            memory.supersede(id_b, id_a)
            llm = FakeLLM(json.dumps({"should_reply": True, "reply": "ок"}))
            bot = _build_bot(memory, llm, path)
            _send(bot, 555, 30, "напої")
            user_payload = json.loads(llm.calls[0][1]["content"])
            relevant = user_payload.get("relevant_memory", [])
            # The remaining active fact may surface, but its partner (b) was
            # superseded so no tension should be reported.
            for item in relevant:
                self.assertNotIn("tensions", item)


if __name__ == "__main__":
    unittest.main()
