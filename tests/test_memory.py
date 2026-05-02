from pathlib import Path
import tempfile
import unittest

from protoagi.memory import MemoryStore


class MemoryStoreTests(unittest.TestCase):
    def test_remember_and_search(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            rowid = memory.remember("The preferred profile is ctx 8192 with CpuMoE 4.", ["runtime"])
            self.assertGreater(rowid, 0)
            hits = memory.search("CpuMoE profile", limit=3)
            self.assertTrue(any("CpuMoE" in hit.text for hit in hits))

    def test_message_history_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.log_message("t1", "user", "hello")
            memory.log_message("t1", "assistant", "hi")
            self.assertEqual(
                memory.recent_messages("t1"),
                [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi"}],
            )

    def test_search_tagged_requires_exact_tag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.remember("chat twelve likes tea", ["telegram_chat_12"])
            memory.remember("chat one-two-three likes coffee", ["telegram_chat_123"])
            hits = memory.search_tagged("likes", "telegram_chat_12", limit=5)
            self.assertEqual([hit.text for hit in hits], ["chat twelve likes tea"])

    def test_search_tagged_all_requires_persona_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.remember("likes calm morning notes", ["telegram_chat_123", "nikola"])
            memory.remember("likes playful morning notes", ["telegram_chat_123", "solomiya"])
            hits = memory.search_tagged_all("morning", ["telegram_chat_123", "solomiya"], limit=5)
            self.assertEqual([hit.text for hit in hits], ["likes playful morning notes"])

    def test_recent_telegram_messages_include_message_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.log_telegram_message(
                chat_id=123,
                message_id=10,
                role="user",
                sender_id=1,
                sender_name="Vadim",
                text="hello",
            )
            memory.log_telegram_message(
                chat_id=123,
                message_id=11,
                role="assistant",
                sender_id=None,
                sender_name="Nikola",
                text="hi",
            )
            self.assertEqual(
                [(item["message_id"], item["text"]) for item in memory.recent_telegram_messages(123)],
                [(10, "hello"), (11, "hi")],
            )

    def test_recent_telegram_messages_can_filter_persona(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.log_telegram_message(
                chat_id=123,
                message_id=10,
                persona_key="mykola",
                role="assistant",
                sender_id=None,
                sender_name="Микола",
                text="calm",
            )
            memory.log_telegram_message(
                chat_id=123,
                message_id=11,
                persona_key="solomiya",
                role="assistant",
                sender_id=None,
                sender_name="Соломія",
                text="warm",
            )
            hits = memory.recent_telegram_messages(123, persona_key="solomiya")
            self.assertEqual([(item["persona_key"], item["text"]) for item in hits], [("solomiya", "warm")])


if __name__ == "__main__":
    unittest.main()
