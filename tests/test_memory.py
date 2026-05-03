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

    def test_recent_tagged_all_returns_latest_exact_tag_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.remember("old self fact", ["telegram_persona_self", "persona:solomiya"])
            memory.remember("wrong persona fact", ["telegram_persona_self", "persona:mykola"])
            memory.remember("new self fact", ["telegram_persona_self", "persona:solomiya"])
            hits = memory.recent_tagged_all(["telegram_persona_self", "persona:solomiya"], limit=3)
            self.assertEqual([hit.text for hit in hits], ["new self fact", "old self fact"])

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

    def test_update_memory_replaces_text_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            rowid = memory.remember("first version", ["tag-a"])
            updated = memory.update_memory(rowid, text="second version", tags=["tag-b"])
            assert updated is not None
            self.assertEqual(updated.text, "second version")
            self.assertEqual(updated.tags, ["tag-b"])
            hits = memory.search("second", limit=3)
            self.assertEqual([hit.id for hit in hits], [rowid])
            stale = memory.search("first", limit=3)
            self.assertEqual(stale, [])

    def test_set_pinned_toggles_score_eligibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            rowid = memory.remember("anchor fact", ["pinme"])
            memory.set_pinned(rowid, True)
            item = memory.get_memory(rowid)
            assert item is not None
            self.assertTrue(item.pinned)
            memory.set_pinned(rowid, False)
            item = memory.get_memory(rowid)
            assert item is not None
            self.assertFalse(item.pinned)

    def test_update_memory_rejects_empty_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            rowid = memory.remember("non-empty", [])
            with self.assertRaises(ValueError):
                memory.update_memory(rowid, text="   ")

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
