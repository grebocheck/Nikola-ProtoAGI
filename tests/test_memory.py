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


if __name__ == "__main__":
    unittest.main()
