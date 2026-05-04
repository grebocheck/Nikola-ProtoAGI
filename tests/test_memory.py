from pathlib import Path
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from protoagi.storage.memory import SCOPE_GLOBAL, SCOPE_USER, MemoryStore


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

    def test_media_blob_can_link_to_memory_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            blob = memory.store_media_blob(
                file_id="photo-1",
                mime="image/jpeg",
                data=b"fake-jpeg",
                caption="white mug on a desk",
            )
            rowid = memory.store_memory(
                "image showed a white mug on a desk",
                tags=["media", "image"],
                media_id=blob.file_id,
            )
            item = memory.get_memory(rowid)
            assert item is not None
            self.assertEqual(item.media_id, "photo-1")
            restored = memory.get_media_blob("photo-1")
            assert restored is not None
            self.assertEqual(restored.caption, "white mug on a desk")
            self.assertEqual(restored.bytes, b"fake-jpeg")

    def test_prune_orphan_media_removes_only_old_unlinked_blobs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.store_media_blob(
                file_id="old-orphan",
                mime="image/jpeg",
                data=b"old",
            )
            linked = memory.store_media_blob(
                file_id="old-linked",
                mime="image/jpeg",
                data=b"linked",
            )
            memory.store_memory("linked image", media_id=linked.file_id)
            old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(timespec="seconds")
            with memory.connect() as conn:
                conn.execute(
                    "UPDATE media_blobs SET created_at = ? WHERE file_id IN (?, ?)",
                    (old, "old-orphan", "old-linked"),
                )
            deleted = memory.prune_orphan_media(older_than_days=60)
            self.assertEqual(deleted, 1)
            self.assertIsNone(memory.get_media_blob("old-orphan"))
            self.assertIsNotNone(memory.get_media_blob("old-linked"))

    def test_rescope_telegram_memories_moves_global_user_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            rowid = memory.store_memory(
                "legacy privacy fact",
                scope=SCOPE_GLOBAL,
                tags=["telegram", "telegram_global", "user:telegram:42", "source_chat:123"],
            )
            result = memory.rescope_telegram_memories(to_scope=SCOPE_USER)
            self.assertEqual(result["updated"], 1)
            item = memory.get_memory(rowid)
            assert item is not None
            self.assertEqual(item.scope, SCOPE_USER)
            self.assertEqual(item.user_id, "telegram:42")
            self.assertEqual(item.chat_id, "123")

    def test_importance_cache_prune_removes_old_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            memory.set_importance_cache(
                "old",
                importance=0.8,
                kind="semantic",
                reasoning="old row",
            )
            memory.set_importance_cache(
                "fresh",
                importance=0.4,
                kind="fact",
                reasoning="fresh row",
            )
            old = (datetime.now(timezone.utc) - timedelta(days=90)).isoformat(timespec="seconds")
            with memory.connect() as conn:
                conn.execute(
                    "UPDATE importance_cache SET created_at = ?, last_accessed_at = ? WHERE key = ?",
                    (old, old, "old"),
                )
            result = memory.prune_importance_cache(older_than_days=60)
            self.assertEqual(result["deleted"], 1)
            self.assertIsNone(memory.get_importance_cache("old"))
            self.assertIsNotNone(memory.get_importance_cache("fresh"))


if __name__ == "__main__":
    unittest.main()
