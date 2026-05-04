import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from protoagi.storage.memory import KIND_PERSONA_SELF, SCOPE_GLOBAL, SCOPE_PERSONA, MemoryStore
from protoagi.storage.service import MemoryService


def _shift_created(memory: MemoryStore, memory_id: int, days_ago: float) -> None:
    new_ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat(timespec="seconds")
    with memory.connect() as conn:
        conn.execute(
            "UPDATE memory_items SET created_at = ? WHERE id = ?",
            (new_ts, memory_id),
        )


class PruneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "memory.sqlite3"
        self.memory = MemoryStore(self.path)
        self.service = MemoryService(self.memory)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_recent_items_are_protected(self) -> None:
        stored = self.service.remember(
            "throwaway note", scope=SCOPE_GLOBAL, importance=0.05
        )
        assert stored is not None
        result = self.service.prune(score_threshold=0.5)
        self.assertEqual(result["deleted"], 0)
        self.assertEqual(result["skipped_recent"], 1)
        self.assertIsNotNone(self.memory.get_memory(stored.memory_id))

    def test_low_score_old_items_are_deleted(self) -> None:
        stored = self.service.remember(
            "noise", scope=SCOPE_GLOBAL, importance=0.05
        )
        assert stored is not None
        _shift_created(self.memory, stored.memory_id, days_ago=120)
        result = self.service.prune(
            keep_newer_than_days=30,
            score_threshold=0.5,
        )
        self.assertEqual(result["deleted"], 1)
        self.assertIsNone(self.memory.get_memory(stored.memory_id))

    def test_persona_self_is_protected(self) -> None:
        stored = self.service.remember(
            "Соломія про себе: інтровертка",
            kind=KIND_PERSONA_SELF,
            scope=SCOPE_PERSONA,
            persona_key="solomiya",
            importance=0.05,
        )
        assert stored is not None
        _shift_created(self.memory, stored.memory_id, days_ago=120)
        result = self.service.prune(
            keep_newer_than_days=30,
            score_threshold=0.99,
        )
        self.assertEqual(result["deleted"], 0)
        self.assertEqual(result["skipped_protected"], 1)

    def test_pinned_item_is_kept(self) -> None:
        stored = self.service.remember(
            "important pinned",
            scope=SCOPE_GLOBAL,
            importance=0.05,
            pinned=True,
        )
        assert stored is not None
        _shift_created(self.memory, stored.memory_id, days_ago=120)
        result = self.service.prune(keep_newer_than_days=30, score_threshold=0.99)
        self.assertEqual(result["deleted"], 0)
        self.assertEqual(result["skipped_pinned"], 1)

    def test_dry_run_does_not_delete(self) -> None:
        stored = self.service.remember(
            "old chatter", scope=SCOPE_GLOBAL, importance=0.05
        )
        assert stored is not None
        _shift_created(self.memory, stored.memory_id, days_ago=120)
        result = self.service.prune(
            keep_newer_than_days=30,
            score_threshold=0.5,
            dry_run=True,
        )
        self.assertEqual(result["deleted"], 1)
        self.assertIsNotNone(self.memory.get_memory(stored.memory_id))
        self.assertEqual(result["plan"][0]["dropped"]["id"], stored.memory_id)
        self.assertEqual(result["plan"][0]["reason"], "score_below_threshold")


if __name__ == "__main__":
    unittest.main()
