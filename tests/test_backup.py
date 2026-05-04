import tempfile
import unittest
from pathlib import Path

from protoagi.storage.backup import backup_database, restore_database
from protoagi.storage.memory import MemoryStore


class BackupRestoreTests(unittest.TestCase):
    def test_backup_restore_round_trip_preserves_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"
            backup_path = Path(tmp) / "backups" / "snapshot.sqlite3"
            memory = MemoryStore(db_path)
            memory_id = memory.store_memory(
                "backup fact with vector",
                tags=["backup"],
                embedding=[0.25, 0.5, 0.75],
                embedding_model="test-embed",
            )

            written = backup_database(db_path, backup_path)
            self.assertEqual(written, backup_path.resolve())

            memory.delete_memory(memory_id)
            self.assertIsNone(memory.get_memory(memory_id))

            restored = restore_database(db_path, backup_path)
            self.assertEqual(restored, db_path.resolve())
            restored_memory = MemoryStore(db_path)
            item = restored_memory.get_memory(memory_id)
            self.assertIsNotNone(item)
            assert item is not None
            self.assertEqual(item.text, "backup fact with vector")
            embeddings = dict(restored_memory.all_embeddings())
            self.assertIn(memory_id, embeddings)
            self.assertEqual(len(embeddings[memory_id]), 3)


if __name__ == "__main__":
    unittest.main()
