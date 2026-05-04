import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from protoagi.cli import main
from protoagi.storage.federation import (
    MemoryFederationError,
    export_memory_bundle,
    import_memory_bundle,
)
from protoagi.storage.memory import MemoryStore


class MemoryFederationTests(unittest.TestCase):
    def test_signed_export_import_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = MemoryStore(root / "source.sqlite3")
            source.remember("federated coffee fact", ["telegram", "preference"])
            bundle = root / "bundle.json"
            result = export_memory_bundle(
                source,
                bundle,
                secret="shared-secret",
                source="lab-a",
                require_tags=["preference"],
            )
            self.assertEqual(result.exported, 1)
            target = MemoryStore(root / "target.sqlite3")
            first = import_memory_bundle(target, bundle, secret="shared-secret")
            second = import_memory_bundle(target, bundle, secret="shared-secret")
            self.assertEqual(first.imported, 1)
            self.assertEqual(second.skipped, 1)
            self.assertTrue(target.search("coffee"))

    def test_incremental_export_applies_deletions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = MemoryStore(root / "source.sqlite3")
            old_id = source.remember("federated old coffee fact", ["share"])
            first_bundle = root / "bundle-full.json"
            export_memory_bundle(
                source,
                first_bundle,
                secret="shared-secret",
                source="lab-a",
                require_tags=["share"],
            )
            target = MemoryStore(root / "target.sqlite3")
            initial_import = import_memory_bundle(target, first_bundle, secret="shared-secret")
            self.assertEqual(initial_import.imported, 1)
            since = source.get_kv("memory_federation:last_export_at")
            self.assertTrue(since)

            source.delete_memory(old_id)
            source.remember("federated new tea fact", ["share"])
            delta_bundle = root / "bundle-delta.json"
            delta = export_memory_bundle(
                source,
                delta_bundle,
                secret="shared-secret",
                source="lab-a",
                require_tags=["share"],
                since=since,
            )
            self.assertEqual(delta.exported, 1)
            self.assertEqual(delta.deleted, 1)
            imported_delta = import_memory_bundle(target, delta_bundle, secret="shared-secret")
            self.assertEqual(imported_delta.imported, 1)
            self.assertEqual(imported_delta.deleted, 1)
            self.assertFalse(target.search("old coffee", limit=5))
            self.assertTrue(target.search("new tea", limit=5))

    def test_incremental_export_replaces_changed_item(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = MemoryStore(root / "source.sqlite3")
            memory_id = source.remember("federated draft coffee fact", ["share"])
            full_bundle = root / "bundle-full.json"
            export_memory_bundle(
                source,
                full_bundle,
                secret="shared-secret",
                source="lab-a",
                require_tags=["share"],
            )
            target = MemoryStore(root / "target.sqlite3")
            import_memory_bundle(target, full_bundle, secret="shared-secret")
            since = source.get_kv("memory_federation:last_export_at")
            self.assertTrue(since)

            updated = source.update_memory(memory_id, text="federated final coffee fact")
            self.assertIsNotNone(updated)
            delta_bundle = root / "bundle-delta.json"
            delta = export_memory_bundle(
                source,
                delta_bundle,
                secret="shared-secret",
                source="lab-a",
                require_tags=["share"],
                since=since,
            )
            self.assertEqual(delta.exported, 1)
            self.assertEqual(delta.deleted, 1)
            imported = import_memory_bundle(target, delta_bundle, secret="shared-secret")
            self.assertEqual(imported.imported, 1)
            self.assertEqual(imported.deleted, 1)
            self.assertFalse(target.search("draft", limit=5))
            self.assertTrue(target.search("final coffee", limit=5))

    def test_signature_mismatch_rejects_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = MemoryStore(root / "source.sqlite3")
            source.remember("signed fact", ["tag"])
            bundle = root / "bundle.json"
            export_memory_bundle(source, bundle, secret="shared-secret")
            payload = json.loads(bundle.read_text(encoding="utf-8"))
            payload["items"][0]["text"] = "tampered"
            bundle.write_text(json.dumps(payload), encoding="utf-8")
            target = MemoryStore(root / "target.sqlite3")
            with self.assertRaises(MemoryFederationError):
                import_memory_bundle(target, bundle, secret="shared-secret")

    def test_cli_export_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source_db = root / "source.sqlite3"
            target_db = root / "target.sqlite3"
            source = MemoryStore(source_db)
            source.remember("cli federated fact", ["share"])
            bundle = root / "bundle.json"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "memory-export",
                            "--db",
                            str(source_db),
                            "--to",
                            str(bundle),
                            "--secret",
                            "secret",
                            "--tag",
                            "share",
                            "--since",
                            "2000-01-01T00:00:00+00:00",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "memory-import",
                            "--db",
                            str(target_db),
                            "--from",
                            str(bundle),
                            "--secret",
                            "secret",
                        ]
                    ),
                    0,
                )
            self.assertTrue(MemoryStore(target_db).search("cli federated"))


if __name__ == "__main__":
    unittest.main()
