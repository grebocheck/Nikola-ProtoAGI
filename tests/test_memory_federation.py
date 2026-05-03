import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from protoagi.cli import main
from protoagi.memory import MemoryStore
from protoagi.memory_federation import (
    MemoryFederationError,
    export_memory_bundle,
    import_memory_bundle,
)


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
