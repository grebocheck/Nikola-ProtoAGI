import unittest
import tempfile
import contextlib
import io
from pathlib import Path

from protoagi.cli import build_parser, classify_tool_response_message, main
from protoagi.memory import SCOPE_GLOBAL, SCOPE_USER, MemoryStore


class CliParserTests(unittest.TestCase):
    def test_telegram_accepts_instance_overrides(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "telegram",
                "--db",
                "data/solomiya.sqlite3",
                "--persona",
                "solomiya",
                "--token",
                "token",
            ]
        )
        self.assertEqual(args.command, "telegram")
        self.assertEqual(args.db, "data/solomiya.sqlite3")
        self.assertEqual(args.persona, "solomiya")

    def test_bench_tools_classifier_detects_tool_paths(self) -> None:
        self.assertEqual(
            classify_tool_response_message({"tool_calls": [{"function": {"name": "recall"}}]}),
            "tool_calls",
        )
        self.assertEqual(
            classify_tool_response_message(
                {"content": '{"tool_request": {"name": "recall", "arguments": {}}}'}
            ),
            "tool_request",
        )
        self.assertEqual(
            classify_tool_response_message(
                {
                    "tool_calls": [{"function": {"name": "recall"}}],
                    "content": '{"tool_request": {"name": "recall", "arguments": {}}}',
                }
            ),
            "both",
        )
        self.assertEqual(classify_tool_response_message({"content": "{}"}), "neither")

    def test_memory_rescope_cli_migrates_legacy_global_user_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "memory.sqlite3"
            memory = MemoryStore(db_path)
            rowid = memory.store_memory(
                "legacy memory",
                scope=SCOPE_GLOBAL,
                tags=["telegram", "telegram_global", "user:telegram:7", "source_chat:99"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(["memory-rescope", "--db", str(db_path), "--to", "user", "--json"]),
                    0,
                )
            migrated = MemoryStore(db_path).get_memory(rowid)
            assert migrated is not None
            self.assertEqual(migrated.scope, SCOPE_USER)
            self.assertEqual(migrated.user_id, "telegram:7")
            self.assertEqual(migrated.chat_id, "99")


if __name__ == "__main__":
    unittest.main()
