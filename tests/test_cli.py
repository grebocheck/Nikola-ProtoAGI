import unittest

from protoagi.cli import build_parser


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


if __name__ == "__main__":
    unittest.main()
