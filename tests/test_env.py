from pathlib import Path
import os
import tempfile
import unittest

from protoagi.env import env_bool, env_int, load_dotenv


class EnvTests(unittest.TestCase):
    def test_load_dotenv_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "PROTOAGI_MODEL=from-file",
                        "QUOTED='hello world'",
                        "export EXPORTED=yes",
                    ]
                ),
                encoding="utf-8",
            )
            os.environ["PROTOAGI_MODEL"] = "existing"
            try:
                loaded = load_dotenv(path)
                self.assertEqual(loaded, 2)
                self.assertEqual(os.environ["PROTOAGI_MODEL"], "existing")
                self.assertEqual(os.environ["QUOTED"], "hello world")
                self.assertEqual(os.environ["EXPORTED"], "yes")
            finally:
                for key in ["PROTOAGI_MODEL", "QUOTED", "EXPORTED"]:
                    os.environ.pop(key, None)

    def test_env_int_and_bool_fallbacks(self) -> None:
        os.environ["BAD_INT"] = "nope"
        os.environ["BOOL_OFF"] = "off"
        try:
            self.assertEqual(env_int("BAD_INT", 7), 7)
            self.assertFalse(env_bool("BOOL_OFF", True))
            self.assertTrue(env_bool("MISSING_BOOL", True))
        finally:
            os.environ.pop("BAD_INT", None)
            os.environ.pop("BOOL_OFF", None)
