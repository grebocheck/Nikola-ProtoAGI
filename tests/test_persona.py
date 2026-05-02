import os
import unittest

from protoagi.persona import get_persona, resolve_persona_key
from protoagi.telegram_bot import TelegramConfig


class PersonaTests(unittest.TestCase):
    def test_resolve_persona_aliases(self) -> None:
        self.assertEqual(resolve_persona_key("nikola"), "mykola")
        self.assertEqual(resolve_persona_key("Соломія"), "solomiya")
        self.assertEqual(resolve_persona_key("unknown"), "mykola")

    def test_persona_prompt_block_contains_deep_identity(self) -> None:
        solomiya = get_persona("solomiya")
        prompt = solomiya.prompt_block()
        self.assertIn("самодостатня", prompt)
        self.assertIn("Telegram-памʼять спільна", prompt)

    def test_telegram_config_reads_persona_from_env(self) -> None:
        old = os.environ.get("PROTOAGI_TELEGRAM_PERSONA")
        try:
            os.environ["PROTOAGI_TELEGRAM_PERSONA"] = "solomiya"
            config = TelegramConfig.from_env()
            self.assertEqual(config.persona_key, "solomiya")
            self.assertEqual(config.bot_name, "Соломія")
        finally:
            if old is None:
                os.environ.pop("PROTOAGI_TELEGRAM_PERSONA", None)
            else:
                os.environ["PROTOAGI_TELEGRAM_PERSONA"] = old


if __name__ == "__main__":
    unittest.main()
