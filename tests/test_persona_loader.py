import json
import tempfile
import unittest
from pathlib import Path

from protoagi.persona import (
    DEFAULT_PERSONA_KEY,
    available_persona_keys,
    get_persona,
    load_personas,
    reload_personas,
)


class PersonaLoaderTests(unittest.TestCase):
    def test_builtin_personas_load_when_dir_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            registry = load_personas(Path(tmp), force=True)
            self.assertIn("mykola", registry)
            self.assertIn("solomiya", registry)
        # restore default registry for the rest of the suite
        reload_personas()

    def test_custom_persona_overrides_builtin(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "mykola.json").write_text(
                json.dumps(
                    {
                        "key": "mykola",
                        "display_name": "Микола Тестовий",
                        "memory_tag": "nikola",
                        "aliases": ["mykola"],
                        "self_model": "alt model",
                        "user_model": "alt user",
                        "relationship_model": "alt relationship",
                        "decision_style": ["alt decision"],
                        "reply_style": ["alt reply"],
                        "memory_policy": ["alt memory"],
                        "initiative_policy": ["alt initiative"],
                        "start_message": "Привіт від тесту",
                        "self_lore": [],
                    }
                ),
                encoding="utf-8",
            )
            registry = load_personas(Path(tmp), force=True)
            self.assertEqual(registry["mykola"].display_name, "Микола Тестовий")
        reload_personas()

    def test_get_persona_falls_back_to_default(self) -> None:
        persona = get_persona("nonexistent")
        self.assertEqual(persona.key, DEFAULT_PERSONA_KEY)

    def test_available_persona_keys_contains_builtins(self) -> None:
        keys = set(available_persona_keys())
        self.assertIn("mykola", keys)
        self.assertIn("solomiya", keys)


if __name__ == "__main__":
    unittest.main()
