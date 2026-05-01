import unittest

from protoagi.harmony import clean_model_content


class HarmonyTests(unittest.TestCase):
    def test_prefers_final_channel(self) -> None:
        raw = (
            "<|channel|>analysis<|message|>private reasoning"
            "<|channel|>final<|message|>Visible answer<|end|>"
        )
        self.assertEqual(clean_model_content(raw), "Visible answer")

    def test_strips_analysis_without_final(self) -> None:
        raw = "<|channel|>analysis<|message|>private"
        self.assertEqual(clean_model_content(raw), "")

    def test_normalizes_unicode_hyphen_spaces(self) -> None:
        self.assertEqual(clean_model_content("gpt\u2011oss uses 3.8\u202fGB"), "gpt-oss uses 3.8 GB")


if __name__ == "__main__":
    unittest.main()
