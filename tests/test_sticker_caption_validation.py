"""Tests for the sticker caption quality gate.

The vision model behind the describer (SmolVLM2-class) occasionally
drifts: it answers in Japanese, hallucinates Ukrainian-looking words
with invented suffixes, or just returns garbage. The gate in
``sticker_describer`` rejects those and triggers a stricter-prompt
retry. These tests pin the rule set.
"""

from __future__ import annotations

import unittest

from protoagi.telegram.sticker_describer import (
    _cyrillic_ratio,
    _has_cjk,
    _is_acceptable_caption,
)


class CjkDetectionTests(unittest.TestCase):
    def test_pure_ukrainian_is_clean(self) -> None:
        self.assertFalse(_has_cjk("Дівчина усміхається."))

    def test_hiragana_flagged(self) -> None:
        self.assertTrue(_has_cjk("ありがとう"))

    def test_katakana_flagged(self) -> None:
        self.assertTrue(_has_cjk("スティッカー"))

    def test_kanji_flagged(self) -> None:
        self.assertTrue(_has_cjk("可愛い"))

    def test_hangul_flagged(self) -> None:
        self.assertTrue(_has_cjk("안녕하세요"))

    def test_mixed_ukrainian_with_cjk_still_flagged(self) -> None:
        self.assertTrue(_has_cjk("Дівчина каже 可愛い."))


class CyrillicRatioTests(unittest.TestCase):
    def test_pure_ukrainian_is_high(self) -> None:
        self.assertGreaterEqual(_cyrillic_ratio("Дівчина усміхається."), 0.95)

    def test_pure_english_is_low(self) -> None:
        self.assertLessEqual(_cyrillic_ratio("Girl is smiling."), 0.05)

    def test_quoted_latin_is_ignored(self) -> None:
        # Real captions often include a quoted Latin meme phrase. We
        # only care that the *narration* is Ukrainian.
        text = 'Дівчина тримає табличку з написом "OK Boomer".'
        self.assertGreaterEqual(_cyrillic_ratio(text), 0.95)

    def test_empty_caption_returns_one(self) -> None:
        # Other gates handle the "too short" case; ratio should not
        # crash or false-positive on empty input.
        self.assertEqual(_cyrillic_ratio(""), 1.0)


class AcceptableCaptionTests(unittest.TestCase):
    def test_normal_caption_accepted(self) -> None:
        ok, reason = _is_acceptable_caption(
            "На стікері аніме-дівчина зашарілася і відводить очі."
        )
        self.assertTrue(ok, reason)
        self.assertEqual(reason, "")

    def test_japanese_rejected(self) -> None:
        ok, reason = _is_acceptable_caption(
            "На стікері аніме-дівчина каже ありがとう."
        )
        self.assertFalse(ok)
        self.assertIn("CJK", reason)

    def test_too_short_rejected(self) -> None:
        ok, reason = _is_acceptable_caption("Стікер.")
        self.assertFalse(ok)
        self.assertIn("short", reason)

    def test_known_hallucination_rejected(self) -> None:
        # "видосикер" — invented from "videostiker"; appeared multiple
        # times in real SmolVLM2 output during testing.
        ok, reason = _is_acceptable_caption(
            "Видосикер з аніме-дівчиною, що зашарілася і відводить очі."
        )
        self.assertFalse(ok)
        self.assertIn("hallucination", reason)
        self.assertIn("видосик", reason)

    def test_mostly_english_rejected(self) -> None:
        ok, reason = _is_acceptable_caption(
            "An anime girl is smiling brightly in the picture."
        )
        self.assertFalse(ok)
        self.assertIn("Cyrillic", reason)

    def test_legitimate_latin_quote_passes(self) -> None:
        # Narration is Ukrainian; quote can be English.
        ok, reason = _is_acceptable_caption(
            'На стікері аніме-дівчина тримає табличку з написом "Stop the war".'
        )
        self.assertTrue(ok, reason)


if __name__ == "__main__":
    unittest.main()
