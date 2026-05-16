"""Tests for the sticker description cache.

Stickers used to be picked at random within a pack, which produced
visually unrelated reactions (a creepy face next to "what a beautiful
girl"). The fix is to caption every sticker once via the vision model
and let the decision LLM pick by id from a list of described stickers.
This file covers the storage layer + the normalizer changes.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from protoagi.storage.memory import MemoryStore
from protoagi.telegram import normalize_sticker_choices


class StickerDescriptionStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_upsert_and_get(self) -> None:
        row = self.store.upsert_sticker_description(
            sticker_id="STICKER_A",
            set_name="Bocchi_the_Rock_sticker_pack2",
            emoji="😳",
            description="Дівчина зашарілася і відводить очі.",
        )
        self.assertEqual(row.sticker_id, "STICKER_A")
        self.assertEqual(row.set_name, "Bocchi_the_Rock_sticker_pack2")
        self.assertEqual(row.emoji, "😳")
        self.assertIn("зашарілася", row.description)
        roundtrip = self.store.get_sticker_description("STICKER_A")
        assert roundtrip is not None
        self.assertEqual(roundtrip.description, row.description)

    def test_upsert_bumps_attempt_count(self) -> None:
        self.store.upsert_sticker_description(
            sticker_id="S", set_name="pack", description=""
        )
        self.store.upsert_sticker_description(
            sticker_id="S", set_name="pack", description=""
        )
        row = self.store.get_sticker_description("S")
        assert row is not None
        self.assertEqual(row.attempt_count, 2)

    def test_upsert_persists_embedding(self) -> None:
        vec = [0.1, 0.2, 0.3, 0.4]
        self.store.upsert_sticker_description(
            sticker_id="V",
            set_name="pack",
            description="test",
            embedding=vec,
            embedding_model="bge-m3",
        )
        row = self.store.get_sticker_description("V")
        assert row is not None
        self.assertIsNotNone(row.embedding)
        assert row.embedding is not None
        self.assertEqual(len(row.embedding), 4)
        # float32 round-trip — allow small tolerance.
        for actual, expected in zip(row.embedding, vec):
            self.assertAlmostEqual(actual, expected, places=4)
        self.assertEqual(row.embedding_model, "bge-m3")

    def test_list_only_described_filters_empty(self) -> None:
        self.store.upsert_sticker_description(
            sticker_id="EMPTY", set_name="pack", description=""
        )
        self.store.upsert_sticker_description(
            sticker_id="FULL", set_name="pack", description="warm smile"
        )
        described = self.store.list_sticker_descriptions(only_described=True)
        self.assertEqual([row.sticker_id for row in described], ["FULL"])

    def test_list_undescribed_respects_max_attempts(self) -> None:
        # Three attempts at the same sticker, all returning empty.
        for _ in range(3):
            self.store.upsert_sticker_description(
                sticker_id="STUCK", set_name="pack", description="",
                failure_reason="vision returned empty",
            )
        pending = self.store.list_undescribed_stickers(max_attempts=3)
        # attempt_count == 3 is NOT < 3 → no longer surfaced.
        self.assertEqual(pending, [])

    def test_list_undescribed_filters_by_pack(self) -> None:
        self.store.upsert_sticker_description(
            sticker_id="A", set_name="pack1", description=""
        )
        self.store.upsert_sticker_description(
            sticker_id="B", set_name="pack2", description=""
        )
        pack1 = self.store.list_undescribed_stickers(set_name="pack1")
        self.assertEqual([r.sticker_id for r in pack1], ["A"])

    def test_mark_used_updates_timestamp(self) -> None:
        self.store.upsert_sticker_description(
            sticker_id="U", set_name="pack", description="x"
        )
        before = self.store.get_sticker_description("U")
        assert before is not None
        self.assertIsNone(before.last_used_at)
        self.store.mark_sticker_used("U")
        after = self.store.get_sticker_description("U")
        assert after is not None
        self.assertIsNotNone(after.last_used_at)

    def test_count_filters(self) -> None:
        self.store.upsert_sticker_description(
            sticker_id="A", set_name="pack1", description=""
        )
        self.store.upsert_sticker_description(
            sticker_id="B", set_name="pack1", description="ok"
        )
        self.store.upsert_sticker_description(
            sticker_id="C", set_name="pack2", description="ok"
        )
        self.assertEqual(self.store.count_sticker_descriptions(), 3)
        self.assertEqual(
            self.store.count_sticker_descriptions(set_name="pack1"), 2
        )
        self.assertEqual(
            self.store.count_sticker_descriptions(only_described=True), 2
        )


class StickerChoiceNormalizerTests(unittest.TestCase):
    def test_sticker_id_path_accepted(self) -> None:
        choices = normalize_sticker_choices([
            {"sticker_id": "CAACAgIAAxk...", "reason": "warm reaction"}
        ])
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0]["sticker_id"], "CAACAgIAAxk...")
        self.assertNotIn("pack", choices[0])

    def test_pack_legacy_path_still_works(self) -> None:
        choices = normalize_sticker_choices([
            {"pack": "Bocchi_the_Rock_sticker_pack2", "emoji": "🙂", "reason": "fun"}
        ])
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0]["pack"], "Bocchi_the_Rock_sticker_pack2")
        self.assertEqual(choices[0]["emoji"], "🙂")

    def test_pack_alias_normalised(self) -> None:
        choices = normalize_sticker_choices([{"pack": "bocchi", "reason": "x"}])
        self.assertEqual(choices[0]["pack"], "Bocchi_the_Rock_sticker_pack2")

    def test_empty_pack_and_no_sticker_id_dropped(self) -> None:
        choices = normalize_sticker_choices([{"reason": "nothing"}])
        self.assertEqual(choices, [])

    def test_new_packs_resolve_via_alias(self) -> None:
        # Sanity: aliases for the newly-added packs round-trip.
        for alias, expected in (
            ("bambuko", "Bambuko_debilizm_UA"),
            ("cringe", "cringeperekladpak_by_fStikBot"),
            ("eminence", "omnvrtEminenceInShadow"),
            ("heridium", "HeridiumPack"),
            ("pomoyka", "pomoyka_vid_mene"),
        ):
            choices = normalize_sticker_choices([{"pack": alias, "reason": "x"}])
            self.assertEqual(
                choices[0]["pack"], expected,
                f"alias {alias} should resolve to {expected}",
            )

    def test_cap_at_two(self) -> None:
        choices = normalize_sticker_choices([
            {"sticker_id": f"S{i}", "reason": "x"} for i in range(5)
        ])
        self.assertEqual(len(choices), 2)


if __name__ == "__main__":
    unittest.main()
