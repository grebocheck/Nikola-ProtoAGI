"""Tests for memory_conflicts: detection + bookkeeping.

A conflict here is two memory items that the consolidate pass did NOT
auto-merge (cosine similarity in the [0.78, 0.92) band, configurable).
We don't claim semantic contradiction yet — these are review candidates
the persona can act on later. This file exercises both the storage
primitives and the scan loop in MemoryService.
"""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from protoagi.storage.memory import MemoryStore
from protoagi.storage.models import (
    CONFLICT_STATUS_DISMISSED,
    CONFLICT_STATUS_KEPT_BOTH,
    CONFLICT_STATUS_SUPERSEDED,
    CONFLICT_STATUS_UNRESOLVED,
)
from protoagi.storage.service import MemoryService


def _unit(*values: float) -> list[float]:
    norm = math.sqrt(sum(v * v for v in values))
    return [v / norm for v in values] if norm > 0 else list(values)


def _rotate_2d(vec: list[float], angle_radians: float) -> list[float]:
    """Rotate the first two coords by ``angle_radians``, leave the rest.

    Cosine similarity between rotated and original = cos(angle), so we
    can directly target a similarity by choosing the angle.
    """

    cos_a = math.cos(angle_radians)
    sin_a = math.sin(angle_radians)
    out = list(vec)
    if len(out) >= 2:
        x, y = out[0], out[1]
        out[0] = x * cos_a - y * sin_a
        out[1] = x * sin_a + y * cos_a
    return out


def _angle_for_similarity(target_cos: float) -> float:
    return math.acos(max(-1.0, min(1.0, target_cos)))


class ConflictStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")
        self.a = self.store.store_memory("fact a")
        self.b = self.store.store_memory("fact b")
        self.c = self.store.store_memory("fact c")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_record_conflict_normalizes_id_order(self) -> None:
        # Pass the bigger id first; row should still store smaller as ``a``.
        cid = self.store.record_conflict(
            self.b, self.a, similarity=0.85, persona_key="solomiya"
        )
        assert cid is not None
        conflict = self.store.get_conflict(cid)
        assert conflict is not None
        self.assertEqual(conflict.memory_a_id, min(self.a, self.b))
        self.assertEqual(conflict.memory_b_id, max(self.a, self.b))
        self.assertEqual(conflict.similarity, 0.85)
        self.assertEqual(conflict.resolution_status, CONFLICT_STATUS_UNRESOLVED)

    def test_record_conflict_dedupes(self) -> None:
        first = self.store.record_conflict(self.a, self.b, similarity=0.85)
        second = self.store.record_conflict(self.b, self.a, similarity=0.90)
        self.assertIsNotNone(first)
        # Existing pair: returns None and keeps the original row unchanged.
        self.assertIsNone(second)
        all_conflicts = self.store.list_unresolved_conflicts()
        self.assertEqual(len(all_conflicts), 1)
        self.assertEqual(all_conflicts[0].similarity, 0.85)

    def test_record_conflict_rejects_self_pair(self) -> None:
        result = self.store.record_conflict(self.a, self.a, similarity=0.99)
        self.assertIsNone(result)

    def test_resolve_conflict_marks_winner(self) -> None:
        cid = self.store.record_conflict(self.a, self.b, similarity=0.85)
        assert cid is not None
        updated = self.store.resolve_conflict(
            cid, status=CONFLICT_STATUS_SUPERSEDED, winner_id=self.a
        )
        assert updated is not None
        self.assertEqual(updated.resolution_status, CONFLICT_STATUS_SUPERSEDED)
        self.assertEqual(updated.resolution_winner_id, self.a)
        self.assertIsNotNone(updated.resolved_at)

    def test_resolve_conflict_requires_winner_for_superseded(self) -> None:
        cid = self.store.record_conflict(self.a, self.b, similarity=0.85)
        with self.assertRaises(ValueError):
            self.store.resolve_conflict(cid, status=CONFLICT_STATUS_SUPERSEDED)

    def test_resolve_conflict_rejects_unknown_status(self) -> None:
        cid = self.store.record_conflict(self.a, self.b, similarity=0.85)
        with self.assertRaises(ValueError):
            self.store.resolve_conflict(cid, status="finished")

    def test_list_unresolved_excludes_resolved(self) -> None:
        cid_a = self.store.record_conflict(self.a, self.b, similarity=0.85)
        cid_b = self.store.record_conflict(self.a, self.c, similarity=0.88)
        self.store.resolve_conflict(cid_a, status=CONFLICT_STATUS_DISMISSED)
        unresolved = self.store.list_unresolved_conflicts()
        self.assertEqual([c.id for c in unresolved], [cid_b])

    def test_list_unresolved_filters_by_persona(self) -> None:
        self.store.record_conflict(self.a, self.b, similarity=0.85, persona_key="solomiya")
        self.store.record_conflict(self.a, self.c, similarity=0.85, persona_key="mykola")
        solo = self.store.list_unresolved_conflicts(persona_key="solomiya")
        myk = self.store.list_unresolved_conflicts(persona_key="mykola")
        self.assertEqual(len(solo), 1)
        self.assertEqual(len(myk), 1)
        self.assertEqual(solo[0].memory_a_id, min(self.a, self.b))

    def test_conflicts_for_memory_returns_either_side(self) -> None:
        self.store.record_conflict(self.a, self.b, similarity=0.85)
        self.store.record_conflict(self.a, self.c, similarity=0.85)
        for_a = self.store.conflicts_for_memory(self.a)
        for_b = self.store.conflicts_for_memory(self.b)
        self.assertEqual(len(for_a), 2)
        self.assertEqual(len(for_b), 1)

    def test_count_conflicts_filters(self) -> None:
        cid = self.store.record_conflict(self.a, self.b, similarity=0.85, persona_key="solomiya")
        self.store.record_conflict(self.a, self.c, similarity=0.85, persona_key="mykola")
        self.assertEqual(self.store.count_conflicts(), 2)
        self.assertEqual(self.store.count_conflicts(persona_key="solomiya"), 1)
        self.store.resolve_conflict(cid, status=CONFLICT_STATUS_KEPT_BOTH)
        self.assertEqual(
            self.store.count_conflicts(status=CONFLICT_STATUS_UNRESOLVED), 1
        )

    def test_schema_version_is_recorded(self) -> None:
        self.assertEqual(self.store.get_kv("schema_version"), "7")


class ConflictScanTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = MemoryStore(Path(self.tmp.name) / "memory.sqlite3")
        # MemoryService normally takes an LLM client + embedding setup,
        # but for the scan we only need the embedding INDEX-free path
        # (it reads stored embeddings via raw SQL).
        self.service = MemoryService(store=self.store)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def _seed_with_vectors(self, *, base: list[float], variants: list[float]) -> list[int]:
        """Insert one base memory + one per variant, with rotated embeddings.

        Variants are angles (in radians); each produces a memory item
        whose embedding has cos(angle) similarity to the base vector.
        """

        ids: list[int] = []
        base_id = self.store.store_memory(
            "anchor",
            persona_key="solomiya",
            embedding=base,
            embedding_model="fake",
        )
        ids.append(base_id)
        for angle in variants:
            rotated = _rotate_2d(base, angle)
            mid = self.store.store_memory(
                f"variant {angle:.2f}",
                persona_key="solomiya",
                embedding=rotated,
                embedding_model="fake",
            )
            ids.append(mid)
        return ids

    def test_scan_records_pair_in_target_band(self) -> None:
        # Cosine 0.85 ⇒ within default [0.78, 0.92) band.
        base = _unit(1.0, 0.0, 0.0)
        ids = self._seed_with_vectors(
            base=base, variants=[_angle_for_similarity(0.85)]
        )
        added = self.service.scan_for_conflicts(persona_key="solomiya")
        self.assertEqual(added, 1)
        conflicts = self.store.list_unresolved_conflicts()
        self.assertEqual(len(conflicts), 1)
        self.assertAlmostEqual(conflicts[0].similarity, 0.85, places=3)
        self.assertEqual(
            sorted([conflicts[0].memory_a_id, conflicts[0].memory_b_id]),
            sorted(ids),
        )

    def test_scan_skips_pair_above_consolidate_threshold(self) -> None:
        # Cosine ~0.95 — consolidate's territory; our scan should ignore.
        base = _unit(1.0, 0.0, 0.0)
        self._seed_with_vectors(
            base=base, variants=[_angle_for_similarity(0.95)]
        )
        added = self.service.scan_for_conflicts(persona_key="solomiya")
        self.assertEqual(added, 0)

    def test_scan_skips_pair_below_minimum_similarity(self) -> None:
        # Cosine ~0.5 — too dissimilar to count as related.
        base = _unit(1.0, 0.0, 0.0)
        self._seed_with_vectors(
            base=base, variants=[_angle_for_similarity(0.5)]
        )
        added = self.service.scan_for_conflicts(persona_key="solomiya")
        self.assertEqual(added, 0)

    def test_scan_is_idempotent(self) -> None:
        base = _unit(1.0, 0.0, 0.0)
        self._seed_with_vectors(
            base=base, variants=[_angle_for_similarity(0.85)]
        )
        first = self.service.scan_for_conflicts(persona_key="solomiya")
        second = self.service.scan_for_conflicts(persona_key="solomiya")
        self.assertEqual(first, 1)
        self.assertEqual(second, 0)

    def test_scan_ignores_superseded_items(self) -> None:
        base = _unit(1.0, 0.0, 0.0)
        ids = self._seed_with_vectors(
            base=base, variants=[_angle_for_similarity(0.85)]
        )
        # Mark the variant as superseded → scan should leave the pair alone.
        self.store.supersede(ids[1], ids[0])
        added = self.service.scan_for_conflicts(persona_key="solomiya")
        self.assertEqual(added, 0)

    def test_scan_respects_persona_scope(self) -> None:
        # solomiya's vectors should not conflict with mykola's even if
        # text similarity is high.
        base = _unit(1.0, 0.0, 0.0)
        rotated = _rotate_2d(base, _angle_for_similarity(0.85))
        self.store.store_memory(
            "solo_a", persona_key="solomiya",
            embedding=base, embedding_model="fake",
        )
        self.store.store_memory(
            "myk_a", persona_key="mykola",
            embedding=rotated, embedding_model="fake",
        )
        added = self.service.scan_for_conflicts(persona_key="solomiya")
        self.assertEqual(added, 0)
        added_myk = self.service.scan_for_conflicts(persona_key="mykola")
        self.assertEqual(added_myk, 0)


if __name__ == "__main__":
    unittest.main()
