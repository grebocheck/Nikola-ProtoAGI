import tempfile
import unittest
from pathlib import Path

from protoagi.storage.memory import MemoryStore
from protoagi.telegram.group_gate import (
    GATE_KV_PREFIX,
    GroupGateConfig,
    GroupReactivityGate,
    parse_trigger_keywords,
)


def _temp_memory() -> tuple[MemoryStore, tempfile.TemporaryDirectory]:
    tmp = tempfile.TemporaryDirectory()
    store = MemoryStore(Path(tmp.name) / "memory.sqlite3")
    return store, tmp


class GroupReactivityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.memory, self._tmp = _temp_memory()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _gate(
        self,
        *,
        ratio: float = 0.0,
        cooldown: int = 120,
        keywords: tuple[str, ...] = (),
        clock_value: float = 0.0,
        sample_value: float = 1.0,
    ) -> GroupReactivityGate:
        config = GroupGateConfig(
            cooldown_seconds=cooldown,
            passive_reply_ratio=ratio,
            trigger_keywords=keywords,
        )
        return GroupReactivityGate(
            self.memory,
            config,
            clock=lambda: clock_value,
            sampler=lambda: sample_value,
        )

    def test_private_chat_always_allowed(self) -> None:
        gate = self._gate()
        decision = gate.evaluate(
            chat_id="1",
            chat_type="private",
            text="hello",
            addressed=False,
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "private")

    def test_addressed_group_allowed(self) -> None:
        gate = self._gate()
        decision = gate.evaluate(
            chat_id="1",
            chat_type="group",
            text="anything",
            addressed=True,
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "addressed")

    def test_unaddressed_group_blocked_by_default(self) -> None:
        gate = self._gate()
        decision = gate.evaluate(
            chat_id="1",
            chat_type="supergroup",
            text="двоє розмовляють між собою",
            addressed=False,
        )
        self.assertFalse(decision.allow)
        self.assertEqual(decision.reason, "passive_disabled")

    def test_keyword_match_lets_through(self) -> None:
        gate = self._gate(keywords=("Микола",))
        decision = gate.evaluate(
            chat_id="1",
            chat_type="group",
            text="а що про це думає Микола?",
            addressed=False,
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "keyword")

    def test_keyword_match_is_case_insensitive_and_unicode_safe(self) -> None:
        gate = self._gate(keywords=("Соломія",))
        decision = gate.evaluate(
            chat_id="1",
            chat_type="group",
            text="СОЛОМІЯ, привіт",
            addressed=False,
        )
        self.assertTrue(decision.allow)

    def test_keyword_does_not_match_inside_word(self) -> None:
        gate = self._gate(keywords=("ai",))
        decision = gate.evaluate(
            chat_id="1",
            chat_type="group",
            text="train",
            addressed=False,
        )
        self.assertFalse(decision.allow)

    def test_passive_sample_passes_when_roll_under_ratio(self) -> None:
        gate = self._gate(ratio=0.05, sample_value=0.01)
        decision = gate.evaluate(
            chat_id="42",
            chat_type="group",
            text="just chatting",
            addressed=False,
        )
        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "passive_sample")
        self.assertIsNotNone(self.memory.get_kv(GATE_KV_PREFIX + "42"))

    def test_passive_sample_skipped_when_roll_above_ratio(self) -> None:
        gate = self._gate(ratio=0.05, sample_value=0.5)
        decision = gate.evaluate(
            chat_id="42",
            chat_type="group",
            text="just chatting",
            addressed=False,
        )
        self.assertFalse(decision.allow)
        self.assertEqual(decision.reason, "passive_skip")
        self.assertIsNone(self.memory.get_kv(GATE_KV_PREFIX + "42"))

    def test_cooldown_blocks_back_to_back_passive_samples(self) -> None:
        first = GroupReactivityGate(
            self.memory,
            GroupGateConfig(cooldown_seconds=120, passive_reply_ratio=1.0),
            clock=lambda: 1000.0,
            sampler=lambda: 0.0,
        )
        first_decision = first.evaluate(
            chat_id="42",
            chat_type="group",
            text="anything",
            addressed=False,
        )
        self.assertTrue(first_decision.allow)
        second = GroupReactivityGate(
            self.memory,
            GroupGateConfig(cooldown_seconds=120, passive_reply_ratio=1.0),
            clock=lambda: 1010.0,
            sampler=lambda: 0.0,
        )
        second_decision = second.evaluate(
            chat_id="42",
            chat_type="group",
            text="anything",
            addressed=False,
        )
        self.assertFalse(second_decision.allow)
        self.assertEqual(second_decision.reason, "cooldown")

    def test_cooldown_releases_after_window(self) -> None:
        early = GroupReactivityGate(
            self.memory,
            GroupGateConfig(cooldown_seconds=120, passive_reply_ratio=1.0),
            clock=lambda: 1000.0,
            sampler=lambda: 0.0,
        )
        early.evaluate(chat_id="9", chat_type="group", text="x", addressed=False)
        late = GroupReactivityGate(
            self.memory,
            GroupGateConfig(cooldown_seconds=120, passive_reply_ratio=1.0),
            clock=lambda: 1300.0,
            sampler=lambda: 0.0,
        )
        decision = late.evaluate(chat_id="9", chat_type="group", text="x", addressed=False)
        self.assertTrue(decision.allow)
        self.assertEqual(decision.reason, "passive_sample")


class TriggerKeywordParsingTests(unittest.TestCase):
    def test_empty_returns_empty_tuple(self) -> None:
        self.assertEqual(parse_trigger_keywords(""), ())
        self.assertEqual(parse_trigger_keywords(None), ())

    def test_csv_is_split_and_trimmed(self) -> None:
        keywords = parse_trigger_keywords("Микола, Соломія,  AGI ")
        self.assertEqual(keywords, ("Микола", "Соломія", "AGI"))


if __name__ == "__main__":
    unittest.main()
