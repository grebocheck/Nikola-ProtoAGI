from pathlib import Path
import tempfile
import unittest

from protoagi.storage.memory import MemoryStore
from protoagi.telegram.style import ReplyStyleTuner


class ReplyStyleTunerTests(unittest.TestCase):
    def test_records_reply_engagement_for_last_sent_arm(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            memory = MemoryStore(Path(tmp) / "memory.sqlite3")
            tuner = ReplyStyleTuner(memory)
            choice = tuner.choose("123")
            tuner.record_sent(
                "123",
                arm=choice.arm,
                reply_chars=42,
                sticker_count=1,
                message_count=1,
            )
            tuner.record_incoming_reply("123")
            state = tuner.state_payload("123")
            self.assertEqual(state["signals"]["reply"], 1)
            self.assertEqual(state["arms"][choice.arm]["trials"], 1)
            self.assertGreater(state["arms"][choice.arm]["successes"], 0)


if __name__ == "__main__":
    unittest.main()
