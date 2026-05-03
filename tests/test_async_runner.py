import asyncio
import time
import unittest
from dataclasses import dataclass

from protoagi.telegram.async_runner import AsyncBotRunner
from protoagi.telegram.constants import OFFSET_KEY


class FakeMemory:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    def get_kv(self, key: str) -> str | None:
        return self.values.get(key)

    def set_kv(self, key: str, value: str) -> None:
        self.values[key] = value


class FakeTelegram:
    def __init__(self) -> None:
        self.updates = [
            {"update_id": 1, "message": {"text": "one"}},
            {"update_id": 2, "message": {"text": "two"}},
        ]

    def get_updates(self, *, offset, timeout_seconds, allowed_updates):
        return list(self.updates)


@dataclass(slots=True)
class FakeTelegramConfig:
    poll_timeout_seconds: int = 0


class SlowBot:
    def __init__(self) -> None:
        self.memory = FakeMemory()
        self.telegram = FakeTelegram()
        self.telegram_config = FakeTelegramConfig()
        self.processed: list[int] = []
        self.error_log_path = "none"

    def process_update(self, update: dict) -> bool:
        time.sleep(0.2)
        self.processed.append(int(update["update_id"]))
        return True

    def maybe_run_initiative(self) -> int:
        return 0

    def maybe_dispatch_reminders(self) -> int:
        return 0

    def maybe_run_reflection(self) -> bool:
        return False

    def _log_loop_exception(self, exc: BaseException) -> None:
        return None


class AsyncBotRunnerTests(unittest.TestCase):
    def test_poll_once_processes_updates_concurrently(self) -> None:
        bot = SlowBot()
        runner = AsyncBotRunner(bot, max_concurrent_updates=2)

        async def run_once() -> tuple[int, float]:
            started = time.perf_counter()
            processed = await runner.poll_once()
            return processed, time.perf_counter() - started

        processed, elapsed = asyncio.run(run_once())
        self.assertEqual(processed, 2)
        self.assertCountEqual(bot.processed, [1, 2])
        self.assertLess(elapsed, 0.35)
        self.assertEqual(bot.memory.values[OFFSET_KEY], "3")


if __name__ == "__main__":
    unittest.main()
