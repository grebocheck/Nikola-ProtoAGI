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
        if offset is None:
            return list(self.updates)
        return [update for update in self.updates if int(update["update_id"]) >= int(offset)]


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


class FlakyBot(SlowBot):
    def __init__(self) -> None:
        super().__init__()
        self.failed_once = False
        self.errors: list[str] = []

    def process_update(self, update: dict) -> bool:
        update_id = int(update["update_id"])
        if update_id == 2 and not self.failed_once:
            self.failed_once = True
            raise RuntimeError("temporary failure")
        self.processed.append(update_id)
        return True

    def _log_loop_exception(self, exc: BaseException) -> None:
        self.errors.append(str(exc))


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

    def test_failed_update_is_replayed_next_poll(self) -> None:
        bot = FlakyBot()
        runner = AsyncBotRunner(bot, max_concurrent_updates=2)

        first = asyncio.run(runner.poll_once())
        self.assertEqual(first, 1)
        self.assertEqual(bot.memory.values[OFFSET_KEY], "2")
        self.assertIn("temporary failure", bot.errors[0])

        second = asyncio.run(runner.poll_once())
        self.assertEqual(second, 1)
        self.assertEqual(bot.processed, [1, 2])
        self.assertEqual(bot.memory.values[OFFSET_KEY], "3")


if __name__ == "__main__":
    unittest.main()
