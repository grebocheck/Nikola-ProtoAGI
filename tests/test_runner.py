import threading
import time
import unittest
from unittest.mock import MagicMock

from protoagi.telegram.runner import BotRunner


class FakeBot:
    """Minimal NikolaBot stand-in for runner tests."""

    def __init__(self) -> None:
        self.poll_calls = 0
        self.initiative_calls = 0
        self.reminder_calls = 0
        self.reflection_calls = 0
        self._poll_event = threading.Event()
        self.error_log_path = "/tmp/dummy"

    def poll_once(self) -> int:
        self.poll_calls += 1
        # Block briefly to mimic long-poll without saturating CPU.
        self._poll_event.wait(timeout=0.05)
        return 0

    def maybe_run_initiative(self) -> int:
        self.initiative_calls += 1
        return 0

    def maybe_dispatch_reminders(self) -> int:
        self.reminder_calls += 1
        return 0

    def maybe_run_reflection(self) -> bool:
        self.reflection_calls += 1
        return False

    def _log_loop_exception(self, exc: BaseException) -> None:
        return None


class BotRunnerTests(unittest.TestCase):
    def test_worker_runs_independently_of_polling(self) -> None:
        bot = FakeBot()
        runner = BotRunner(bot, worker_tick_seconds=0.05)
        runner.start()
        try:
            time.sleep(0.3)
        finally:
            runner.stop(timeout=2.0)
        # Worker should have ticked multiple times and called every periodic
        # branch.
        self.assertGreater(bot.initiative_calls, 1)
        self.assertGreater(bot.reminder_calls, 1)
        self.assertGreater(bot.reflection_calls, 1)

    def test_stop_releases_worker(self) -> None:
        bot = FakeBot()
        runner = BotRunner(bot, worker_tick_seconds=0.05)
        runner.start()
        runner.stop(timeout=2.0)
        # After stop the worker thread is gone.
        self.assertIsNone(runner._worker)

    def test_run_polls_and_workers_until_stopped(self) -> None:
        bot = FakeBot()
        runner = BotRunner(bot, worker_tick_seconds=0.05)

        def stop_soon() -> None:
            time.sleep(0.2)
            runner._stop.set()
            bot._poll_event.set()

        threading.Thread(target=stop_soon, daemon=True).start()
        runner.run()
        self.assertGreater(bot.poll_calls, 0)
        self.assertGreater(bot.reminder_calls, 0)


if __name__ == "__main__":
    unittest.main()
