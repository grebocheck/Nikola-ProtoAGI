"""Threaded supervisor for the Telegram bot.

The default ``NikolaBot.run_forever`` is a single thread that interleaves
long-poll, initiative checks, reminders, and reflection. While the long-poll
sleeps for ``poll_timeout_seconds`` (25 by default), reminders are not
delivered. The supervisor runs polling in the main loop and offloads
periodic tasks (initiative, reminders, reflection) to a worker thread, so a
just-due reminder fires within ~1 second instead of waiting up to 25s.

SQLite is opened with WAL mode and short-lived per-call connections, so
multiple threads can read while a single writer commits — see
``MemoryStore``.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from ..openai_compat import OpenAICompatError
from .api import TelegramApiError, is_telegram_polling_conflict


if TYPE_CHECKING:
    from .bot import NikolaBot


_DEFAULT_WORKER_TICK_SECONDS = 5.0


class BotRunner:
    """Run the Telegram polling loop and a worker thread for periodic tasks."""

    def __init__(
        self,
        bot: "NikolaBot",
        *,
        worker_tick_seconds: float = _DEFAULT_WORKER_TICK_SECONDS,
    ) -> None:
        self.bot = bot
        self.worker_tick_seconds = worker_tick_seconds
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None

    def start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._stop.clear()
        worker = threading.Thread(
            target=self._worker_loop,
            name="protoagi-telegram-worker",
            daemon=True,
        )
        worker.start()
        self._worker = worker

    def stop(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None:
            worker.join(timeout=timeout)
        self._worker = None

    def run(self) -> None:
        """Run polling on the current thread until interrupted.

        Workers (initiative, reminders, reflection) run on a background
        thread for the lifetime of this call.
        """

        self.start()
        try:
            self._poll_loop()
        finally:
            self.stop()

    # ------------------------------------------------------------------
    # Loops

    def _poll_loop(self) -> None:
        bot = self.bot
        while not self._stop.is_set():
            try:
                bot.poll_once()
            except TelegramApiError as exc:
                if is_telegram_polling_conflict(exc):
                    raise
                print(f"Telegram poll transient error: {exc}", flush=True)
                self._stop.wait(5)
            except (OpenAICompatError, OSError) as exc:
                print(f"Telegram poll transient error: {exc}", flush=True)
                self._stop.wait(5)
            except Exception as exc:
                bot._log_loop_exception(exc)  # type: ignore[attr-defined]
                print(
                    f"Telegram poll unexpected error: {exc}; see {bot.error_log_path}",
                    flush=True,
                )
                self._stop.wait(5)

    def _worker_loop(self) -> None:
        bot = self.bot
        while not self._stop.is_set():
            try:
                bot.maybe_run_initiative()
                bot.maybe_dispatch_reminders()
                bot.maybe_run_reflection()
            except TelegramApiError as exc:
                if is_telegram_polling_conflict(exc):
                    # Polling conflict surfaced from a sticker / send call.
                    # The poll loop will hit the same and raise — bail.
                    return
                print(f"Telegram worker transient error: {exc}", flush=True)
            except (OpenAICompatError, OSError) as exc:
                print(f"Telegram worker transient error: {exc}", flush=True)
            except Exception as exc:
                bot._log_loop_exception(exc)  # type: ignore[attr-defined]
                print(
                    f"Telegram worker unexpected error: {exc}; see {bot.error_log_path}",
                    flush=True,
                )
            self._stop.wait(self.worker_tick_seconds)


__all__ = ["BotRunner"]
