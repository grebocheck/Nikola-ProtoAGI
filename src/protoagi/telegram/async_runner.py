from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Callable

from ..openai_compat import OpenAICompatError
from .api import TelegramApiError, is_telegram_polling_conflict
from .constants import OFFSET_KEY


if TYPE_CHECKING:
    from .bot import NikolaBot


class AsyncBotRunner:
    """Async supervisor for concurrent Telegram update handling.

    The bot implementation stays synchronous; this runner uses
    ``asyncio.to_thread`` around the blocking Telegram and LLM work, then
    bounds concurrent update handling with a semaphore.
    """

    def __init__(
        self,
        bot: "NikolaBot",
        *,
        worker_tick_seconds: float = 5.0,
        max_concurrent_updates: int = 2,
    ) -> None:
        self.bot = bot
        self.worker_tick_seconds = worker_tick_seconds
        self.max_concurrent_updates = max(1, max_concurrent_updates)
        self._stop: asyncio.Event | None = None
        self._semaphore = asyncio.Semaphore(self.max_concurrent_updates)

    def stop(self) -> None:
        if self._stop is not None:
            self._stop.set()

    async def run(self) -> None:
        self._stop = asyncio.Event()
        async with asyncio.TaskGroup() as group:
            group.create_task(self._poll_loop())
            group.create_task(self._periodic_loop("initiative", self.bot.maybe_run_initiative))
            group.create_task(self._periodic_loop("reminders", self.bot.maybe_dispatch_reminders))
            group.create_task(self._periodic_loop("reflection", self.bot.maybe_run_reflection))

    async def poll_once(self) -> int:
        offset_text = await asyncio.to_thread(self.bot.memory.get_kv, OFFSET_KEY)
        offset = int(offset_text) if offset_text else None
        updates = await asyncio.to_thread(
            self.bot.telegram.get_updates,
            offset=offset,
            timeout_seconds=self.bot.telegram_config.poll_timeout_seconds,
            allowed_updates=["message", "edited_message", "message_reaction"],
        )
        if not updates:
            return 0
        tasks = [asyncio.create_task(self._process_update(update)) for update in updates]
        # ``return_exceptions=True`` keeps a single failure from cancelling
        # the rest. Offsets only acknowledge the contiguous successful prefix:
        # advancing past a failed update would ask Telegram to drop it.
        results = await asyncio.gather(*tasks, return_exceptions=True)
        processed = 0
        successful_ids: list[int] = []
        failed_ids: list[int] = []
        for update, item in zip(updates, results):
            update_id = int(update.get("update_id", 0))
            if isinstance(item, BaseException):
                self.bot._log_loop_exception(item)  # type: ignore[attr-defined]
                failed_ids.append(update_id)
                continue
            successful_ids.append(update_id)
            if item:
                processed += 1
        if failed_ids:
            next_offset = min(failed_ids)
            acknowledged = [item for item in successful_ids if item < next_offset]
            if acknowledged:
                await asyncio.to_thread(
                    self.bot.memory.set_kv,
                    OFFSET_KEY,
                    str(max(acknowledged) + 1),
                )
        elif successful_ids:
            await asyncio.to_thread(
                self.bot.memory.set_kv,
                OFFSET_KEY,
                str(max(successful_ids) + 1),
            )
        return processed

    async def _process_update(self, update: dict) -> bool:
        async with self._semaphore:
            return bool(await asyncio.to_thread(self.bot.process_update, update))

    async def _poll_loop(self) -> None:
        while not self._stopped():
            try:
                await self.poll_once()
            except TelegramApiError as exc:
                if is_telegram_polling_conflict(exc):
                    raise
                print(f"Telegram async poll transient error: {exc}", flush=True)
                await self._sleep(5)
            except (OpenAICompatError, OSError) as exc:
                print(f"Telegram async poll transient error: {exc}", flush=True)
                await self._sleep(5)
            except Exception as exc:
                self.bot._log_loop_exception(exc)  # type: ignore[attr-defined]
                print(
                    f"Telegram async poll unexpected error: {exc}; see {self.bot.error_log_path}",
                    flush=True,
                )
                await self._sleep(5)

    async def _periodic_loop(self, name: str, callback: Callable[[], object]) -> None:
        while not self._stopped():
            try:
                await asyncio.to_thread(callback)
            except TelegramApiError as exc:
                if is_telegram_polling_conflict(exc):
                    return
                print(f"Telegram async {name} transient error: {exc}", flush=True)
            except (OpenAICompatError, OSError) as exc:
                print(f"Telegram async {name} transient error: {exc}", flush=True)
            except Exception as exc:
                self.bot._log_loop_exception(exc)  # type: ignore[attr-defined]
                print(
                    f"Telegram async {name} unexpected error: {exc}; see {self.bot.error_log_path}",
                    flush=True,
                )
            await self._sleep(self.worker_tick_seconds)

    async def _sleep(self, seconds: float) -> None:
        if self._stop is None:
            await asyncio.sleep(seconds)
            return
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            return

    def _stopped(self) -> bool:
        return bool(self._stop is not None and self._stop.is_set())


__all__ = ["AsyncBotRunner"]
