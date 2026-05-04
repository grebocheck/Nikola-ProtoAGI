"""Compatibility facade for Telegram orchestration."""

from __future__ import annotations

from .orchestrator import NikolaBot, build_nikola_bot

__all__ = ["NikolaBot", "build_nikola_bot"]
