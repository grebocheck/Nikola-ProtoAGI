"""Compatibility facade for the SQLite memory store.

The implementation lives in :mod:`protoagi.storage.memory`; this module keeps
the long-standing ``protoagi.memory`` import path stable for callers and tests.
"""

from __future__ import annotations

from .storage.memory import *  # noqa: F401,F403
