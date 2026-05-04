"""Compatibility facade for the local admin server."""

from __future__ import annotations

from .admin_server import _stats, make_handler, serve

__all__ = ["_stats", "make_handler", "serve"]
