"""
In-memory TTL cache for GitHub data and computed summaries.

Key strategy
------------
- Exact key:   "owner/repo:branch"   → cached payload
- Latest key:  "owner/repo:latest"   → points to exact key string

On cache write we store both keys.
On cache read  we resolve the latest pointer first, then fetch the payload.
This avoids scanning internal store keys.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any


class TTLCache:
    """Simple async-safe in-memory cache with per-entry TTL."""

    def __init__(self, default_ttl: int = 600) -> None:
        self._store: dict[str, tuple[float, Any]] = {}
        self._lock = asyncio.Lock()
        self.default_ttl = default_ttl

    @staticmethod
    def make_key(owner: str, repo: str, branch: str) -> str:
        return f"{owner}/{repo}:{branch}"

    @staticmethod
    def make_latest_key(owner: str, repo: str) -> str:
        return f"{owner}/{repo}:latest"

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            expires_at, value = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            return value

    async def resolve_latest(self, owner: str, repo: str) -> Any | None:
        """Resolve the 'latest' pointer and return the cached payload, or None."""
        latest_key = self.make_latest_key(owner, repo)
        exact_key = await self.get(latest_key)
        if exact_key is None:
            return None
        return await self.get(exact_key)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        ttl = ttl if ttl is not None else self.default_ttl
        async with self._lock:
            self._store[key] = (time.monotonic() + ttl, value)

    async def store_with_latest(
        self, owner: str, repo: str, branch: str, value: Any, ttl: int | None = None,
    ) -> None:
        """Store payload under the exact key and set the latest pointer."""
        exact_key = self.make_key(owner, repo, branch)
        latest_key = self.make_latest_key(owner, repo)
        await self.set(exact_key, value, ttl)
        await self.set(latest_key, exact_key, ttl)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()
