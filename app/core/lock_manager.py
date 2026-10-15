import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_LOCK_TIMEOUT = float(os.getenv("THREAD_LOCK_TIMEOUT", "5.0"))


class ThreadLockManager:
    """
    Manages per-thread (chat_id) asyncio locks with reference counting
    to prevent race conditions, checkpoint write collisions, and memory leaks.
    """

    def __init__(self, default_timeout: Optional[float] = None):
        self._locks: dict[str, asyncio.Lock] = {}
        self._ref_counts: dict[str, int] = {}
        self._meta_lock = asyncio.Lock()
        self._default_timeout = default_timeout

    @property
    def default_timeout(self) -> float:
        if self._default_timeout is not None:
            return self._default_timeout
        return float(os.getenv("THREAD_LOCK_TIMEOUT", "5.0"))

    async def acquire(self, key: str, timeout: Optional[float] = None) -> bool:
        """
        Acquire an exclusive lock for the given key (e.g. chat_id).
        Returns True if acquired, False if timeout exceeded.
        """
        effective_timeout = timeout if timeout is not None else self.default_timeout
        async with self._meta_lock:
            if key not in self._locks:
                self._locks[key] = asyncio.Lock()
                self._ref_counts[key] = 0
            self._ref_counts[key] += 1
            lock = self._locks[key]

        try:
            if effective_timeout is not None:
                await asyncio.wait_for(lock.acquire(), timeout=effective_timeout)
            else:
                await lock.acquire()
            return True
        except (asyncio.TimeoutError, Exception) as e:
            logger.warning("Could not acquire thread lock for key=%s (timeout=%s): %s", key, effective_timeout, e)
            async with self._meta_lock:
                self._ref_counts[key] -= 1
                if self._ref_counts[key] <= 0:
                    self._locks.pop(key, None)
                    self._ref_counts.pop(key, None)
            return False

    async def release(self, key: str) -> None:
        """
        Release the exclusive lock for the given key and clean up if no other
        requests are waiting.
        """
        async with self._meta_lock:
            lock = self._locks.get(key)
            if lock and lock.locked():
                lock.release()
            if key in self._ref_counts:
                self._ref_counts[key] -= 1
                if self._ref_counts[key] <= 0:
                    self._locks.pop(key, None)
                    self._ref_counts.pop(key, None)

    def is_locked(self, key: str) -> bool:
        """Check if a key is currently held by any task."""
        lock = self._locks.get(key)
        return bool(lock and lock.locked())

    @asynccontextmanager
    async def lock(self, key: str, timeout: Optional[float] = DEFAULT_LOCK_TIMEOUT):
        """
        Context manager for acquiring and safely releasing a lock on a key.
        Yields True if acquired, or raises TimeoutError if not acquired.
        """
        acquired = await self.acquire(key, timeout=timeout)
        if not acquired:
            raise TimeoutError(f"Could not acquire thread lock for key '{key}' within {timeout}s.")
        try:
            yield
        finally:
            await self.release(key)


# Global singleton instance
thread_lock_manager = ThreadLockManager()
