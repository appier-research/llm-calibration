"""Async semaphore utilities for rate-limited API calls."""

import asyncio
from contextlib import asynccontextmanager
from typing import AsyncIterator


class AsyncSemaphore:
    """
    Wrapper around asyncio.Semaphore with additional utilities.
    
    Provides rate limiting for concurrent API calls to avoid
    overwhelming servers or hitting rate limits.
    """

    def __init__(self, max_concurrent: int = 16):
        """
        Args:
            max_concurrent: Maximum number of concurrent operations.
        """
        self.max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_count = 0
        self._total_acquired = 0

    @asynccontextmanager
    async def acquire(self) -> AsyncIterator[None]:
        """
        Async context manager to acquire and release the semaphore.
        
        Usage:
            async with semaphore.acquire():
                await make_api_call()
        """
        await self._semaphore.acquire()
        self._active_count += 1
        self._total_acquired += 1
        try:
            yield
        finally:
            self._active_count -= 1
            self._semaphore.release()

    @property
    def active_count(self) -> int:
        """Number of currently active operations."""
        return self._active_count

    @property
    def available(self) -> int:
        """Number of available slots."""
        return self.max_concurrent - self._active_count

    @property
    def total_acquired(self) -> int:
        """Total number of times semaphore was acquired."""
        return self._total_acquired

    def stats(self) -> dict:
        """Get semaphore statistics."""
        return {
            "max_concurrent": self.max_concurrent,
            "active_count": self._active_count,
            "available": self.available,
            "total_acquired": self._total_acquired,
        }


async def run_with_semaphore(
    semaphore: AsyncSemaphore,
    coros: list,
    return_exceptions: bool = False,
) -> list:
    """
    Run multiple coroutines with semaphore-based concurrency control.
    
    Args:
        semaphore: The semaphore to use for rate limiting.
        coros: List of coroutines to execute.
        return_exceptions: If True, exceptions are returned instead of raised.
    
    Returns:
        List of results in the same order as input coroutines.
    """
    async def wrapped(coro):
        async with semaphore.acquire():
            return await coro
    
    wrapped_coros = [wrapped(c) for c in coros]
    return await asyncio.gather(*wrapped_coros, return_exceptions=return_exceptions)


