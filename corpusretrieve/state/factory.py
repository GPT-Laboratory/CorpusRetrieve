from __future__ import annotations

from functools import lru_cache

from corpusretrieve.config import get_settings
from corpusretrieve.state.base import JobStore
from corpusretrieve.state.memory import MemoryJobStore


@lru_cache
def get_job_store() -> JobStore:
    """Resolve the configured job store.

    The Redis adapter is imported lazily so a memory-backed development or
    test run does not require the Redis client.
    """
    settings = get_settings()
    if settings.job_mode == "rq":
        from corpusretrieve.state.redis import RedisJobStore

        return RedisJobStore(settings)
    return MemoryJobStore()
