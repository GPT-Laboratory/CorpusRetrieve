from __future__ import annotations

from functools import lru_cache

from corpusretrieve.config import get_settings
from corpusretrieve.storage.base import ArtifactStore
from corpusretrieve.storage.memory import MemoryArtifactStore


@lru_cache
def get_artifact_store() -> ArtifactStore:
    """Resolve the configured artifact store.

    The S3 adapter is imported lazily so a memory-backed development or test
    run does not require boto3 or a reachable object store.
    """
    settings = get_settings()
    if settings.storage_backend == "s3":
        from corpusretrieve.storage.s3 import S3ArtifactStore

        return S3ArtifactStore(settings)
    return MemoryArtifactStore()
