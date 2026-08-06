from __future__ import annotations

from functools import lru_cache
from threading import RLock

from cryptography.fernet import Fernet

from corpusretrieve.config import get_settings


class EphemeralCredentialStore:
    def put(self, job_id: str, credential: str) -> None: ...
    def take(self, job_id: str) -> str | None: ...
    def delete(self, job_id: str) -> None: ...


class MemoryCredentialStore(EphemeralCredentialStore):
    def __init__(self) -> None:
        self._values: dict[str, str] = {}
        self._lock = RLock()

    def put(self, job_id: str, credential: str) -> None:
        with self._lock:
            self._values[job_id] = credential

    def take(self, job_id: str) -> str | None:
        with self._lock:
            return self._values.pop(job_id, None)

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._values.pop(job_id, None)


class RedisCredentialStore(EphemeralCredentialStore):
    def __init__(self) -> None:
        import redis

        settings = get_settings()
        self.client = redis.Redis.from_url(settings.redis_url)
        self.ttl = settings.credential_ttl_seconds
        self.cipher = Fernet(settings.credential_encryption_key.encode())

    def _key(self, job_id: str) -> str:
        return f"corpusretrieve:job:{job_id}:credential"

    def put(self, job_id: str, credential: str) -> None:
        encrypted = self.cipher.encrypt(credential.encode())
        self.client.setex(self._key(job_id), self.ttl, encrypted)

    def take(self, job_id: str) -> str | None:
        key = self._key(job_id)
        pipe = self.client.pipeline()
        pipe.get(key)
        pipe.delete(key)
        encrypted, _ = pipe.execute()
        return self.cipher.decrypt(encrypted).decode() if encrypted else None

    def delete(self, job_id: str) -> None:
        self.client.delete(self._key(job_id))


@lru_cache
def get_credential_store() -> EphemeralCredentialStore:
    if get_settings().job_mode == "rq":
        return RedisCredentialStore()
    return MemoryCredentialStore()
