from __future__ import annotations

from threading import RLock
from typing import BinaryIO


class MemoryArtifactStore:
    def __init__(self) -> None:
        self._objects: dict[str, tuple[bytes, str | None]] = {}
        self._lock = RLock()

    def ping(self) -> bool:
        return True

    def put_fileobj(self, key: str, stream: BinaryIO, content_type: str | None = None) -> None:
        stream.seek(0)
        self.put_bytes(key, stream.read(), content_type)

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        with self._lock:
            self._objects[key] = (bytes(data), content_type)

    def get_bytes(self, key: str) -> bytes:
        with self._lock:
            if key not in self._objects:
                raise FileNotFoundError(key)
            return self._objects[key][0]

    def stream(self, key: str, chunk_size: int = 1024 * 1024):
        data = self.get_bytes(key)
        for start in range(0, len(data), chunk_size):
            yield data[start:start + chunk_size]

    def list(self, prefix: str) -> list[dict]:
        with self._lock:
            return [
                {"key": key, "size_bytes": len(data), "content_type": content_type}
                for key, (data, content_type) in sorted(self._objects.items())
                if key.startswith(prefix)
            ]

    def delete_prefix(self, prefix: str) -> None:
        with self._lock:
            for key in [key for key in self._objects if key.startswith(prefix)]:
                del self._objects[key]

    def exists(self, key: str) -> bool:
        with self._lock:
            return key in self._objects
