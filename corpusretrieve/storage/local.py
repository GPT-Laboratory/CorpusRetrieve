"""Artifact store on the local file system.

The memory store keeps every artifact in the server process: a large corpus
costs several gigabytes of RAM, and stopping the server discards the run. The
local store writes each object to a file under ``LOCAL_DATA_DIR``, so a
single-machine deployment keeps its outputs across restarts and serves large
files without holding them in memory.
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from corpusretrieve.config import Settings

_CHUNK = 1024 * 1024


class LocalArtifactStore:
    def __init__(self, settings: Settings) -> None:
        self.root = (Path(settings.local_data_dir) / "objects").resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        relative = PurePosixPath(key)
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError(f"invalid storage key: {key!r}")
        return self.root.joinpath(*relative.parts)

    def ping(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK)

    def put_fileobj(self, key: str, stream: BinaryIO, content_type: str | None = None) -> None:
        target = self._path(key)
        target.parent.mkdir(parents=True, exist_ok=True)
        stream.seek(0)
        # write beside the target and rename, so a reader never sees half a file
        handle, temporary = tempfile.mkstemp(dir=target.parent, prefix=".part-")
        try:
            with os.fdopen(handle, "wb") as out:
                shutil.copyfileobj(stream, out, _CHUNK)
            os.replace(temporary, target)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def put_bytes(self, key: str, data: bytes, content_type: str | None = None) -> None:
        import io

        self.put_fileobj(key, io.BytesIO(data), content_type)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        return path.read_bytes()

    def stream(self, key: str, chunk_size: int = _CHUNK):
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(key)
        with path.open("rb") as handle:
            yield from iter(lambda: handle.read(chunk_size), b"")

    def local_path(self, key: str) -> Path:
        """The file behind a key, for callers that can memory-map it."""
        return self._path(key)

    def list(self, prefix: str) -> list[dict]:
        base = self._path(prefix.rstrip("/")) if prefix.strip("/") else self.root
        search = base if base.is_dir() else base.parent
        out = []
        for path in sorted(search.rglob("*")):
            if not path.is_file() or path.name.startswith(".part-"):
                continue
            key = path.relative_to(self.root).as_posix()
            if key.startswith(prefix):
                out.append({"key": key, "size_bytes": path.stat().st_size,
                            "content_type": None})
        return out

    def delete_prefix(self, prefix: str) -> None:
        base = self._path(prefix.rstrip("/"))
        if base.is_dir():
            shutil.rmtree(base, ignore_errors=True)
        elif base.is_file():
            base.unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()
