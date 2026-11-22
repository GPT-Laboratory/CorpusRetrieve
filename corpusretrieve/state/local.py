"""Job store on the local file system.

With the memory store, a server restart forgets every job, so the console has
nothing to show for a build that finished overnight. The local store keeps one
JSON file per job specification and state under ``LOCAL_DATA_DIR/jobs``.
"""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from threading import RLock
from time import time

from corpusretrieve.config import Settings
from corpusretrieve.models import JobSpec, JobState


class LocalJobStore:
    def __init__(self, settings: Settings) -> None:
        self.root = (Path(settings.local_data_dir) / "jobs").resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._recover_interrupted()

    def _recover_interrupted(self) -> None:
        """Mark jobs that were running when the server stopped as failed.

        Inline jobs run inside the server process, so a job still queued or
        processing at start-up can never finish; without this it would show
        as running forever.
        """
        for state in self.list_states(limit=10_000):
            if state.state in {"queued", "processing"}:
                self.update_state(
                    state.job_id, state="failed", stage="failed",
                    message="Interrupted: the server stopped while this job was running",
                    error="JobInterrupted: resubmit the files to build them again")

    def _file(self, job_id: str, kind: str) -> Path:
        if not job_id.isalnum():
            raise ValueError(f"invalid job id: {job_id!r}")
        return self.root / job_id / f"{kind}.json"

    def _write(self, path: Path, value: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=".part-")
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            json.dump(value, out, ensure_ascii=False)
        os.replace(temporary, path)

    def _read(self, path: Path) -> dict | None:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def ping(self) -> bool:
        return self.root.is_dir() and os.access(self.root, os.W_OK)

    def create(self, spec: JobSpec, state: JobState) -> None:
        with self._lock:
            self._write(self._file(spec.job_id, "spec"), spec.to_dict())
            self._write(self._file(spec.job_id, "state"), state.to_dict())

    def get_spec(self, job_id: str) -> JobSpec | None:
        value = self._read(self._file(job_id, "spec"))
        return JobSpec.from_dict(value) if value else None

    def get_state(self, job_id: str) -> JobState | None:
        value = self._read(self._file(job_id, "state"))
        return JobState.from_dict(value) if value else None

    def update_state(self, job_id: str, **updates) -> JobState:
        with self._lock:
            state = self.get_state(job_id)
            if state is None:
                raise KeyError(job_id)
            for key, value in updates.items():
                setattr(state, key, value)
            state.updated_at = time()
            self._write(self._file(job_id, "state"), state.to_dict())
            return state

    def delete(self, job_id: str) -> None:
        with self._lock:
            for kind in ("spec", "state"):
                self._file(job_id, kind).unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                (self.root / job_id).rmdir()

    def list_states(self, limit: int = 50) -> list[JobState]:
        states = [s for s in (self.get_state(p.name) for p in self.root.iterdir()
                              if p.is_dir() and p.name.isalnum()) if s is not None]
        return sorted(states, key=lambda s: s.created_at, reverse=True)[:limit]
