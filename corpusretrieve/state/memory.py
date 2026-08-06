from __future__ import annotations

from threading import RLock
from time import time

from corpusretrieve.models import JobSpec, JobState


class MemoryJobStore:
    def __init__(self) -> None:
        self._specs: dict[str, JobSpec] = {}
        self._states: dict[str, JobState] = {}
        self._lock = RLock()

    def ping(self) -> bool:
        return True

    def create(self, spec: JobSpec, state: JobState) -> None:
        with self._lock:
            self._specs[spec.job_id] = spec
            self._states[spec.job_id] = state

    def get_spec(self, job_id: str) -> JobSpec | None:
        with self._lock:
            return self._specs.get(job_id)

    def get_state(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._states.get(job_id)

    def update_state(self, job_id: str, **updates) -> JobState:
        with self._lock:
            state = self._states[job_id]
            for key, value in updates.items():
                setattr(state, key, value)
            state.updated_at = time()
            return state

    def delete(self, job_id: str) -> None:
        with self._lock:
            self._specs.pop(job_id, None)
            self._states.pop(job_id, None)
