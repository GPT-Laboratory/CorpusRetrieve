from __future__ import annotations

import json
from time import time

import redis

from corpusretrieve.config import Settings
from corpusretrieve.models import JobSpec, JobState


class RedisJobStore:
    def __init__(self, settings: Settings) -> None:
        self.client = redis.Redis.from_url(settings.redis_url, decode_responses=True)
        self.ttl = settings.job_ttl_seconds

    def ping(self) -> bool:
        return bool(self.client.ping())

    def _spec_key(self, job_id: str) -> str:
        return f"corpusretrieve:job:{job_id}:spec"

    def _state_key(self, job_id: str) -> str:
        return f"corpusretrieve:job:{job_id}:state"

    def create(self, spec: JobSpec, state: JobState) -> None:
        pipe = self.client.pipeline()
        pipe.setex(self._spec_key(spec.job_id), self.ttl, json.dumps(spec.to_dict()))
        pipe.setex(self._state_key(spec.job_id), self.ttl, json.dumps(state.to_dict()))
        pipe.execute()

    def get_spec(self, job_id: str) -> JobSpec | None:
        value = self.client.get(self._spec_key(job_id))
        return JobSpec.from_dict(json.loads(value)) if value else None

    def get_state(self, job_id: str) -> JobState | None:
        value = self.client.get(self._state_key(job_id))
        return JobState.from_dict(json.loads(value)) if value else None

    def update_state(self, job_id: str, **updates) -> JobState:
        state = self.get_state(job_id)
        if state is None:
            raise KeyError(job_id)
        for key, value in updates.items():
            setattr(state, key, value)
        state.updated_at = time()
        self.client.setex(self._state_key(job_id), self.ttl, json.dumps(state.to_dict()))
        return state

    def delete(self, job_id: str) -> None:
        self.client.delete(self._spec_key(job_id), self._state_key(job_id))
