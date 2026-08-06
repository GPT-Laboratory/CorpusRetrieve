from __future__ import annotations

from dataclasses import asdict
from time import monotonic
from typing import Any, Protocol

from corpusretrieve.models import STAGES, StageRecord


class ProgressSink(Protocol):
    """Receives a pipeline progress update."""

    def __call__(
        self,
        stage: str,
        percent: int,
        message: str,
        stages: list[dict[str, Any]],
    ) -> None: ...


class StageTracker:
    """Tracks the status, timing, and detail of every pipeline stage.

    The pipeline reports through this object instead of writing job state
    directly, which keeps persistence concerns out of the processing code.
    """

    def __init__(self, sink: ProgressSink) -> None:
        self._sink = sink
        self._records = {
            name: StageRecord(name=name, index=index)
            for index, name in enumerate(STAGES)
        }
        self._started: dict[str, float] = {}
        self._current = STAGES[0]

    def snapshot(self) -> list[dict[str, Any]]:
        return [asdict(self._records[name]) for name in STAGES]

    def start(self, *names: str) -> None:
        for name in names:
            self._records[name].status = "running"
            self._started.setdefault(name, monotonic())
        self._current = names[0]

    def update(self, name: str, **detail: Any) -> None:
        self._records[name].detail.update(detail)

    def finish(self, *names: str) -> None:
        for name in names:
            record = self._records[name]
            record.status = "done"
            started = self._started.get(name)
            if started is not None:
                record.elapsed_ms = int((monotonic() - started) * 1000)

    def fail(self, name: str, reason: str) -> None:
        record = self._records[name]
        record.status = "error"
        record.detail["error"] = reason

    def report(self, percent: int, message: str, stage: str | None = None) -> None:
        self._current = stage or self._current
        self._sink(self._current, percent, message, self.snapshot())
