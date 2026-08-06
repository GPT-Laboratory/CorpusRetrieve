from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import time
from typing import Any

STAGES: tuple[str, ...] = ("extract", "enrich", "chunk", "embed", "graph", "package")


@dataclass
class InputFile:
    filename: str
    storage_key: str
    size_bytes: int
    content_type: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class JobSpec:
    job_id: str
    corpus_name: str
    files: list[InputFile]
    metadata: dict[str, Any] = field(default_factory=dict)
    embedder: str = "hash"
    enable_vlm: bool = False
    vlm_model: str = "claude-sonnet-4-6"
    submission_mode: str = "corpus"
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JobSpec:
        return cls(
            **{k: v for k, v in value.items() if k != "files"},
            files=[InputFile(**item) for item in value["files"]],
        )


@dataclass
class StageRecord:
    """Observable outcome of one pipeline stage."""

    name: str
    index: int
    status: str = "pending"
    detail: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int | None = None


@dataclass
class JobState:
    job_id: str
    state: str = "queued"
    stage: str = "queued"
    progress: int = 0
    message: str = ""
    error: str | None = None
    manifest_key: str | None = None
    bundle_key: str | None = None
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    stages: list[dict[str, Any]] = field(
        default_factory=lambda: [
            asdict(StageRecord(name=name, index=index))
            for index, name in enumerate(STAGES)
        ]
    )
    warnings: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time)
    updated_at: float = field(default_factory=time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> JobState:
        return cls(**value)
