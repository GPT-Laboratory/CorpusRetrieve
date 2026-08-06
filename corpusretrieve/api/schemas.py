from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class JobLinks(BaseModel):
    status: str
    events: str
    manifest: str
    documents: str
    corpus_chunks: str
    index: str
    graph: str
    artifacts: str
    bundle: str


class JobAccepted(BaseModel):
    job_id: str = Field(description="Stable identifier for this processing job.")
    state: Literal["queued"]
    submission_mode: Literal["document", "corpus", "archive"]
    files_received: int
    files_accepted: int
    warnings: list[str] = Field(default_factory=list)
    links: JobLinks


class UploadChoice(BaseModel):
    method: Literal["POST"]
    endpoint: str
    file_field: str


class APIGuide(BaseModel):
    service: str
    version: str
    how_to_upload: dict[str, UploadChoice]
    next_steps: list[str]
    documentation: dict[str, str]
    capabilities: str


class StageStatus(BaseModel):
    name: str
    index: int
    status: Literal["pending", "running", "done", "error"]
    detail: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int | None = None


class JobStatus(BaseModel):
    job_id: str
    state: Literal["queued", "processing", "completed", "failed"]
    stage: str
    progress: int = Field(ge=0, le=100)
    message: str
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    stages: list[StageStatus] = Field(default_factory=list)
    created_at: float
    updated_at: float
    outputs_ready: bool
    links: JobLinks


class ArtifactItem(BaseModel):
    path: str
    size_bytes: int
    media_type: str
    download: str


class ArtifactList(BaseModel):
    job_id: str
    state: str
    artifacts: list[ArtifactItem]


class DocumentLinks(BaseModel):
    extraction: str
    chunks: str
    assets: str


class DocumentSummary(BaseModel):
    doc_id: str
    source: str
    sha256: str
    chunks: int
    metadata: dict[str, Any]
    links: DocumentLinks


class DocumentList(BaseModel):
    job_id: str
    documents: list[DocumentSummary]


class AssetList(BaseModel):
    job_id: str
    document_id: str
    assets: list[ArtifactItem]


class IndexFiles(BaseModel):
    vectors: str
    sparse_encoder: str


class IndexDescription(BaseModel):
    job_id: str
    format: str
    schema_version: str
    model: str
    dimension: int
    metric: str
    dense_field: str
    sparse_field: str
    records: int
    documents: int
    files: IndexFiles


class CapabilityInputs(BaseModel):
    upload_endpoints: dict[str, str]
    formats: dict[str, str]
    max_files_per_job: int
    max_bytes_per_file: int
    max_archive_uncompressed_bytes: int
    folder_upload: str


class CapabilityVLM(BaseModel):
    provider: str
    api_key_configured: bool
    request_override_header: str
    default_model: str


class CapabilitiesResponse(BaseModel):
    service: str
    version: str
    inputs: CapabilityInputs
    embedders: list[str]
    vlm: CapabilityVLM
    stages: list[str]
    outputs: dict[str, str]


class ProblemField(BaseModel):
    field: str
    message: str
    code: str


class ProblemDetail(BaseModel):
    type: str = Field(examples=["urn:corpusretrieve:problem:unsupported-file-type"])
    title: str
    status: int
    detail: str
    instance: str
    code: str
    request_id: str
    errors: list[ProblemField] | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "unavailable"]
    service: str
    version: str
    storage: str
    job_mode: str
    embedders: list[str]
    vlm_api_key_configured: bool
    console_enabled: bool
    checks: dict[str, bool]


class DeleteResponse(BaseModel):
    message: str
