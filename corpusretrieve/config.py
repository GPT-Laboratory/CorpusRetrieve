from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

VERSION = "2.0.0"


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class Settings:
    """Immutable runtime configuration resolved from the environment."""

    app_name: str = os.getenv("APP_NAME", "CorpusRetrieve")
    version: str = VERSION
    deployment_mode: str = os.getenv("DEPLOYMENT_MODE", "development")

    # memory: development and tests (lost on restart, held in RAM).
    # local: one machine, persisted under LOCAL_DATA_DIR. s3: production.
    storage_backend: str = os.getenv("STORAGE_BACKEND", "memory")
    local_data_dir: str = os.getenv("LOCAL_DATA_DIR", "corpusretrieve-data")
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "http://localhost:9000")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "corpusretrieve")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "change-this-secret")
    s3_bucket: str = os.getenv("S3_BUCKET", "corpusretrieve")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")

    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    job_mode: str = os.getenv("JOB_MODE", "inline")
    job_ttl_seconds: int = _int("JOB_TTL_SECONDS", 604800)
    artifact_retention_days: int = _int("ARTIFACT_RETENTION_DAYS", 7)

    credential_ttl_seconds: int = _int("CREDENTIAL_TTL_SECONDS", 1800)
    credential_encryption_key: str = os.getenv("CREDENTIAL_ENCRYPTION_KEY", "")

    max_upload_bytes: int = _int("MAX_UPLOAD_BYTES", 100 * 1024 * 1024)
    max_files_per_job: int = _int("MAX_FILES_PER_JOB", 2000)
    max_form_field_bytes: int = _int("MAX_FORM_FIELD_BYTES", 32 * 1024 * 1024)
    embedding_batch_size: int = _int("EMBEDDING_BATCH_SIZE", 256)
    max_corpus_uncompressed_bytes: int = _int(
        "MAX_CORPUS_UNCOMPRESSED_BYTES", 500 * 1024 * 1024
    )

    # "hash" (hashing-v1) stays the default so existing consumers keep working;
    # docs/EMBEDDINGS.md compares hash-v2 and the semantic models
    default_embedder: str = os.getenv("DEFAULT_EMBEDDER", "hash")
    default_vlm_model: str = os.getenv("DEFAULT_VLM_MODEL", "claude-sonnet-4-6")
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "").strip()
    )

    emit_page_previews: bool = _bool("EMIT_PAGE_PREVIEWS", True)
    page_preview_dpi: int = _int("PAGE_PREVIEW_DPI", 110)
    max_page_previews_per_document: int = _int("MAX_PAGE_PREVIEWS_PER_DOCUMENT", 200)

    console_enabled: bool = _bool("CONSOLE_ENABLED", True)
    console_max_projection_points: int = _int("CONSOLE_MAX_PROJECTION_POINTS", 5000)

    delete_inputs_after_processing: bool = _bool("DELETE_INPUTS_AFTER_PROCESSING", True)
    # no cross-origin callers unless configured; the bundled web interface is
    # same-origin and is unaffected
    cors_origins: tuple[str, ...] = tuple(
        x.strip() for x in os.getenv("CORS_ORIGINS", "").split(",") if x.strip()
    )
    # comma-separated keys accepted as "Authorization: Bearer <key>" or
    # "X-API-Key: <key>". Empty means the API is open (development only).
    api_keys: tuple[str, ...] = tuple(
        x.strip() for x in os.getenv("API_KEYS", "").split(",") if x.strip()
    )

    # ---- build-quality and language settings ------------------------------
    # every listed Tesseract language must be installed, or OCR pages count
    # as degraded. TESSERACT_CMD points at the binary when it is not on PATH
    # (the usual case on Windows).
    ocr_langs: str = os.getenv("OCR_LANGS", "eng")
    tesseract_cmd: str = os.getenv("TESSERACT_CMD", "")
    # fail: a degraded build is published but the job is marked failed and the
    # manifest says release_ready=false. warn: the job completes with warnings.
    degraded_policy: str = os.getenv("DEGRADED_POLICY", "fail")
    # When enabled, a build requires every indexed unit to resolve to a
    # publication record (manifest metadata or explicitly labelled PDF fallback).
    require_bibliographic: bool = _bool("REQUIRE_BIBLIOGRAPHIC", False)
    max_vlm_calls: int = _int("MAX_VLM_CALLS", 0)          # 0 = unlimited
    vlm_max_retries: int = _int("VLM_MAX_RETRIES", 3)
    vlm_timeout_seconds: int = _int("VLM_TIMEOUT_SECONDS", 120)
    # figure descriptions stop for the rest of the job after this many
    # consecutive failures, or at once on an authentication, permission or
    # billing error; the remaining figures are counted, not attempted
    vlm_max_consecutive_failures: int = _int("VLM_MAX_CONSECUTIVE_FAILURES", 25)
    # keep dense vectors inside vectors.jsonl as well as dense.npy; turn off to
    # roughly halve the index size when every consumer reads dense.npy
    index_inline_vectors: bool = _bool("INDEX_INLINE_VECTORS", True)

    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parent / "web" / "static"

    def validate(self) -> None:
        if self.deployment_mode not in {"development", "production"}:
            raise ValueError("DEPLOYMENT_MODE must be 'development' or 'production'")
        if self.deployment_mode == "production":
            if self.storage_backend != "s3" or self.job_mode != "rq":
                raise ValueError("production requires STORAGE_BACKEND=s3 and JOB_MODE=rq")
            if not self.api_keys:
                raise ValueError("production requires API_KEYS")
            if "*" in self.cors_origins:
                raise ValueError("production may not use wildcard CORS_ORIGINS")
            if self.s3_secret_key in {"", "change-this-secret", "replace-with-a-long-random-secret"}:
                raise ValueError("production requires a non-default S3_SECRET_KEY")
        if self.storage_backend not in {"memory", "local", "s3"}:
            raise ValueError("STORAGE_BACKEND must be 'memory', 'local' or 's3'")
        if self.job_mode not in {"inline", "rq"}:
            raise ValueError("JOB_MODE must be 'inline' or 'rq'")
        if not valid_embedder_spec(self.default_embedder):
            raise ValueError("DEFAULT_EMBEDDER must be 'hash', 'hash-v2', 'bge-m3', "
                             "'auto' or 'st:<sentence-transformers model id>'")
        if (self.max_vlm_calls < 0 or self.vlm_max_retries < 0
                or self.vlm_timeout_seconds < 1 or self.vlm_max_consecutive_failures < 1):
            raise ValueError("VLM limits must be positive (MAX_VLM_CALLS and "
                             "VLM_MAX_RETRIES may be 0)")
        if self.degraded_policy not in {"fail", "warn"}:
            raise ValueError("DEGRADED_POLICY must be 'fail' or 'warn'")
        if self.max_files_per_job < 1 or self.max_upload_bytes < 1:
            raise ValueError("Upload limits must be positive")
        if self.max_form_field_bytes < 1024:
            raise ValueError("MAX_FORM_FIELD_BYTES must be at least 1024")
        if self.embedding_batch_size < 1:
            raise ValueError("EMBEDDING_BATCH_SIZE must be positive")
        if self.max_corpus_uncompressed_bytes < self.max_upload_bytes:
            raise ValueError(
                "MAX_CORPUS_UNCOMPRESSED_BYTES must be at least MAX_UPLOAD_BYTES"
            )
        if not 40 <= self.page_preview_dpi <= 400:
            raise ValueError("PAGE_PREVIEW_DPI must be between 40 and 400")
        if self.job_mode == "rq" and not self.credential_encryption_key:
            raise ValueError("CREDENTIAL_ENCRYPTION_KEY is required when JOB_MODE=rq")
        if self.job_mode == "rq":
            try:
                Fernet(self.credential_encryption_key.encode())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "CREDENTIAL_ENCRYPTION_KEY must be a valid Fernet key"
                ) from exc


EMBEDDER_KEYWORDS = frozenset({"hash", "hash-v2", "bge-m3", "auto"})


def valid_embedder_spec(spec: str) -> bool:
    """A keyword, or ``st:`` followed by a sentence-transformers model id."""
    return spec in EMBEDDER_KEYWORDS or (spec.startswith("st:") and len(spec) > 3)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
