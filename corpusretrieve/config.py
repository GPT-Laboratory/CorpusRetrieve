from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from cryptography.fernet import Fernet
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

VERSION = "1.0.0"


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

    storage_backend: str = os.getenv("STORAGE_BACKEND", "memory")
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
    max_files_per_job: int = _int("MAX_FILES_PER_JOB", 100)
    max_corpus_uncompressed_bytes: int = _int(
        "MAX_CORPUS_UNCOMPRESSED_BYTES", 500 * 1024 * 1024
    )

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
    cors_origins: tuple[str, ...] = tuple(
        x.strip() for x in os.getenv("CORS_ORIGINS", "*").split(",") if x.strip()
    )

    @property
    def static_dir(self) -> Path:
        return Path(__file__).resolve().parent / "web" / "static"

    def validate(self) -> None:
        if self.storage_backend not in {"memory", "s3"}:
            raise ValueError("STORAGE_BACKEND must be 'memory' or 's3'")
        if self.job_mode not in {"inline", "rq"}:
            raise ValueError("JOB_MODE must be 'inline' or 'rq'")
        if self.default_embedder not in {"hash", "bge-m3"}:
            raise ValueError("DEFAULT_EMBEDDER must be 'hash' or 'bge-m3'")
        if self.max_files_per_job < 1 or self.max_upload_bytes < 1:
            raise ValueError("Upload limits must be positive")
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


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.validate()
    return settings
