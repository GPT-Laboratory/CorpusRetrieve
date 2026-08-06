from __future__ import annotations

import json
import mimetypes
import re
import shutil
import uuid
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import SpooledTemporaryFile
from typing import Annotated, Literal

from fastapi import APIRouter, BackgroundTasks, File, Form, Header, UploadFile
from fastapi.responses import JSONResponse
from starlette.datastructures import Headers

from corpusretrieve.api.common import job_links
from corpusretrieve.api.errors import APIError
from corpusretrieve.api.formats import (
    SUPPORTED_EXTENSIONS,
    SUPPORTED_FORMATS_INLINE,
    available_embedders,
)
from corpusretrieve.api.schemas import JobAccepted, ProblemDetail
from corpusretrieve.config import get_settings
from corpusretrieve.models import InputFile, JobSpec, JobState
from corpusretrieve.queueing import enqueue_job
from corpusretrieve.security import get_credential_store
from corpusretrieve.state import get_job_store
from corpusretrieve.storage import get_artifact_store

router = APIRouter(prefix="/v1")

MAX_CORPUS_NAME_LENGTH = 200
ARCHIVE_ENTRY_MULTIPLIER = 10
SPOOL_THRESHOLD_BYTES = 8 * 1024 * 1024

UPLOAD_ERRORS = {
    400: {"model": ProblemDetail, "description": "Invalid upload or archive."},
    413: {"model": ProblemDetail, "description": "Upload limit exceeded."},
    415: {"model": ProblemDetail, "description": "Unsupported file format."},
    422: {"model": ProblemDetail, "description": "Invalid form field."},
    503: {"model": ProblemDetail, "description": "Queue or storage unavailable."},
}
ACCEPTED_RESPONSE = {
    202: {
        "description": "Job accepted for asynchronous processing.",
        "headers": {
            "Location": {
                "description": "Relative URL of the new job status resource.",
                "schema": {"type": "string"},
            },
            "Retry-After": {
                "description": "Suggested seconds before the first status poll.",
                "schema": {"type": "integer", "example": 1},
            },
            "X-Request-ID": {
                "description": "Request correlation identifier.",
                "schema": {"type": "string"},
            },
        },
    }
}

FILES_DESCRIPTION = (
    "Select one or more files. One supported document creates a document job; "
    "multiple supported documents create a corpus job; one .zip creates an "
    "archive job and may contain nested folders. Do not combine a ZIP with other "
    "files. Supported document extensions: " + SUPPORTED_FORMATS_INLINE
)


def _json_object(raw: str, field: str) -> dict:
    try:
        value = json.loads(raw or "{}")
    except json.JSONDecodeError as exc:
        raise APIError(
            422,
            "invalid-json-field",
            f"{field} must contain a valid JSON object: {exc.msg}.",
            "Invalid JSON form field",
        ) from exc
    if not isinstance(value, dict):
        raise APIError(
            422,
            "invalid-json-field",
            f"{field} must be a JSON object, not {type(value).__name__}.",
            "Invalid JSON form field",
        )
    return value


def _safe_filename(raw: str | None) -> str:
    filename = Path((raw or "upload.bin").replace("\\", "/")).name
    filename = re.sub(r"[\x00-\x1f\x7f]", "", filename).strip()
    return filename or "upload.bin"


def _upload_size(upload: UploadFile) -> int:
    upload.file.seek(0, 2)
    size = upload.file.tell()
    upload.file.seek(0)
    return size


def _config_key(filename: str, seen: dict[str, int]) -> str:
    occurrence = seen.get(filename, 0)
    seen[filename] = occurrence + 1
    return filename if occurrence == 0 else f"{filename}#{occurrence + 1}"


def _document_entry(config: dict, key: str, label: str) -> tuple[str | None, dict]:
    entry = config.get(key, {})
    if not isinstance(entry, dict):
        raise APIError(
            422, "invalid-document-config", f"document_config.{label} must be an object."
        )
    document_id = entry.get("document_id")
    if document_id is not None and (
        not isinstance(document_id, str) or not document_id.strip()
    ):
        raise APIError(
            422,
            "invalid-document-id",
            f"document_config.{label}.document_id must be a non-empty string.",
        )
    metadata = entry.get("metadata", {})
    if not isinstance(metadata, dict):
        raise APIError(
            422,
            "invalid-document-metadata",
            f"document_config.{label}.metadata must be an object.",
        )
    return (document_id.strip() if document_id else None), metadata


def _validate_submission(
    uploads: list[UploadFile],
    corpus_name: str,
    embedder: str | None,
    enable_vlm: bool,
    request_api_key: str | None,
) -> tuple[str, str, str | None]:
    settings = get_settings()
    if not uploads:
        raise APIError(422, "files-required", "At least one document is required.")
    if len(uploads) > settings.max_files_per_job:
        raise APIError(
            413,
            "too-many-files",
            f"A job accepts at most {settings.max_files_per_job} documents.",
        )

    name = corpus_name.strip()
    if not name or len(name) > MAX_CORPUS_NAME_LENGTH:
        raise APIError(
            422,
            "invalid-corpus-name",
            f"corpus_name must contain between 1 and {MAX_CORPUS_NAME_LENGTH} characters.",
        )

    api_key = (request_api_key or "").strip() or None
    if enable_vlm and not (api_key or settings.anthropic_api_key):
        raise APIError(
            422,
            "vlm-api-key-required",
            "Enable VLM only after configuring ANTHROPIC_API_KEY or sending "
            "X-Anthropic-API-Key.",
        )

    selected = embedder or settings.default_embedder
    if selected not in available_embedders():
        raise APIError(
            422,
            "embedder-unavailable",
            f"Available embedders: {', '.join(available_embedders())}.",
        )
    return name, selected, api_key


def _store_uploads(
    job_id: str, uploads: list[UploadFile], document_config: dict
) -> list[InputFile]:
    settings = get_settings()
    store = get_artifact_store()
    inputs: list[InputFile] = []
    seen: dict[str, int] = {}
    for index, upload in enumerate(uploads):
        filename = _safe_filename(upload.filename)
        if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise APIError(
                415,
                "unsupported-file-type",
                f"{filename} is not supported. Accepted extensions: "
                f"{SUPPORTED_FORMATS_INLINE}.",
            )
        size = _upload_size(upload)
        if size == 0:
            raise APIError(422, "empty-file", f"{filename} is empty.")
        if size > settings.max_upload_bytes:
            raise APIError(
                413,
                "file-too-large",
                f"{filename} is {size} bytes; the per-file limit is "
                f"{settings.max_upload_bytes} bytes.",
            )

        key = _config_key(filename, seen)
        document_id, metadata = _document_entry(document_config, key, key)
        storage_key = f"jobs/{job_id}/inputs/{index:04d}-{filename}"
        store.put_fileobj(storage_key, upload.file, upload.content_type)
        inputs.append(
            InputFile(
                filename=filename,
                storage_key=storage_key,
                size_bytes=size,
                content_type=upload.content_type,
                document_id=document_id,
                metadata=metadata,
            )
        )
    return inputs


def _submit(
    background_tasks: BackgroundTasks,
    uploads: list[UploadFile],
    submission_mode: Literal["document", "corpus", "archive"],
    files_received: int,
    corpus_name: str,
    metadata: dict,
    document_config: dict,
    embedder: str | None,
    enable_vlm: bool,
    vlm_model: str | None,
    request_api_key: str | None,
    warnings: list[str] | None = None,
) -> JSONResponse:
    settings = get_settings()
    warnings = warnings or []
    name, selected_embedder, api_key = _validate_submission(
        uploads, corpus_name, embedder, enable_vlm, request_api_key
    )

    job_id = uuid.uuid4().hex[:24]
    store = get_artifact_store()
    try:
        inputs = _store_uploads(job_id, uploads, document_config)
    except APIError:
        store.delete_prefix(f"jobs/{job_id}/")
        raise
    except Exception as exc:  # noqa: BLE001 - reported as a storage failure
        store.delete_prefix(f"jobs/{job_id}/")
        raise APIError(
            503,
            "artifact-storage-unavailable",
            "The uploaded files could not be written to artifact storage.",
        ) from exc

    spec = JobSpec(
        job_id=job_id,
        corpus_name=name,
        files=inputs,
        metadata=metadata,
        embedder=selected_embedder,
        enable_vlm=enable_vlm,
        vlm_model=vlm_model or settings.default_vlm_model,
        submission_mode=submission_mode,
        warnings=warnings,
    )
    state = JobState(job_id=job_id, message="Job accepted", warnings=warnings)

    jobs = get_job_store()
    credentials = get_credential_store()
    try:
        jobs.create(spec, state)
        if api_key:
            credentials.put(job_id, api_key)
        enqueue_job(job_id, background_tasks)
    except Exception as exc:  # noqa: BLE001 - reported as a queue failure
        credentials.delete(job_id)
        store.delete_prefix(f"jobs/{job_id}/")
        jobs.delete(job_id)
        raise APIError(
            503,
            "job-queue-unavailable",
            "The job could not be queued. Retry after checking Redis and the worker.",
            headers={"Retry-After": "5"},
        ) from exc

    accepted = JobAccepted(
        job_id=job_id,
        state="queued",
        submission_mode=submission_mode,
        files_received=files_received,
        files_accepted=len(inputs),
        warnings=warnings,
        links=job_links(job_id),
    )
    return JSONResponse(
        accepted.model_dump(),
        status_code=202,
        headers={"Location": accepted.links.status, "Retry-After": "1"},
    )


@router.post(
    "/jobs",
    response_model=JobAccepted,
    status_code=202,
    tags=["Upload"],
    summary="Upload documents or one ZIP archive",
    description=(
        "Use the `files` picker for every submission. Select one document, "
        "multiple documents, or exactly one ZIP containing a folder or corpus. "
        "CorpusRetrieve determines the processing mode automatically."
    ),
    operation_id="upload_files",
    responses={**ACCEPTED_RESPONSE, **UPLOAD_ERRORS},
)
async def upload_files(
    background_tasks: BackgroundTasks,
    files: Annotated[list[UploadFile], File(description=FILES_DESCRIPTION, min_length=1)],
    corpus_name: Annotated[
        str, Form(description="Logical corpus name used in IDs and manifests.")
    ] = "corpus",
    metadata: Annotated[
        str, Form(description='Corpus metadata as JSON, for example {"owner":"team"}.')
    ] = "{}",
    document_config: Annotated[
        str,
        Form(
            description=(
                "Optional JSON object keyed by filename or ZIP path. Each value "
                'may contain document_id and metadata. Example: '
                '{"a.pdf":{"document_id":"a","metadata":{"language":"en"}}}'
            )
        ),
    ] = "{}",
    embedder: Annotated[
        Literal["hash", "bge-m3"] | None,
        Form(description="hash is always available; bge-m3 requires the optional model."),
    ] = None,
    enable_vlm: Annotated[
        bool, Form(description="Use Anthropic VLM enrichment for extracted figures.")
    ] = False,
    vlm_model: Annotated[
        str | None, Form(description="Optional Anthropic model override.")
    ] = None,
    anthropic_api_key: Annotated[
        str | None,
        Header(
            alias="X-Anthropic-API-Key",
            description="Optional per-request override. Never persisted in artifacts.",
        ),
    ] = None,
):
    parsed_metadata = _json_object(metadata, "metadata")
    parsed_config = _json_object(document_config, "document_config")
    archives = [
        upload
        for upload in files
        if Path(_safe_filename(upload.filename)).suffix.lower() == ".zip"
    ]

    if archives:
        if len(files) != 1:
            raise APIError(
                422,
                "zip-must-be-uploaded-alone",
                "Upload exactly one ZIP without additional files.",
            )
        uploads, normalized_config, warnings = expand_archive(archives[0], parsed_config)
        try:
            return _submit(
                background_tasks,
                uploads,
                "archive",
                1,
                corpus_name,
                parsed_metadata,
                normalized_config,
                embedder,
                enable_vlm,
                vlm_model,
                anthropic_api_key,
                warnings,
            )
        finally:
            for upload in uploads:
                upload.file.close()

    return _submit(
        background_tasks,
        files,
        "document" if len(files) == 1 else "corpus",
        len(files),
        corpus_name,
        parsed_metadata,
        parsed_config,
        embedder,
        enable_vlm,
        vlm_model,
        anthropic_api_key,
    )


def _reject_unsafe_entry(entry: zipfile.ZipInfo) -> PurePosixPath:
    member = PurePosixPath(entry.filename.replace("\\", "/"))
    if member.is_absolute() or ".." in member.parts:
        raise APIError(
            400, "unsafe-archive-path", f"Unsafe archive path rejected: {entry.filename}."
        )
    if entry.flag_bits & 0x1:
        raise APIError(
            400,
            "encrypted-archive-entry",
            f"Encrypted archive entries are not supported: {entry.filename}.",
        )
    return member


def expand_archive(
    archive_upload: UploadFile, user_config: dict
) -> tuple[list[UploadFile], dict, list[str]]:
    """Expand one ZIP into in-memory uploads, enforcing every safety limit."""
    settings = get_settings()
    archive_size = _upload_size(archive_upload)
    if archive_size == 0:
        raise APIError(422, "empty-archive", "The ZIP archive is empty.")
    if archive_size > settings.max_upload_bytes:
        raise APIError(
            413,
            "archive-too-large",
            f"The ZIP is {archive_size} bytes; the upload limit is "
            f"{settings.max_upload_bytes} bytes.",
        )

    generated: list[UploadFile] = []
    normalized_config: dict = {}
    skipped: list[str] = []
    seen: dict[str, int] = {}
    total_uncompressed = 0

    try:
        archive_upload.file.seek(0)
        with zipfile.ZipFile(archive_upload.file) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            if len(entries) > settings.max_files_per_job * ARCHIVE_ENTRY_MULTIPLIER:
                raise APIError(
                    413,
                    "archive-entry-limit",
                    "The archive contains too many entries to inspect safely.",
                )
            for entry in entries:
                member = _reject_unsafe_entry(entry)
                raw_path = str(member)
                filename = _safe_filename(member.name)
                if (
                    Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS
                    or entry.file_size == 0
                ):
                    skipped.append(raw_path)
                    continue
                if entry.file_size > settings.max_upload_bytes:
                    raise APIError(
                        413,
                        "archive-file-too-large",
                        f"{raw_path} exceeds the per-file uncompressed limit.",
                    )
                total_uncompressed += entry.file_size
                if total_uncompressed > settings.max_corpus_uncompressed_bytes:
                    raise APIError(
                        413,
                        "archive-uncompressed-limit",
                        "The extracted corpus would exceed the configured uncompressed limit.",
                    )
                if len(generated) >= settings.max_files_per_job:
                    raise APIError(
                        413,
                        "too-many-files",
                        f"A corpus can contain at most {settings.max_files_per_job} documents.",
                    )

                key = _config_key(filename, seen)
                supplied = user_config.get(raw_path, user_config.get(key, {}))
                if not isinstance(supplied, dict):
                    raise APIError(
                        422,
                        "invalid-document-config",
                        f"document_config entry for {raw_path} must be an object.",
                    )
                supplied_metadata = supplied.get("metadata", {})
                if not isinstance(supplied_metadata, dict):
                    raise APIError(
                        422,
                        "invalid-document-metadata",
                        f"metadata for {raw_path} must be an object.",
                    )
                normalized_config[key] = {
                    **supplied,
                    "metadata": {"archive_path": raw_path, **supplied_metadata},
                }

                spooled = SpooledTemporaryFile(  # noqa: SIM115 - closed by the caller
                    max_size=SPOOL_THRESHOLD_BYTES, mode="w+b"
                )
                with archive.open(entry) as source:
                    shutil.copyfileobj(source, spooled, length=1024 * 1024)
                spooled.seek(0)
                media_type = (
                    mimetypes.guess_type(filename)[0] or "application/octet-stream"
                )
                generated.append(
                    UploadFile(
                        file=spooled,
                        filename=filename,
                        size=entry.file_size,
                        headers=Headers({"content-type": media_type}),
                    )
                )
    except APIError:
        for upload in generated:
            upload.file.close()
        raise
    except (zipfile.BadZipFile, OSError) as exc:
        for upload in generated:
            upload.file.close()
        raise APIError(
            400,
            "invalid-zip-archive",
            "The uploaded file is not a readable ZIP archive.",
        ) from exc

    if not generated:
        raise APIError(
            422,
            "archive-has-no-supported-documents",
            "The archive contains no supported documents. Accepted extensions: "
            f"{SUPPORTED_FORMATS_INLINE}.",
        )

    warnings = []
    if skipped:
        preview = ", ".join(skipped[:10])
        remainder = f" and {len(skipped) - 10} more" if len(skipped) > 10 else ""
        warnings.append(
            f"Skipped unsupported or empty archive entries: {preview}{remainder}."
        )
    return generated, normalized_config, warnings
