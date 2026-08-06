from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from corpusretrieve.api.common import (
    artifact_download,
    artifact_json,
    get_job_state,
    job_status_payload,
)
from corpusretrieve.api.formats import available_embedders
from corpusretrieve.api.schemas import (
    APIGuide,
    ArtifactList,
    DeleteResponse,
    HealthResponse,
    JobStatus,
    ProblemDetail,
)
from corpusretrieve.config import get_settings
from corpusretrieve.security import get_credential_store
from corpusretrieve.state import get_job_store
from corpusretrieve.storage import get_artifact_store

router = APIRouter(prefix="/v1")
logger = logging.getLogger(__name__)

NOT_FOUND = {404: {"model": ProblemDetail, "description": "Job or resource not found."}}
ARTIFACT_ERRORS = {
    **NOT_FOUND,
    409: {"model": ProblemDetail, "description": "Output is not ready."},
    410: {"model": ProblemDetail, "description": "Output retention has expired."},
}


@router.get(
    "",
    response_model=APIGuide,
    tags=["Service"],
    summary="Start here: API guide",
    operation_id="api_guide",
)
def api_guide():
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.version,
        "how_to_upload": {
            "all_uploads": {
                "method": "POST",
                "endpoint": "/v1/jobs",
                "file_field": "files",
            },
        },
        "next_steps": [
            "Use the returned links.status URL until state is completed.",
            "Follow the returned links for documents, index, graph, and bundle.",
        ],
        "documentation": {"swagger": "/docs", "redoc": "/redoc"},
        "capabilities": "/v1/capabilities",
    }


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    tags=["Jobs"],
    summary="Get job status and output links",
    operation_id="get_job_status",
    responses=NOT_FOUND,
)
def get_job(job_id: str):
    return job_status_payload(get_job_state(job_id))


@router.get(
    "/jobs/{job_id}/manifest",
    tags=["Artifacts"],
    summary="Read the complete output manifest",
    operation_id="get_manifest",
    response_model=dict[str, Any],
    responses=ARTIFACT_ERRORS,
)
def get_manifest(job_id: str):
    return JSONResponse(artifact_json(get_job_state(job_id), "manifest.json"))


@router.get(
    "/jobs/{job_id}/artifacts",
    response_model=ArtifactList,
    tags=["Artifacts"],
    summary="List every generated output file",
    operation_id="list_artifacts",
    responses=NOT_FOUND,
)
def list_artifacts(job_id: str):
    state = get_job_state(job_id)
    return {
        "job_id": job_id,
        "state": state.state,
        "artifacts": [
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "media_type": item["media_type"],
                "download": f"/v1/jobs/{job_id}/artifacts/download?path={item['path']}",
            }
            for item in state.artifacts
        ],
    }


@router.get(
    "/jobs/{job_id}/artifacts/download",
    response_class=StreamingResponse,
    tags=["Artifacts"],
    summary="Download one output by its artifact path",
    description="Obtain valid path values from GET /v1/jobs/{job_id}/artifacts.",
    operation_id="download_artifact",
    responses={
        200: {
            "description": "The requested artifact using its recorded media type.",
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        **ARTIFACT_ERRORS,
    },
)
def download_artifact(
    job_id: str,
    path: Annotated[
        str,
        Query(
            description=(
                "Artifact path returned by the artifact list, for example "
                "index/vectors.jsonl."
            )
        ),
    ],
):
    return artifact_download(get_job_state(job_id), path)


@router.get(
    "/jobs/{job_id}/bundle",
    response_class=StreamingResponse,
    tags=["Artifacts"],
    summary="Download all outputs as one ZIP",
    operation_id="download_bundle",
    responses={
        200: {
            "description": "ZIP containing the manifest and every output artifact.",
            "content": {
                "application/zip": {"schema": {"type": "string", "format": "binary"}}
            },
        },
        **ARTIFACT_ERRORS,
    },
)
def download_bundle(job_id: str):
    return artifact_download(
        get_job_state(job_id),
        "bundle.zip",
        media_type="application/zip",
        filename=f"corpusretrieve-{job_id}.zip",
    )


@router.delete(
    "/jobs/{job_id}",
    response_model=DeleteResponse,
    tags=["Jobs"],
    summary="Delete a job and all remotely stored inputs and outputs",
    operation_id="delete_job",
    responses=NOT_FOUND,
)
def delete_job(job_id: str):
    get_job_state(job_id)
    get_credential_store().delete(job_id)
    get_artifact_store().delete_prefix(f"jobs/{job_id}/")
    get_job_store().delete(job_id)
    return {"message": f"Job {job_id} and its stored objects were deleted."}


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
    summary="Check service dependencies",
    operation_id="health",
    responses={503: {"model": HealthResponse, "description": "Dependency unavailable."}},
)
def health():
    settings = get_settings()
    checks = {"artifact_store": False, "job_store": False}
    try:
        checks["artifact_store"] = get_artifact_store().ping()
        checks["job_store"] = get_job_store().ping()
    except Exception as exc:  # noqa: BLE001 - health must never raise
        logger.warning("Health dependency check failed: %s", exc)
    healthy = all(checks.values())
    return JSONResponse(
        {
            "status": "ok" if healthy else "unavailable",
            "service": settings.app_name,
            "version": settings.version,
            "storage": settings.storage_backend,
            "job_mode": settings.job_mode,
            "embedders": available_embedders(),
            "vlm_api_key_configured": bool(settings.anthropic_api_key),
            "console_enabled": settings.console_enabled,
            "checks": checks,
        },
        status_code=200 if healthy else 503,
    )
