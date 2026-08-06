from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from fastapi.responses import StreamingResponse

from corpusretrieve.api.errors import APIError
from corpusretrieve.models import JobState
from corpusretrieve.state import get_job_store
from corpusretrieve.storage import get_artifact_store

JOB_ID_PATTERN = re.compile(r"[0-9a-f]{24}")


def get_job_state(job_id: str) -> JobState:
    if not JOB_ID_PATTERN.fullmatch(job_id):
        raise APIError(404, "job-not-found", "No job exists with that ID.")
    state = get_job_store().get_state(job_id)
    if state is None:
        raise APIError(404, "job-not-found", "No job exists with that ID.")
    return state


def job_links(job_id: str) -> dict[str, str]:
    base = f"/v1/jobs/{job_id}"
    return {
        "status": base,
        "events": f"{base}/events",
        "manifest": f"{base}/manifest",
        "documents": f"{base}/documents",
        "corpus_chunks": f"{base}/corpus/chunks",
        "index": f"{base}/index",
        "graph": f"{base}/graph",
        "artifacts": f"{base}/artifacts",
        "bundle": f"{base}/bundle",
    }


def job_status_payload(state: JobState) -> dict[str, Any]:
    return {
        "job_id": state.job_id,
        "state": state.state,
        "stage": state.stage,
        "progress": state.progress,
        "message": state.message,
        "error": state.error,
        "warnings": state.warnings,
        "stages": state.stages,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "outputs_ready": state.state == "completed",
        "links": job_links(state.job_id),
    }


def artifact_entry(state: JobState, path: str) -> dict[str, Any]:
    entry = next((item for item in state.artifacts if item["path"] == path), None)
    if entry is None:
        if state.state not in {"completed", "failed"}:
            raise APIError(
                409,
                "artifact-not-ready",
                "The requested output is not ready. Poll the job status or event stream.",
            )
        raise APIError(404, "artifact-not-found", f"No output exists at path: {path}.")
    if not get_artifact_store().exists(entry["key"]):
        raise APIError(
            410,
            "artifact-expired",
            "The output is no longer available under the configured retention policy.",
        )
    return entry


def artifact_bytes(state: JobState, path: str) -> bytes:
    return get_artifact_store().get_bytes(artifact_entry(state, path)["key"])


def artifact_json(state: JobState, path: str) -> Any:
    try:
        return json.loads(artifact_bytes(state, path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise APIError(
            500,
            "invalid-generated-json",
            f"The generated JSON output is invalid: {path}.",
        ) from exc


def artifact_download(
    state: JobState,
    path: str,
    media_type: str | None = None,
    filename: str | None = None,
) -> StreamingResponse:
    entry = artifact_entry(state, path)
    name = re.sub(r"[\x00-\x1f\x7f\"\\]", "_", filename or Path(path).name)
    return StreamingResponse(
        get_artifact_store().stream(entry["key"]),
        media_type=media_type or entry["media_type"],
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )


def manifest_data(job_id: str) -> tuple[JobState, dict[str, Any]]:
    state = get_job_state(job_id)
    if not state.manifest_key:
        raise APIError(
            409,
            "manifest-not-ready",
            "The manifest is not ready. Poll the job until state is completed.",
        )
    return state, artifact_json(state, "manifest.json")
