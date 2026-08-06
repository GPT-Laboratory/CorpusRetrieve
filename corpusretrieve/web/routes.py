from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from corpusretrieve.api.common import (
    artifact_bytes,
    artifact_json,
    get_job_state,
    job_status_payload,
    manifest_data,
)
from corpusretrieve.api.errors import APIError
from corpusretrieve.api.outputs import job_event_stream, require_document
from corpusretrieve.config import get_settings
from corpusretrieve.models import STAGES
from corpusretrieve.storage import get_artifact_store
from corpusretrieve.web.projection import project_vectors
from corpusretrieve.web.views import (
    DEFAULT_HIDDEN_NODE_TYPES,
    document_chunks,
    document_elements,
    graph_view,
    grouped_artifacts,
)

router = APIRouter(prefix="/console", include_in_schema=False)

PROJECTION_CACHE_TEMPLATE = "jobs/{job_id}/derived/projection.json"


@router.get("/settings")
def console_settings():
    """Everything the single-page application needs before its first upload."""
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.version,
        "stages": list(STAGES),
        "default_embedder": settings.default_embedder,
        "default_vlm_model": settings.default_vlm_model,
        "vlm_api_key_configured": bool(settings.anthropic_api_key),
        "max_files_per_job": settings.max_files_per_job,
        "max_bytes_per_file": settings.max_upload_bytes,
        "page_previews_enabled": settings.emit_page_previews,
    }


@router.get("/jobs/{job_id}")
def console_job(job_id: str):
    return job_status_payload(get_job_state(job_id))


@router.get("/jobs/{job_id}/events")
def console_events(job_id: str):
    get_job_state(job_id)
    return StreamingResponse(
        job_event_stream(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/documents")
def console_documents(job_id: str):
    state, manifest = manifest_data(job_id)
    documents = []
    for item in manifest.get("documents", []):
        doc_id = item["doc_id"]
        documents.append(
            {
                "doc_id": doc_id,
                "source": item["source"],
                "chunks": item["chunks"],
                "metadata": item.get("metadata", {}),
                "has_previews": any(
                    artifact["path"].startswith(
                        f"documents/{doc_id}/assets/pages/"
                    )
                    for artifact in state.artifacts
                ),
            }
        )
    return {
        "job_id": job_id,
        "corpus_name": manifest.get("corpus_name"),
        "documents": documents,
        "index": manifest.get("index", {}),
        "knowledge_graph": manifest.get("knowledge_graph", {}),
        "warnings": manifest.get("submission", {}).get("warnings", []),
    }


@router.get("/jobs/{job_id}/documents/{document_id}/elements")
def console_elements(
    job_id: str,
    document_id: str,
    page: Annotated[int | None, Query(ge=1)] = None,
    kind: str | None = None,
):
    state, manifest = manifest_data(job_id)
    require_document(manifest, document_id)
    extraction = artifact_json(state, f"documents/{document_id}/extraction.json")
    return document_elements(extraction, page, kind)


@router.get("/jobs/{job_id}/documents/{document_id}/chunks")
def console_chunks(job_id: str, document_id: str):
    state, manifest = manifest_data(job_id)
    require_document(manifest, document_id)
    document = artifact_json(state, f"documents/{document_id}/chunks.json")
    return document_chunks(document)


@router.get("/jobs/{job_id}/graph")
def console_graph(
    job_id: str,
    hide: str = ",".join(DEFAULT_HIDDEN_NODE_TYPES),
):
    state = get_job_state(job_id)
    graph = artifact_json(state, "knowledge_graph/knowledge_graph.json")
    hidden = {value for value in hide.split(",") if value}
    return JSONResponse(graph_view(graph, hidden))


@router.get("/jobs/{job_id}/projection")
def console_projection(job_id: str):
    """Project the job's dense vectors to two dimensions, caching the result."""
    state = get_job_state(job_id)
    store = get_artifact_store()
    cache_key = PROJECTION_CACHE_TEMPLATE.format(job_id=job_id)
    if store.exists(cache_key):
        return JSONResponse(json.loads(store.get_bytes(cache_key)))

    settings = get_settings()
    raw = artifact_bytes(state, "index/vectors.jsonl").decode("utf-8")
    projection = project_vectors(
        raw.splitlines(), settings.console_max_projection_points
    )
    if not projection["points"]:
        raise APIError(
            409, "projection-unavailable", "This job produced no dense vectors."
        )
    store.put_bytes(
        cache_key,
        json.dumps(projection).encode("utf-8"),
        "application/json",
    )
    return JSONResponse(projection)


@router.get("/jobs/{job_id}/artifacts")
def console_artifacts(job_id: str) -> dict[str, Any]:
    """Artifacts grouped by output family for the downloads panel."""
    state = get_job_state(job_id)
    return {
        "job_id": job_id,
        "state": state.state,
        "artifact_groups": grouped_artifacts(job_id, state.artifacts),
        "bundle": f"/v1/jobs/{job_id}/bundle",
    }
