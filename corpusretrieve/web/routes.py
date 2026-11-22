from __future__ import annotations

import json
from typing import Annotated, Any

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from corpusretrieve.api.common import (
    artifact_json,
    get_job_state,
    job_status_payload,
    manifest_data,
)
from corpusretrieve.api.errors import APIError
from corpusretrieve.api.formats import embedder_catalog, ocr_preflight
from corpusretrieve.api.outputs import job_event_stream, require_document
from corpusretrieve.config import get_settings
from corpusretrieve.models import STAGES
from corpusretrieve.state import get_job_store
from corpusretrieve.storage import get_artifact_store
from corpusretrieve.web.projection import project_records
from corpusretrieve.web.views import (
    artifact_group,
    document_chunks,
    document_elements,
    document_graph,
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
        "require_bibliographic": settings.require_bibliographic,
        "degraded_policy": settings.degraded_policy,
        "storage_backend": settings.storage_backend,
        "embedders": embedder_catalog(),
        "ocr": ocr_preflight(settings.ocr_langs),
    }


@router.get("/jobs")
def console_jobs(limit: Annotated[int, Query(ge=1, le=200)] = 30):
    """Recent jobs, newest first, so a finished job can be reopened after a reload."""
    states = get_job_store().list_states(limit)
    return {"jobs": [
        {key: payload[key] for key in ("job_id", "state", "stage", "corpus_name", "created_at",
                                       "updated_at", "artifacts_available", "release_ready",
                                       "message")}
        for payload in (job_status_payload(state) for state in states)
    ]}


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
    # one pass over the artifact list (a large build has >100,000 entries)
    with_previews = {
        artifact["path"].split("/", 2)[1]
        for artifact in state.artifacts
        if artifact["path"].startswith("documents/") and "/assets/pages/" in artifact["path"]
    }
    documents = []
    for item in manifest.get("documents", []):
        doc_id = item["doc_id"]
        quality = item.get("quality") or {}
        documents.append(
            {
                "doc_id": doc_id,
                "source": item["source"],
                "chunks": item["chunks"],
                "metadata": item.get("metadata", {}),
                "has_previews": doc_id in with_previews,
                "degraded": bool(quality.get("degraded")),
                "pages": quality.get("pages"),
            }
        )
    quality = manifest.get("quality") or {}
    return {
        "job_id": job_id,
        "corpus_name": manifest.get("corpus_name"),
        "documents": documents,
        "index": manifest.get("index", {}),
        "knowledge_graph": {key: value for key, value in
                            (manifest.get("knowledge_graph") or {}).items()
                            if key not in {"artifacts"}},
        "warnings": manifest.get("submission", {}).get("warnings", []),
        "release_ready": manifest.get("release_ready"),
        "quality_reasons": quality.get("reasons", []),
        "complete": manifest.get("complete", True),
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
def console_graph(job_id: str, doc_id: str | None = None):
    """The knowledge graph of one document (the first when none is named)."""
    state, manifest = manifest_data(job_id)
    documents = manifest.get("documents", [])
    if not documents:
        raise APIError(409, "graph-unavailable", "This job produced no documents.")
    doc_id = doc_id or documents[0]["doc_id"]
    require_document(manifest, doc_id)
    document = artifact_json(state, f"documents/{doc_id}/chunks.json")
    index = manifest.get("index") or {}
    view = document_graph(document, manifest.get("corpus_name", "corpus"),
                          index.get("model", "hashing-v1"))
    view["corpus"] = {key: value for key, value in
                      (manifest.get("knowledge_graph") or {}).items()
                      if key in {"nodes", "edges", "node_types", "edge_types"}}
    return JSONResponse(view)


@router.get("/jobs/{job_id}/projection")
def console_projection(job_id: str):
    """Project the job's dense vectors to two dimensions, caching the result."""
    state = get_job_state(job_id)
    store = get_artifact_store()
    cache_key = PROJECTION_CACHE_TEMPLATE.format(job_id=job_id)
    if store.exists(cache_key):
        return JSONResponse(json.loads(store.get_bytes(cache_key)))

    settings = get_settings()
    projection = project_sample(state, settings.console_max_projection_points)
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
    bundle = next((a for a in state.artifacts if a["path"] == "bundle.zip"), None)
    return {
        "job_id": job_id,
        "state": state.state,
        # corpus-level outputs, listed in full; per-document files are summarised
        "artifacts": [
            {**item, "download": f"/v1/jobs/{job_id}/artifacts/download?path={item['path']}"}
            for item in state.artifacts
            if artifact_group(item["path"]) != "Documents"
        ],
        "artifact_groups": grouped_artifacts(job_id, state.artifacts),
        "bundle": f"/v1/jobs/{job_id}/bundle" if bundle else None,
        "bundle_size_bytes": bundle["size_bytes"] if bundle else None,
    }


def project_sample(state, limit: int) -> dict[str, Any]:
    """Project an even sample of the whole index, not just its first rows."""
    import io

    import numpy as np

    metadata = artifact_json(state, "index/metadata.json")
    total = int(metadata.get("records") or 0)
    if not total:
        return {"method": "none", "n": 0, "points": []}
    stride = max(1, -(-total // limit))
    rows: list[int] = []
    payloads: list[dict[str, Any]] = []
    inline: list[list[float]] = []
    entry = next(a for a in state.artifacts if a["path"] == "index/vectors.jsonl")
    store = get_artifact_store()
    buffer, row = b"", 0
    for block in store.stream(entry["key"]):
        buffer += block
        *lines, buffer = buffer.split(b"\n")
        for line in lines:
            if line.strip() and row % stride == 0 and len(rows) < limit:
                record = json.loads(line)
                rows.append(row)
                payloads.append({"id": record["id"], "payload": record.get("payload", {})})
                inline.append(record.get("vector") or [])
            row += line.strip() != b""
    dense_name = metadata.get("dense_file", "index/dense.npy")
    dense_entry = next((a for a in state.artifacts if a["path"] == dense_name), None)
    if dense_entry is not None:
        local = getattr(store, "local_path", None)
        matrix = (np.load(local(dense_entry["key"]), mmap_mode="r") if local
                  else np.load(io.BytesIO(store.get_bytes(dense_entry["key"]))))
        vectors = np.asarray(matrix[rows], dtype="float32")
    else:
        vectors = np.asarray(inline, dtype="float32")
    result = project_records(payloads, vectors)
    result["sampled_from"] = total
    return result
