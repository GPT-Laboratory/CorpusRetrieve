from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import quote

from fastapi import APIRouter
from fastapi.responses import JSONResponse, StreamingResponse

from corpusretrieve.api.common import (
    artifact_download,
    artifact_json,
    get_job_state,
    job_status_payload,
    manifest_data,
)
from corpusretrieve.api.errors import APIError
from corpusretrieve.api.formats import SUPPORTED_FORMATS, available_embedders
from corpusretrieve.api.schemas import (
    AssetList,
    CapabilitiesResponse,
    DocumentList,
    IndexDescription,
    ProblemDetail,
)
from corpusretrieve.config import get_settings
from corpusretrieve.models import STAGES
from corpusretrieve.state import get_job_store

router = APIRouter(prefix="/v1")

RESOURCE_ERRORS = {
    404: {"model": ProblemDetail, "description": "Job or resource not found."},
    409: {"model": ProblemDetail, "description": "Output is not ready."},
    410: {"model": ProblemDetail, "description": "Output retention has expired."},
}
EVENT_POLL_SECONDS = 0.5


def require_document(manifest: dict[str, Any], document_id: str) -> dict[str, Any]:
    document = next(
        (
            item
            for item in manifest.get("documents", [])
            if item["doc_id"] == document_id
        ),
        None,
    )
    if document is None:
        raise APIError(
            404,
            "document-not-found",
            f"Document {document_id} is not part of this job.",
        )
    return document


@router.get(
    "/capabilities",
    response_model=CapabilitiesResponse,
    tags=["Service"],
    summary="List upload choices, formats, limits, and outputs",
    operation_id="get_capabilities",
)
def capabilities():
    settings = get_settings()
    return {
        "service": settings.app_name,
        "version": settings.version,
        "inputs": {
            "upload_endpoints": {"all": "/v1/jobs"},
            "formats": SUPPORTED_FORMATS,
            "max_files_per_job": settings.max_files_per_job,
            "max_bytes_per_file": settings.max_upload_bytes,
            "max_archive_uncompressed_bytes": settings.max_corpus_uncompressed_bytes,
            "folder_upload": "Send folder files together or send one ZIP to /v1/jobs.",
        },
        "embedders": available_embedders(),
        "vlm": {
            "provider": "anthropic",
            "api_key_configured": bool(settings.anthropic_api_key),
            "request_override_header": "X-Anthropic-API-Key",
            "default_model": settings.default_vlm_model,
        },
        "stages": list(STAGES),
        "outputs": {
            "documents": "extraction, chunks, and assets",
            "corpus": "combined chunks",
            "index": "metadata, dense/sparse vectors, and sparse encoder",
            "knowledge_graph": "JSON, GraphML, and Cypher",
            "package": "manifest, artifact catalog, and complete ZIP bundle",
        },
    }


@router.get(
    "/jobs/{job_id}/events",
    response_class=StreamingResponse,
    tags=["Jobs"],
    summary="Stream job progress with server-sent events",
    description="Emits event: job messages until the job completes or fails.",
    operation_id="stream_job_events",
    responses={
        200: {
            "description": "Server-sent job events.",
            "content": {"text/event-stream": {"schema": {"type": "string"}}},
        },
        404: RESOURCE_ERRORS[404],
    },
)
def stream_job_events(job_id: str):
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


async def job_event_stream(job_id: str):
    last_update = None
    while True:
        state = get_job_store().get_state(job_id)
        if state is None:
            yield "event: deleted\ndata: {}\n\n"
            break
        if state.updated_at != last_update:
            payload = json.dumps(job_status_payload(state))
            yield f"event: job\ndata: {payload}\n\n"
            last_update = state.updated_at
        if state.state in {"completed", "failed"}:
            break
        await asyncio.sleep(EVENT_POLL_SECONDS)


@router.get(
    "/jobs/{job_id}/documents",
    response_model=DocumentList,
    tags=["Documents"],
    summary="List documents processed in the corpus",
    operation_id="list_documents",
    responses=RESOURCE_ERRORS,
)
def list_documents(job_id: str):
    _, manifest = manifest_data(job_id)
    documents = []
    for item in manifest.get("documents", []):
        base = f"/v1/jobs/{job_id}/documents/{item['doc_id']}"
        documents.append(
            {
                **item,
                "links": {
                    "extraction": f"{base}/extraction",
                    "chunks": f"{base}/chunks",
                    "assets": f"{base}/assets",
                },
            }
        )
    return {"job_id": job_id, "documents": documents}


@router.get(
    "/jobs/{job_id}/documents/{document_id}/extraction",
    tags=["Documents"],
    summary="Read extracted elements and provenance for one document",
    operation_id="get_document_extraction",
    response_model=dict[str, Any],
    responses=RESOURCE_ERRORS,
)
def get_document_extraction(job_id: str, document_id: str):
    state, manifest = manifest_data(job_id)
    require_document(manifest, document_id)
    return JSONResponse(
        artifact_json(state, f"documents/{document_id}/extraction.json")
    )


@router.get(
    "/jobs/{job_id}/documents/{document_id}/chunks",
    tags=["Documents"],
    summary="Read retrieval chunks for one document",
    operation_id="get_document_chunks",
    response_model=dict[str, Any],
    responses=RESOURCE_ERRORS,
)
def get_document_chunks(job_id: str, document_id: str):
    state, manifest = manifest_data(job_id)
    require_document(manifest, document_id)
    return JSONResponse(artifact_json(state, f"documents/{document_id}/chunks.json"))


@router.get(
    "/jobs/{job_id}/documents/{document_id}/assets",
    tags=["Documents"],
    summary="List extracted figures, page previews, and other assets",
    operation_id="list_document_assets",
    response_model=AssetList,
    responses=RESOURCE_ERRORS,
)
def list_document_assets(job_id: str, document_id: str):
    state, manifest = manifest_data(job_id)
    require_document(manifest, document_id)
    prefix = f"documents/{document_id}/assets/"
    assets = [
        {
            "path": item["path"][len(prefix):],
            "size_bytes": item["size_bytes"],
            "media_type": item["media_type"],
            "download": (
                f"/v1/jobs/{job_id}/documents/{document_id}/assets/"
                f"{quote(item['path'][len(prefix):], safe='/')}"
            ),
        }
        for item in state.artifacts
        if item["path"].startswith(prefix)
    ]
    return {"job_id": job_id, "document_id": document_id, "assets": assets}


@router.get(
    "/jobs/{job_id}/documents/{document_id}/assets/{asset_path:path}",
    response_class=StreamingResponse,
    tags=["Documents"],
    summary="Download one document asset",
    operation_id="download_document_asset",
    responses={
        200: {
            "description": "Asset bytes using the artifact's recorded media type.",
            "content": {
                "application/octet-stream": {
                    "schema": {"type": "string", "format": "binary"}
                }
            },
        },
        **RESOURCE_ERRORS,
    },
)
def download_document_asset(job_id: str, document_id: str, asset_path: str):
    state, manifest = manifest_data(job_id)
    require_document(manifest, document_id)
    return artifact_download(state, f"documents/{document_id}/assets/{asset_path}")


@router.get(
    "/jobs/{job_id}/corpus/chunks",
    tags=["Corpus"],
    summary="Read all chunks across the corpus",
    operation_id="get_corpus_chunks",
    response_model=list[dict[str, Any]],
    responses=RESOURCE_ERRORS,
)
def get_corpus_chunks(job_id: str):
    return JSONResponse(artifact_json(get_job_state(job_id), "corpus/chunks.json"))


@router.get(
    "/jobs/{job_id}/index",
    response_model=IndexDescription,
    tags=["Index"],
    summary="Read portable index metadata and download links",
    operation_id="get_index",
    responses=RESOURCE_ERRORS,
)
def get_index(job_id: str):
    metadata = artifact_json(get_job_state(job_id), "index/metadata.json")
    base = f"/v1/jobs/{job_id}/index"
    return {
        "job_id": job_id,
        **metadata,
        "files": {
            "vectors": f"{base}/vectors",
            "sparse_encoder": f"{base}/sparse-encoder",
        },
    }


@router.get(
    "/jobs/{job_id}/index/vectors",
    response_class=StreamingResponse,
    tags=["Index"],
    summary="Download vector records as JSON Lines",
    description=(
        "Each line contains a stable ID, dense vector, sparse vector, collection, "
        "text, document identity, and provenance payload."
    ),
    operation_id="download_vectors",
    responses={
        200: {
            "description": "Portable vector records.",
            "content": {"application/x-ndjson": {"schema": {"type": "string"}}},
        },
        **RESOURCE_ERRORS,
    },
)
def download_vectors(job_id: str):
    return artifact_download(
        get_job_state(job_id),
        "index/vectors.jsonl",
        media_type="application/x-ndjson",
    )


@router.get(
    "/jobs/{job_id}/index/sparse-encoder",
    tags=["Index"],
    summary="Read the fitted sparse encoder",
    operation_id="get_sparse_encoder",
    response_model=dict[str, Any],
    responses=RESOURCE_ERRORS,
)
def get_sparse_encoder(job_id: str):
    return JSONResponse(
        artifact_json(get_job_state(job_id), "index/sparse_encoder.json")
    )


@router.get(
    "/jobs/{job_id}/graph",
    tags=["Knowledge Graph"],
    summary="Read the knowledge graph as JSON",
    operation_id="get_graph",
    response_model=dict[str, Any],
    responses=RESOURCE_ERRORS,
)
def get_graph(job_id: str):
    return JSONResponse(
        artifact_json(get_job_state(job_id), "knowledge_graph/knowledge_graph.json")
    )


@router.get(
    "/jobs/{job_id}/graph/graphml",
    response_class=StreamingResponse,
    tags=["Knowledge Graph"],
    summary="Download the knowledge graph as GraphML",
    operation_id="download_graphml",
    responses={
        200: {
            "description": "GraphML file.",
            "content": {"application/graphml+xml": {"schema": {"type": "string"}}},
        },
        **RESOURCE_ERRORS,
    },
)
def download_graphml(job_id: str):
    return artifact_download(
        get_job_state(job_id),
        "knowledge_graph/knowledge_graph.graphml",
        media_type="application/graphml+xml",
    )


@router.get(
    "/jobs/{job_id}/graph/cypher",
    response_class=StreamingResponse,
    tags=["Knowledge Graph"],
    summary="Download Neo4j or Memgraph Cypher statements",
    operation_id="download_cypher",
    responses={
        200: {
            "description": "Cypher import file.",
            "content": {"text/plain": {"schema": {"type": "string"}}},
        },
        **RESOURCE_ERRORS,
    },
)
def download_cypher(job_id: str):
    return artifact_download(
        get_job_state(job_id),
        "knowledge_graph/knowledge_graph.cypher",
        media_type="text/plain; charset=utf-8",
    )
