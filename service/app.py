"""docparse evidence service — turns documents into retrievable, grounded evidence.

Scope (deliberately bounded):
  IN   : documents of any type, any field
  OUT  : ranked, reranked, graph-grounded evidence with full provenance
  NOT  : answer generation / LLM chat (that is a separate Stage-7 service that
         CALLS this one). This service never talks to an LLM for answers.

Pipeline exposed:
  /extract  parse+enrich a document        (Stage 1-3)
  /chunk    parse -> typed chunks           (Stage 4)
  /ingest   chunks -> vectors + graph       (Stage 5)   [persisted to INDEX_DIR]
  /search   query -> grounded evidence      (Stage 6: hybrid+rerank+graph-walk)
  /healthz  liveness + index stats

The service is stateful in exactly one place: a persisted index directory
(INDEX_DIR) holding the qdrant db, the knowledge-graph JSON, the sparse
encoder, and a slim chunk cache. On startup it loads that index into a single
Retriever reused across all requests. Restarting the process does not lose the
index. Nothing in this file is field-specific.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from docparse import ExtractionConfig, chunk_result, extract
from docparse.vectorization import (EdgePolicy, IngestConfig, RetrieveConfig,
                                    Retriever, ingest, load_retriever)

app = FastAPI(
    title="docparse evidence service",
    version="2.0",
    description="Documents in, grounded evidence out. Parse - chunk - vectorize "
                "- knowledge graph - hybrid retrieval - rerank - graph expansion. "
                "Field-agnostic. Answer generation lives in a separate service.")

# --- configuration (all env-driven; no code changes to deploy) --------------
INDEX_DIR = Path(os.environ.get("INDEX_DIR", "/data/index"))
QDRANT_URL = os.environ.get("QDRANT_URL")                 # http://qdrant:6333
EMBEDDER = os.environ.get("EMBEDDER", "auto")             # auto|hash|bge-m3|st:<m>
RERANKER = os.environ.get("RERANKER", "auto")             # auto|lexical|ce:<m>|off
INDEX_DIR.mkdir(parents=True, exist_ok=True)

_RETRIEVER: Retriever | None = None


def _retriever() -> Retriever:
    """Lazy-load the index once; reuse across requests."""
    global _RETRIEVER
    if _RETRIEVER is None:
        if not (INDEX_DIR / "sparse_encoder.json").exists():
            raise HTTPException(409, "no index yet — POST /ingest first")
        _RETRIEVER = load_retriever(INDEX_DIR, embedder=EMBEDDER,
                                    reranker=RERANKER, qdrant_url=QDRANT_URL)
    return _RETRIEVER


def _save_upload(f: UploadFile) -> str:
    suffix = Path(f.filename or "u.bin").suffix or ".bin"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=INDEX_DIR) as tf:
        tf.write(f.file.read())
        return tf.name


# --------------------------------------------------------------------- Stage 1-3

@app.post("/extract", tags=["parse"])
async def extract_endpoint(file: UploadFile = File(...),
                           scientific: bool = Query(True,
                               description="enable role/claim/citation enrichment")):
    path = _save_upload(file)
    try:
        cfg = ExtractionConfig(scientific=scientific,
                               table_serializations=("markdown", "kv"))
        return JSONResponse(extract(path, cfg).to_dict())
    finally:
        os.unlink(path)


# --------------------------------------------------------------------- Stage 4

@app.post("/chunk", tags=["parse"])
async def chunk_endpoint(file: UploadFile = File(...)):
    path = _save_upload(file)
    try:
        result = extract(path, ExtractionConfig(table_serializations=("markdown", "kv")))
        return JSONResponse([chunk_result(result).to_dict()])
    finally:
        os.unlink(path)


# --------------------------------------------------------------------- Stage 5

@app.post("/ingest", tags=["index"])
async def ingest_endpoint(file: UploadFile = File(...),
                          tenant: str = Query("default")):
    """Ingest a chunks.json into the persistent index. Idempotent: re-ingesting
    the same document upserts (no duplicates). Invalidates the in-memory
    retriever so the next /search sees the new content."""
    global _RETRIEVER
    path = _save_upload(file)
    try:
        cfg = IngestConfig(tenant=tenant, embedder=EMBEDDER,
                           db_location=QDRANT_URL or str(INDEX_DIR / "qdrant"),
                           graph_outdir=str(INDEX_DIR))
        res, *_ = ingest(path, cfg)
        _RETRIEVER = None                    # force reload with new index
        return JSONResponse({"doc_id": res.doc_id, "collections": res.collections,
                             "graph": res.graph_stats, "validation": res.validation,
                             "timings": res.timings})
    except Exception as exc:
        raise HTTPException(500, f"ingest failed: {exc}")
    finally:
        os.unlink(path)


# --------------------------------------------------------------------- Stage 6

@app.get("/search", tags=["retrieve"])
def search_endpoint(
    q: str = Query(..., description="natural-language query"),
    k: int = Query(5, ge=1, le=50),
    collections: str | None = Query(None, description="comma-separated type-groups"),
    tenant: str | None = Query(None),
    doc_ids: str | None = Query(None, description="comma-separated doc ids to restrict to"),
    rerank: bool = Query(True, description="cross-encoder reranking of candidates"),
    rerank_alpha: float = Query(1.0, ge=0.0, le=1.0,
        description="1=pure rerank, 0=retrieval order, blend between"),
    expand: bool = Query(True, description="attach graph-neighbour evidence"),
    include_references: bool = Query(False),
    max_hops: int = Query(1, ge=0, le=3),
):
    """Grounded evidence for a query: hybrid retrieval -> rerank -> graph walk.
    Returns ranked bundles, each with provenance and supporting neighbours."""
    r = _retriever()
    cfg = RetrieveConfig(
        limit=k,
        collections=collections.split(",") if collections else None,
        tenant=tenant,
        doc_ids=doc_ids.split(",") if doc_ids else None,
        rerank=rerank, rerank_alpha=rerank_alpha,
        expand=expand, include_references=include_references,
        edge_policy=EdgePolicy(max_hops=max_hops,
                               include_citations=include_references),
    )
    bundles = r.retrieve(q, cfg)
    return {"query": q, "results": [_present(b) for b in bundles]}


def _present(bundle) -> dict:
    """Shape a bundle for API consumers (e.g. a Stage-7 answer service).
    Everything needed to ground and cite an answer is here."""
    p = bundle.hit.payload
    return {
        "chunk_id": bundle.hit.chunk_id,
        "chunk_type": bundle.hit.chunk_type,
        "score": round(bundle.hit.score, 4),
        "retrieval_score": p.get("retrieval_score"),
        "rerank_score": p.get("rerank_score"),
        "text": p.get("raw_text", "")[:1200],
        "provenance": {
            "doc_id": p.get("doc_id"), "pages": p.get("pages"),
            "section_path": p.get("section_path"),
            "figure_label": p.get("figure_label"),
            "table_label": p.get("table_label"),
            "char_start": p.get("char_start"), "char_end": p.get("char_end"),
            "visual_ref": p.get("visual_ref"),
        },
        "evidence": bundle.neighbours,
    }


# --------------------------------------------------------------------- health

@app.get("/healthz", tags=["ops"])
def healthz():
    out = {"status": "ok", "embedder": EMBEDDER, "reranker": RERANKER,
           "qdrant": QDRANT_URL or "local", "index_dir": str(INDEX_DIR),
           "has_index": (INDEX_DIR / "sparse_encoder.json").exists()}
    if _RETRIEVER is not None:
        out["index"] = _RETRIEVER.stats()
    return out
