"""Stage-1 web application API (developer mode).

Adds a document-in / evidence-out web workflow on top of the existing service:

    POST /api/jobs            upload a document -> {job_id}
    GET  /api/jobs/{id}/stream  SSE: six-stage pipeline progress, live
    GET  /api/jobs/{id}/manifest        what artifacts exist + summary
    GET  /api/jobs/{id}/elements        typed elements with bbox provenance
    GET  /api/jobs/{id}/chunks          chunks + which elements/pages they cover
    GET  /api/jobs/{id}/page/{n}.png    rendered page image for the overlay
    GET  /api/jobs/{id}/graph           knowledge-graph nodes/edges (for viz)
    GET  /api/jobs/{id}/download/{art}  download a portable artifact

This module is import-safe even if FastAPI isn't installed elsewhere; the
existing evidence-service endpoints live in service.app and are unaffected.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from service.pipeline_runner import PipelineRunner, new_job_id

JOBS_DIR = Path(
    os.getenv(
        "JOBS_DIR",
        str(Path(tempfile.gettempdir()) / "corpusretrieve" / "jobs"),
    )
)
EMBEDDER = os.environ.get("EMBEDDER", "hash")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="CorpusRetrieve web (Stage 1)", version="1.0",
              description="Upload a document, watch it become evidence, download "
                          "portable chunks + embeddings + knowledge graph.")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_runner = PipelineRunner(JOBS_DIR, embedder_spec=EMBEDDER)

_DOWNLOAD_MIME = {
    "chunks": ("chunks.json", "application/json"),
    "extraction": ("extraction.json", "application/json"),
    "embeddings": ("embeddings.jsonl", "application/x-ndjson"),
    "embeddings_meta": ("embeddings_meta.json", "application/json"),
    "graph_json": ("knowledge_graph.json", "application/json"),
    "graph_graphml": ("knowledge_graph.graphml", "application/xml"),
    "graph_cypher": ("knowledge_graph.cypher", "text/plain"),
    "sparse_encoder": ("sparse_encoder.json", "application/json"),
}


# --------------------------------------------------------------------- upload

@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...)):
    job_id = new_job_id()
    jd = _runner.job_dir(job_id)
    suffix = Path(file.filename or "upload.bin").suffix or ".bin"
    src = jd / f"source{suffix}"
    src.write_bytes(await file.read())
    (jd / "status.json").write_text(json.dumps(
        {"job_id": job_id, "filename": file.filename, "state": "uploaded"}),
        encoding="utf-8")
    return {"job_id": job_id, "filename": file.filename}


# --------------------------------------------------------------------- stream

@app.get("/api/jobs/{job_id}/stream")
def stream_job(job_id: str, describe_figures: bool = False):
    jd = _runner.job_dir(job_id)
    src = next((p for p in jd.glob("source.*")), None)
    if src is None:
        raise HTTPException(404, "job not found or no source uploaded")

    def gen():
        yield _sse({"event": "run_started", "stages": _runner_stages()}, "meta")
        try:
            for stage_event in _runner.run(job_id, str(src),
                                           describe_figures=describe_figures):
                yield _sse(json.loads(stage_event.to_json()), "stage")
        except Exception as exc:  # noqa: BLE001
            yield _sse({"error": str(exc)}, "error")
            return
        # final: hand the manifest so the client can render downloads/summary
        manifest = jd / "manifest.json"
        if manifest.exists():
            yield _sse(json.loads(manifest.read_text(encoding="utf-8")), "manifest")
        yield _sse({"event": "run_complete"}, "done")

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --------------------------------------------------------------------- reads

@app.get("/api/jobs/{job_id}/manifest")
def get_manifest(job_id: str):
    return _read_json(job_id, "manifest.json")


@app.get("/api/jobs/{job_id}/elements")
def get_elements(job_id: str, page: int | None = None, kind: str | None = None):
    """Typed elements with bbox + page, plus page dimensions, so a UI can draw
    provenance overlays. Optional page/kind filters."""
    ex = _read_json(job_id, "extraction.json")
    els = ex.get("elements", [])
    if page is not None:
        els = [e for e in els if e.get("page") == page]
    if kind:
        els = [e for e in els if e.get("kind") == kind]
    return {"page_dims": ex.get("stats", {}).get("page_dims", {}),
            "page_count": ex.get("page_count"),
            "count": len(els),
            "elements": [{
                "id": e.get("id"), "kind": e.get("kind"), "page": e.get("page"),
                "bbox": e.get("bbox"), "role": (e.get("metadata") or {}).get("role"),
                "text": (e.get("text") or "")[:600],
            } for e in els]}


@app.get("/api/jobs/{job_id}/chunks")
def get_chunks(job_id: str):
    """Chunks with the info needed to trace each back to source regions:
    element_ids, pages, section_path, char offsets, type, provenance labels."""
    data = _read_json(job_id, "chunks.json")
    doc = data[0] if isinstance(data, list) else data
    out = []
    for c in doc.get("chunks", []):
        out.append({
            "chunk_id": c.get("chunk_id"), "chunk_type": c.get("chunk_type"),
            "role": c.get("role"), "pages": c.get("pages", []),
            "section_path": c.get("section_path", []),
            "element_ids": c.get("element_ids", []),
            "char_start": c.get("char_start"), "char_end": c.get("char_end"),
            "token_estimate": c.get("token_estimate"),
            "label": (c.get("metadata") or {}).get("label"),
            "parent_chunk_id": c.get("parent_chunk_id"),
            "text": c.get("text", ""),
        })
    return {"doc_id": doc.get("doc_id"), "stats": doc.get("stats", {}),
            "chunks": out}


@app.get("/api/jobs/{job_id}/graph")
def get_graph(job_id: str, hide: str | None = "VectorPoint,Quantity,Unit,TableKV"):
    """KG nodes/edges for in-browser visualization. `hide` drops bookkeeping
    node types by default to keep the graph readable."""
    data = _read_json(job_id, "knowledge_graph.json")
    hidden = set((hide or "").split(",")) if hide else set()
    keep = {n["id"] for n in data["nodes"] if n.get("ntype") not in hidden}
    nodes = [{"id": n["id"], "type": n.get("ntype"),
              "label": (n.get("label") or n.get("title") or n.get("name")
                        or n.get("citation_key") or n["id"].split(":")[-1]),
              "role": n.get("role"), "text": (n.get("text") or "")[:200]}
             for n in data["nodes"] if n["id"] in keep]
    edges = [{"source": e["source"], "target": e["target"], "rel": e.get("rel")}
             for e in data["edges"]
             if e["source"] in keep and e["target"] in keep]
    return {"nodes": nodes, "edges": edges,
            "hidden_types": sorted(hidden), "n_nodes": len(nodes),
            "n_edges": len(edges)}


# --------------------------------------------------------------------- images

@app.get("/api/jobs/{job_id}/page/{page_no}.png")
def page_image(job_id: str, page_no: int, dpi: int = 120):
    """Render a PDF page to PNG on demand for the overlay viewer. Cached on disk."""
    jd = _runner.job_dir(job_id)
    out = jd / "pages" / f"p{page_no}_{dpi}.png"
    if not out.exists():
        src = next((p for p in jd.glob("source.*")), None)
        if src is None or src.suffix.lower() != ".pdf":
            raise HTTPException(404, "no PDF source for page rendering")
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise HTTPException(500, "PyMuPDF not available for page rendering")
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open(str(src))
        if page_no < 1 or page_no > doc.page_count:
            raise HTTPException(404, f"page {page_no} out of range")
        pix = doc[page_no - 1].get_pixmap(dpi=dpi)
        pix.save(str(out))
    return FileResponse(out, media_type="image/png")


# --------------------------------------------------------------------- download

@app.get("/api/jobs/{job_id}/download/{artifact}")
def download(job_id: str, artifact: str):
    if artifact not in _DOWNLOAD_MIME:
        raise HTTPException(404, f"unknown artifact '{artifact}'. "
                            f"available: {', '.join(_DOWNLOAD_MIME)}")
    fname, mime = _DOWNLOAD_MIME[artifact]
    path = _runner.job_dir(job_id) / fname
    if not path.exists():
        raise HTTPException(404, f"{fname} not generated yet")
    dl = f"{job_id}_{fname}"
    return FileResponse(path, media_type=mime, filename=dl)


@app.get("/api/healthz")
def healthz():
    return {"status": "ok", "embedder": EMBEDDER, "jobs_dir": str(JOBS_DIR)}


# --------------------------------------------------------------------- frontend

_STATIC = Path(__file__).parent / "static"

from fastapi.staticfiles import StaticFiles  # noqa: E402
if _STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")


@app.get("/")
def index():
    idx = _STATIC / "index.html"
    if not idx.exists():
        raise HTTPException(404, "frontend not built")
    return FileResponse(idx, media_type="text/html")


# --------------------------------------------------------------------- helpers

def _sse(payload: dict, event: str | None = None) -> str:
    prefix = f"event: {event}\n" if event else ""
    return f"{prefix}data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _runner_stages():
    from service.pipeline_runner import STAGES
    return STAGES


def _read_json(job_id: str, name: str):
    path = _runner.job_dir(job_id) / name
    if not path.exists():
        raise HTTPException(404, f"{name} not found for job {job_id}")
    return json.loads(path.read_text(encoding="utf-8"))
