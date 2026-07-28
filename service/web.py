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

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from service.pipeline_runner import PipelineRunner, new_job_id

JOBS_DIR = Path(os.environ.get(
    "JOBS_DIR", str(Path(tempfile.gettempdir()) / "corpusretrieve" / "jobs")))
EMBEDDER = os.environ.get("EMBEDDER", "hash")
JOBS_DIR.mkdir(parents=True, exist_ok=True)

# document types the pipeline can ingest; anything else in a folder/zip is skipped
SUPPORTED_EXT = {".pdf", ".docx", ".html", ".htm", ".csv", ".xlsx",
                 ".json", ".txt", ".png", ".jpg", ".jpeg"}
# noise that shows up in folders/zips and should never become a job
_IGNORE_NAMES = {".ds_store", "thumbs.db"}

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


# --------------------------------------------------------------- batch upload

BATCHES_DIR = JOBS_DIR.parent / "batches"
BATCHES_DIR.mkdir(parents=True, exist_ok=True)


def _spawn_job(filename: str, data: bytes) -> dict:
    """Create one job dir from raw bytes; returns {job_id, filename}."""
    job_id = new_job_id()
    jd = _runner.job_dir(job_id)
    suffix = Path(filename).suffix or ".bin"
    (jd / f"source{suffix}").write_bytes(data)
    (jd / "status.json").write_text(json.dumps(
        {"job_id": job_id, "filename": filename, "state": "uploaded"}),
        encoding="utf-8")
    return {"job_id": job_id, "filename": filename}


def _is_supported(name: str) -> bool:
    base = Path(name).name.lower()
    if base in _IGNORE_NAMES or base.startswith("._"):
        return False
    return Path(name).suffix.lower() in SUPPORTED_EXT


def _expand_zip(data: bytes) -> list[tuple[str, bytes]]:
    """Return [(filename, bytes)] for supported members of a zip, recursively.
    Nested zips are expanded too; junk and unsupported files are skipped."""
    out: list[tuple[str, bytes]] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        return out
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename
        if Path(name).name.lower().startswith("__macosx"):
            continue
        try:
            member = zf.read(info)
        except Exception:
            continue
        if name.lower().endswith(".zip"):
            out.extend(_expand_zip(member))
        elif _is_supported(name):
            out.append((Path(name).name, member))
    return out


@app.post("/api/batches")
async def create_batch(files: List[UploadFile] = File(...)):
    """Accept multiple files, a folder (browser sends its files), and/or zips.
    Each supported document becomes its own job; zips are expanded. Returns the
    batch id and the list of spawned jobs so the client can process each."""
    batch_id = new_job_id()
    jobs: list[dict] = []
    skipped: list[str] = []

    for f in files:
        raw = await f.read()
        name = f.filename or "upload.bin"
        if name.lower().endswith(".zip"):
            members = _expand_zip(raw)
            if not members:
                skipped.append(f"{name} (empty or no supported files)")
            for mname, mdata in members:
                jobs.append(_spawn_job(mname, mdata))
        elif _is_supported(name):
            jobs.append(_spawn_job(name, raw))
        else:
            skipped.append(name)

    if not jobs:
        raise HTTPException(
            400, f"no supported documents found. Supported: "
                 f"{', '.join(sorted(SUPPORTED_EXT))}. Skipped: {skipped}")

    (BATCHES_DIR / f"{batch_id}.json").write_text(json.dumps(
        {"batch_id": batch_id, "jobs": jobs, "skipped": skipped}),
        encoding="utf-8")
    return {"batch_id": batch_id, "count": len(jobs), "jobs": jobs,
            "skipped": skipped}


@app.get("/api/batches/{batch_id}")
def get_batch(batch_id: str):
    """Batch membership + per-job completion status (done if manifest exists)."""
    meta_path = BATCHES_DIR / f"{batch_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, "batch not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    jobs = []
    for j in meta["jobs"]:
        jd = _runner.job_dir(j["job_id"])
        manifest = jd / "manifest.json"
        done = manifest.exists()
        summary = None
        if done:
            summary = json.loads(manifest.read_text(encoding="utf-8")).get("summary")
        jobs.append({**j, "done": done, "summary": summary})
    return {"batch_id": batch_id, "jobs": jobs, "skipped": meta.get("skipped", [])}


@app.get("/api/batches/{batch_id}/download")
def download_batch(batch_id: str):
    """Zip every finished job's artifacts into one archive, one folder per doc."""
    meta_path = BATCHES_DIR / f"{batch_id}.json"
    if not meta_path.exists():
        raise HTTPException(404, "batch not found")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    buf = io.BytesIO()
    n_files = 0
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for j in meta["jobs"]:
            jd = _runner.job_dir(j["job_id"])
            if not (jd / "manifest.json").exists():
                continue
            stem = Path(j["filename"]).stem or j["job_id"]
            folder = f"{stem}_{j['job_id'][:6]}"
            for art in ("chunks.json", "embeddings.jsonl",
                        "knowledge_graph.json", "knowledge_graph.cypher",
                        "knowledge_graph.graphml", "extraction.json"):
                p = jd / art
                if p.exists():
                    zf.write(p, f"{folder}/{art}")
                    n_files += 1
    if n_files == 0:
        raise HTTPException(409, "no finished jobs to download yet")
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition":
                 f'attachment; filename="corpusretrieve_batch_{batch_id[:6]}.zip"'})


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
    node types by default to keep the graph readable. Nodes carry degree and
    full attributes so the explorer can inspect any node without another call."""
    data = _read_json(job_id, "knowledge_graph.json")
    hidden = set(x for x in (hide or "").split(",") if x) if hide else set()
    keep = {n["id"] for n in data["nodes"] if n.get("ntype") not in hidden}
    edges = [{"source": e["source"], "target": e["target"], "rel": e.get("rel")}
             for e in data["edges"]
             if e["source"] in keep and e["target"] in keep]
    degree: dict[str, int] = {}
    for e in edges:
        degree[e["source"]] = degree.get(e["source"], 0) + 1
        degree[e["target"]] = degree.get(e["target"], 0) + 1
    nodes = [{"id": n["id"], "type": n.get("ntype"),
              "label": (n.get("label") or n.get("title") or n.get("name")
                        or n.get("citation_key") or n["id"].split(":")[-1]),
              "role": n.get("role"), "text": (n.get("text") or "")[:280],
              "pages": n.get("pages"), "degree": degree.get(n["id"], 0),
              "attrs": {k: v for k, v in n.items()
                        if k not in ("id", "ntype", "text") and v not in (None, "", [])}}
             for n in data["nodes"] if n["id"] in keep]
    # all types present in the full graph (so the UI can offer show/hide toggles)
    all_types: dict[str, int] = {}
    for n in data["nodes"]:
        t = n.get("ntype", "?")
        all_types[t] = all_types.get(t, 0) + 1
    return {"nodes": nodes, "edges": edges,
            "hidden_types": sorted(hidden), "all_types": all_types,
            "n_nodes": len(nodes), "n_edges": len(edges)}


# --------------------------------------------------------------------- images

@app.get("/api/jobs/{job_id}/projection")
def get_projection(job_id: str):
    """2D projection of the chunk embeddings for the embedding-map view.
    Uses UMAP when available (best neighbourhood structure), else falls back to
    PCA (fast, deterministic, no extra dependency). Cached on disk per job.
    Points are colored/labeled by chunk type so the map ties back to the
    typed-chunk model."""
    jd = _runner.job_dir(job_id)
    cache = jd / "projection.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    src = jd / "embeddings.jsonl"
    if not src.exists():
        raise HTTPException(404, "embeddings not generated yet")
    recs = [json.loads(l) for l in src.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not recs:
        raise HTTPException(404, "no embedding records")

    import numpy as np
    X = np.array([r["vector"] for r in recs], dtype="float32")
    method = "pca"
    if len(recs) >= 4:
        try:
            import umap  # type: ignore
            n_nb = min(15, len(recs) - 1)
            xy = umap.UMAP(n_neighbors=n_nb, min_dist=0.1,
                           n_components=2, random_state=42).fit_transform(X)
            method = "umap"
        except Exception:
            from sklearn.decomposition import PCA
            xy = PCA(n_components=2, random_state=42).fit_transform(X)
    else:
        xy = np.zeros((len(recs), 2), dtype="float32")

    xy = np.asarray(xy, dtype="float32")
    lo, hi = xy.min(0), xy.max(0)
    span = np.where((hi - lo) == 0, 1.0, hi - lo)
    norm = (xy - lo) / span                      # 0..1 for easy client scaling

    points = []
    for r, (x, y) in zip(recs, norm):
        p = r.get("payload", {})
        points.append({
            "id": r["id"],
            "chunk_id": p.get("chunk_id"),
            "type": p.get("chunk_type"),
            "x": round(float(x), 4), "y": round(float(y), 4),
            "pages": p.get("pages"),
            "text": (p.get("raw_text") or "")[:160],
        })
    out = {"method": method, "n": len(points), "points": points}
    cache.write_text(json.dumps(out), encoding="utf-8")
    return out


@app.get("/api/jobs/{job_id}/page/{page_no}.png")
def page_image(job_id: str, page_no: int, dpi: int = 120):
    """Render a PDF page to PNG on demand for the overlay viewer. Cached on disk."""
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
