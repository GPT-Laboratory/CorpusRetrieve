# CorpusRetrieve Web — the glass-box evidence layer

**Vendor-neutral. Portable chunks + embeddings + knowledge graph.**

Upload a document and watch it become retrievable, grounded evidence — with
nothing locked to us. Three modes serve three audiences from the same run:

- **Learn mode** (demo / onboarding): a guided, scroll-driven story — "how a
  document becomes evidence" — with a sticky animation that transforms the
  document through all six stages as you scroll, ending in an interactive
  sandbox (drag chunk size, watch chunk count and the precision/context
  trade-off update live) and one-click downloads.
- **Explain mode** (non-technical): a plain-language, step-by-step walkthrough
  of what happened to *your* document at each stage. Designed so a
  non-technical user can explain, in their own words, what each stage does
  after a single walkthrough.
- **Inspect mode** (developer / debugging): the full pipeline spine with live
  timings, the source document with bounding-box provenance overlays, a
  typed-chunk map, an interactive knowledge-graph explorer, a 2D embedding map
  (UMAP/PCA), per-chunk inspection, and portable-artifact downloads with
  copy-paste load snippets.

The mode toggle is persistent (remembered across sessions).

> The 2D embedding map uses UMAP when `umap-learn` is installed (best
> neighbourhood structure) and falls back to PCA otherwise — no required extra
> dependency. `pip install umap-learn` to upgrade the projection.

## Run
```
pip install -r requirements.txt          # includes fastapi, uvicorn, multipart
EMBEDDER=hash uvicorn service.web:app --reload --port 8000
```
Open http://localhost:8000

- `EMBEDDER=hash` (default) runs fully offline. Use `bge-m3` for real embeddings
  (`pip install FlagEmbedding`, ~2 GB first download).
- Jobs are written to `JOBS_DIR` (default `/tmp/docparse_jobs`); each job is a
  self-contained folder of artifacts.

## What you can do
1. **Upload** a PDF (or DOCX/HTML/XLSX/CSV/JSON/image/text) — drag-drop or button.
   You can also upload **multiple files at once, a whole folder** (the "Folder"
   button, recurses into subfolders), or a **.zip** (expanded server-side,
   nested zips included). Junk like `.DS_Store`/`Thumbs.db`/`__MACOSX` is
   skipped. When more than one document is uploaded it becomes a **batch**: each
   file runs through the pipeline and appears in a switcher strip (amber dot =
   processing, green = done). Click any finished file to open it, or use
   **Download all** to get one zip with a folder of artifacts per document.
   Batch API: `POST /api/batches`, `GET /api/batches/{id}`,
   `GET /api/batches/{id}/download`.
2. **Choose your view** with the persistent Explain ↔ Inspect toggle.
3. **Explain mode**: read the six-stage plain-language walkthrough of your
   document, then download the results (chunks / embeddings / graph).
4. **Inspect mode** gives you all of:
   - the six-stage spine streaming live over SSE with counts + timing;
   - **Source**: the rendered page with bounding-box provenance overlays —
     click a chunk to jump to and highlight its exact source region;
   - **Typed-chunk map**: every chunk as a colored span (color = type, width ∝
     tokens), metadata on hover, click to open in the source view;
   - **Knowledge graph**: an interactive Sigma.js explorer — click any node to
     inspect it and its relationships, toggle node types via the legend, dim to
     a selected chunk's neighbourhood, and export PNG / JSON / Cypher;
   - **Inspector**: chunk id, role, section path, pages, char span, source
     element ids, full text;
   - **Portable artifacts**: chunks.json, embeddings.jsonl, and the KG as
     JSON / Cypher / GraphML, with copy-paste load snippets for Qdrant / Neo4j /
     pgvector.

## API (also usable headless)
```
POST /api/jobs                      upload -> {job_id}
GET  /api/jobs/{id}/stream          SSE six-stage progress
GET  /api/jobs/{id}/manifest        artifacts + summary
GET  /api/jobs/{id}/elements?page=1 typed elements + bbox + page_dims
GET  /api/jobs/{id}/chunks          chunks + element/page provenance
GET  /api/jobs/{id}/page/{n}.png    rendered page (PDF sources)
GET  /api/jobs/{id}/graph           KG nodes/edges for visualization
GET  /api/jobs/{id}/download/{art}  chunks|embeddings|graph_json|graph_cypher|graph_graphml|extraction
```

## Portability (the go/no-go benchmark, tested in tests/test_web.py)
- `embeddings.jsonl` loads into Qdrant with zero hand-editing; payloads carry
  provenance (doc_id, pages, chunk_type, section_path, bbox refs).
- `knowledge_graph.cypher` is idempotent (MERGE-based), one statement per line,
  brace/quote balanced — `cypher-shell < knowledge_graph.cypher` just works.

## Note on the vector DB
The web app writes portable files; it does not require a running Qdrant. For the
separate always-on evidence *service* (Stage 6 retrieval), use the Qdrant server
(see service/README.md) — local-file Qdrant locks to one process.
