# CorpusRetrieve Web — Stage 1 (developer mode)

A web front-end over the evidence pipeline: **upload a document → watch it flow
through all six stages live → inspect any chunk's source region → download
portable chunks + embeddings + knowledge graph** to load into your own stack.

This is Stage 1 of the rollout: the developer/inspection mode. It validates the
core wedge — full-pipeline transparency + vendor-neutral portable artifacts.

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
2. **Watch the six-stage spine** stream live over SSE: parse → extract → enrich
   → chunk → embed → graph, each showing counts and timing.
3. **Inspect chunks** (left): filter by type; click one to select it.
4. **Trace provenance** (center): the selected chunk jumps to its source page and
   highlights the exact regions it came from, drawn as bounding boxes over the
   rendered page. Click any box to inspect that element.
5. **Read metadata** (right): chunk id, role, section path, pages, char span,
   source element ids, full text.
6. **Download portable artifacts**: chunks.json, embeddings.jsonl
   (`{id, vector, sparse, payload}` — loads into any vector DB), and the KG as
   JSON / Cypher / GraphML. Copy-paste load snippets for Qdrant / Neo4j / pgvector
   are shown inline.

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
