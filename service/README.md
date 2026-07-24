# docparse evidence service

**Documents in → retrievable, grounded evidence out.** Field-agnostic:
climate papers, legal contracts, medical records, financial filings, code docs
— the pipeline treats them all as typed elements. It does NOT generate answers;
answer generation (Stage 7) is a separate service that *calls* this one.

## Pipeline
```
/extract  Stage 1-3  parse + enrich          (any file type)
/chunk    Stage 4     typed chunks
/ingest   Stage 5     vectors + knowledge graph  (persisted)
/search   Stage 6     hybrid retrieval → rerank → graph-walk grounding
/healthz              liveness + index stats
```

## Stage 6 — what /search does
1. **Hybrid retrieval** (recall): dense + BM25 sparse, fused by RRF, over-fetched.
2. **Reranking** (precision): cross-encoder reorders candidates. `rerank_alpha`
   blends rerank vs retrieval score (1=pure rerank, 0=retrieval order).
3. **Graph-walk** (grounding): each hit is expanded with its knowledge-graph
   neighbours — the figure a claim cites, the table a row belongs to, the
   section a paragraph sits in — so consumers get provenance-complete evidence.

Every result carries full provenance (doc, pages, section_path, figure/table
label, char offsets, visual_ref) — everything a downstream answer service needs
to cite sources.

## Run — local (no Docker)
```
pip install -r requirements.txt
EMBEDDER=hash RERANKER=lexical uvicorn service.app:app --reload
```
Swagger UI: http://localhost:8000/docs

## Run — Docker (production shape)
```
docker compose up            # api + qdrant + neo4j
```
Set `EMBEDDER=bge-m3` and `RERANKER=ce:BAAI/bge-reranker-v2-m3` (bake into image).

## IMPORTANT: use the Qdrant *server*, not local-file mode, for the service
Local-file Qdrant (`INDEX_DIR/qdrant`) takes an **exclusive lock** — only one
process can open it. That's fine for the CLI and tests, but a service that both
ingests and searches needs concurrent access. Set `QDRANT_URL=http://qdrant:6333`
(docker-compose does this) so ingest and search share one server. The knowledge
graph and sparse encoder persist as files in `INDEX_DIR` regardless.

## Configuration (all env vars)
| Var        | Default | Meaning                                    |
|------------|---------|--------------------------------------------|
| INDEX_DIR  | /data/index | where graph + sparse encoder + cache live |
| QDRANT_URL | (unset) | Qdrant server; unset = local-file (single-proc) |
| EMBEDDER   | auto    | auto \| hash \| bge-m3 \| st:<model>       |
| RERANKER   | auto    | auto \| lexical \| ce:<model> \| off       |

## Example
```
curl -F file=@contract.pdf http://localhost:8000/chunk > chunks.json
curl -F file=@chunks.json  "http://localhost:8000/ingest?tenant=legal"
curl "http://localhost:8000/search?q=termination+clause+notice+period&k=5&tenant=legal"
```

## Evaluate retrieval quality (pick settings by measurement)
```
python -m docparse.vectorization.evaluate INDEX_DIR gold.json \
    --embedder bge-m3 --reranker ce:BAAI/bge-reranker-v2-m3 --k 10
# diff two runs to compare knobs:
python -m docparse.vectorization.evaluate INDEX_DIR gold.json --set rerank_alpha=0.5
```
