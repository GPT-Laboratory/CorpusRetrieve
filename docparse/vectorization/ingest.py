"""Stage 5 orchestrator: chunks.json -> vector DB + knowledge graph.

Pipeline (mirrors the report's ingestion diagram):
  load typed chunks -> canonicalize (2 text views + structured quantities)
  -> fit corpus sparse encoder -> embed dense in token-length buckets
  -> upsert to per-type collections (idempotent UUID5 point ids)
  -> project structure to the knowledge graph with vector-point links
  -> validation: self-retrieval smoke test + cross-store consistency
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from docparse.vectorization.canonical import normalize_chunk, payload_for
from docparse.vectorization.embedding import BM25SparseEncoder, Embedder, resolve_embedder
from docparse.vectorization.graph import KnowledgeGraph
from docparse.vectorization.vectorstore import COLLECTION_OF, VectorStore


@dataclass
class IngestConfig:
    tenant: str = "default"
    embedder: str = "auto"            # hash | bge-m3 | st:<model> | auto
    embed_version: str = "v1"
    db_location: str = ":memory:"     # or a directory path, or server URL later
    graph_outdir: str | None = None
    batch_size: int = 64
    include_references: bool = True   # embed them; retrieval excludes by default


@dataclass
class IngestResult:
    doc_id: str
    collections: dict[str, int] = field(default_factory=dict)
    graph_stats: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    exports: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    doc_ids: list[str] = field(default_factory=list)   # all documents actually ingested
    warnings: list[str] = field(default_factory=list)


def _buckets(chunks: list[dict], size: int) -> list[list[dict]]:
    """Batch by token-length bucket so short claims don't queue behind long
    sections (report: 'batch by model family and token length bucket')."""
    order = sorted(chunks, key=lambda c: c.get("token_estimate", 0))
    return [order[i:i + size] for i in range(0, len(order), size)]


def ingest(chunks_path: str | Path, cfg: IngestConfig | None = None,
           embedder: Embedder | None = None
           ) -> tuple[IngestResult, VectorStore, KnowledgeGraph, BM25SparseEncoder]:
    cfg = cfg or IngestConfig()
    t0 = time.time()
    data = json.loads(Path(chunks_path).read_text(encoding="utf-8"))
    # BUG-8 fix: a chunks.json from Stage 4 batch processing (e.g. a directory
    # or zip of several source documents) is a *list* of per-document chunk
    # bundles. This used to silently take only data[0] and drop every other
    # document with no warning at all -- an operator ingesting 50 documents
    # would see a normal-looking successful report with 49 documents simply
    # gone. Every document in the list is now actually ingested; a warning is
    # still recorded (not just silently handled) so this remains visible.
    docs = data if isinstance(data, list) else [data]
    warnings: list[str] = []
    if len(docs) > 1:
        warnings.append(
            f"{len(docs)} documents were present in {chunks_path}; all {len(docs)} "
            "were ingested into this shared vector store / knowledge graph."
        )
    doc = docs[0]

    all_chunks: list[dict] = []
    per_doc_chunks: list[tuple[dict, list[dict]]] = []
    for d in docs:
        d_chunks = [normalize_chunk(c) for c in d["chunks"]]
        if not cfg.include_references:
            d_chunks = [c for c in d_chunks if c["chunk_type"] != "reference"]
        per_doc_chunks.append((d, d_chunks))
        all_chunks.extend(d_chunks)
    chunks = all_chunks

    emb = embedder or resolve_embedder(cfg.embedder)
    # Fit the sparse (BM25) encoder on the full combined corpus across all
    # documents -- correct IDF statistics need the whole corpus, not just the
    # first document's chunks.
    sparse = BM25SparseEncoder().fit(c["canonical_text"] for c in chunks)
    store = VectorStore(cfg.db_location, dense_dim=emb.dim)
    store.ensure_collections()
    t1 = time.time()

    records = []
    for batch in _buckets(chunks, cfg.batch_size):
        vecs = emb.encode([c["canonical_text"] for c in batch])
        for c, v in zip(batch, vecs):
            composite = (f"{cfg.tenant}:{c['doc_id']}:{c['chunk_type']}:"
                         f"{c['chunk_id']}:{emb.name}:{cfg.embed_version}")
            records.append({
                "collection": COLLECTION_OF.get(c["chunk_type"], "sections"),
                "composite_id": composite,
                "dense": v,
                "sparse": sparse.encode(c["canonical_text"]),
                "payload": payload_for(c, doc, emb.name, cfg.embed_version, cfg.tenant),
            })
    counts = store.upsert_chunks(records)
    t2 = time.time()

    # Build the knowledge graph once per document (not once over the merged
    # chunk list) so each document gets its own correctly-scoped Paper node
    # and section tree, rather than every document's chunks being incorrectly
    # attributed to a single Paper node. KnowledgeGraph.build_from_chunks()'s
    # returned stats are cumulative totals over the whole graph object, so the
    # final call's graph_stats already reflects the combined total across all
    # documents processed so far.
    kg = KnowledgeGraph()
    graph_stats: dict = {}
    for d, d_chunks in per_doc_chunks:
        doc_norm = dict(d)
        doc_norm["chunks"] = d_chunks
        graph_stats = kg.build_from_chunks(doc_norm, cfg.tenant, emb.name, cfg.embed_version)
    t3 = time.time()

    res = IngestResult(doc_id=doc["doc_id"], collections=counts, graph_stats=graph_stats,
                       doc_ids=[d["doc_id"] for d in docs], warnings=warnings)
    res.validation = validate(store, kg, chunks, emb, sparse)
    if cfg.graph_outdir:
        res.exports = kg.export(cfg.graph_outdir)
        sparse.save(Path(cfg.graph_outdir) / "sparse_encoder.json")
        # persist a slim chunk cache so a restarted service can restore
        # neighbour text for graph expansion without reparsing source docs
        import json as _json
        cache = [{"chunk_id": c["chunk_id"], "text": c.get("text", ""),
                  "pages": c.get("pages", []),
                  "metadata": {"label": (c.get("metadata") or {}).get("label")}}
                 for c in chunks]
        (Path(cfg.graph_outdir) / "chunk_cache.json").write_text(
            _json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    res.timings = {"setup_s": round(t1 - t0, 2), "embed_upsert_s": round(t2 - t1, 2),
                   "graph_s": round(t3 - t2, 2)}
    return res, store, kg, sparse


# ------------------------------------------------------------------- validation

def validate(store: VectorStore, kg: KnowledgeGraph, chunks: list[dict],
             emb: Embedder, sparse: BM25SparseEncoder) -> dict:
    """The report's rule — evaluate retrieval before trusting the stack.
    Corpus-native smoke tests, no external gold set required."""
    out: dict = {}

    # 1. self-retrieval: querying a chunk's own canonical text must return it
    #    top-1 in its collection (embedding + upsert + search plumbing intact)
    probe = [c for c in chunks if c["chunk_type"] in
             ("claim", "section_text", "figure", "table_row_group")][:20]
    hit1 = 0
    for c in probe:
        v = emb.encode([c["canonical_text"]])[0]
        hits = store.hybrid_search(COLLECTION_OF[c["chunk_type"]], v,
                                   sparse.encode(c["canonical_text"]), limit=1,
                                   include_references=True)
        hit1 += bool(hits and hits[0].chunk_id == c["chunk_id"])
    out["self_retrieval_top1"] = f"{hit1}/{len(probe)}"

    # 2. references excluded from default retrieval
    ref = next((c for c in chunks if c["chunk_type"] == "reference"), None)
    if ref is not None:
        v = emb.encode([ref["canonical_text"]])[0]
        hits = store.hybrid_search("references", v, sparse.encode(ref["canonical_text"]),
                                   limit=3)                     # default: exclude
        out["references_hidden_by_default"] = not any(
            h.chunk_id == ref["chunk_id"] for h in hits)

    # 3. vector<->graph consistency: every chunk node links a vector point
    linked = sum(1 for _, _, k in kg.g.edges(keys=True)
                 if k == "NODE_LINKS_VECTOR_POINT")
    out["graph_vector_links"] = linked
    out["graph_vector_links_complete"] = linked >= len(chunks)

    # 4. every claim has at least one support edge (chunk/figure/table)
    claims = [n for n, d in kg.g.nodes(data=True) if d.get("ntype") == "Claim"]
    out["claims_with_support"] = (
        f"{sum(1 for n in claims if kg.supports_of(n))}/{len(claims)}")
    return out


# ------------------------------------------------------------------- retrieval

def hybrid_query(store: VectorStore, emb: Embedder, sparse: BM25SparseEncoder,
                 query: str, collections: list[str] | None = None,
                 limit: int = 5, **kw):
    dense = emb.encode([query])[0]
    return store.search_all(dense, sparse.encode(query), limit_per=limit,
                            collections=collections, **kw)
