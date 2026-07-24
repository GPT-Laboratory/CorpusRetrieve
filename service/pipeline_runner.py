"""Stage-1 web pipeline runner.

Runs the six pipeline stages as discrete, observable steps and yields a
progress event after each, so the HTTP layer can stream them over SSE. The
same run produces, on disk under the job directory, every artifact the UI
needs: the extraction JSON (with per-element bbox provenance), chunks.json,
the vector records, and the knowledge-graph exports.

Design notes
------------
* One job = one directory (``JOBS_DIR/<job_id>``). Everything is a file, so a
  restarted process can still serve artifacts and the frontend can download
  them directly.
* Stages are explicit and named so the UI's six-stage spine maps 1:1 to what
  actually ran. Each stage reports counts + timing, not just "done".
* Embeddings use the offline HashingEmbedder by default so the whole thing
  runs with zero downloads for the demo; swap via EMBEDDER env.
* The vector store is exported as a portable JSONL of {id, vector, payload}
  records (loadable into any vector DB) rather than only living inside Qdrant —
  that is the whole point of the "portable artifacts" thesis.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from docparse import ExtractionConfig, chunk_result, extract
from docparse.vectorization.canonical import normalize_chunk, payload_for
from docparse.vectorization.embedding import BM25SparseEncoder, resolve_embedder
from docparse.vectorization.graph import KnowledgeGraph
from docparse.vectorization.vectorstore import COLLECTION_OF, point_uuid

STAGES = ["parse", "extract", "enrich", "chunk", "embed", "graph"]


@dataclass
class StageEvent:
    job_id: str
    stage: str
    status: str                       # started | done | error
    index: int                        # 0..5
    total: int = len(STAGES)
    detail: dict[str, Any] = field(default_factory=dict)
    elapsed_ms: int | None = None
    message: str | None = None

    def to_json(self) -> str:
        return json.dumps({k: v for k, v in asdict(self).items() if v is not None})


def new_job_id() -> str:
    return uuid.uuid4().hex[:12]


class PipelineRunner:
    """Executes one document through the pipeline, yielding StageEvents."""

    def __init__(self, jobs_dir: str | Path, embedder_spec: str = "hash",
                 tenant: str = "default", embed_version: str = "v1",
                 describe_figures: bool = False):
        self.jobs_dir = Path(jobs_dir)
        self.embedder_spec = embedder_spec
        self.tenant = tenant
        self.embed_version = embed_version
        self.describe_figures = describe_figures

    def job_dir(self, job_id: str) -> Path:
        d = self.jobs_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def run(self, job_id: str, source_path: str,
            describe_figures: bool | None = None) -> Iterator[StageEvent]:
        jd = self.job_dir(job_id)
        emb = resolve_embedder(self.embedder_spec)
        want_vlm = self.describe_figures if describe_figures is None else describe_figures

        def ev(stage, status, i, **kw):
            return StageEvent(job_id=job_id, stage=stage, status=status, index=i, **kw)

        # ---- 1. parse (format detection + backend dispatch) ---------------
        t = time.time()
        yield ev("parse", "started", 0)
        cfg = ExtractionConfig(table_serializations=("markdown", "kv"),
                               assets_dir=str(jd / "assets"),
                               emit_page_images=False)
        try:
            result = extract(source_path, cfg)
        except Exception as exc:  # noqa: BLE001
            yield ev("parse", "error", 0, message=str(exc))
            return
        yield ev("parse", "done", 0, elapsed_ms=_ms(t),
                 detail={"doc_type": result.doc_type, "mime": result.mime,
                         "pages": result.page_count,
                         "backend": result.stats.get("backend")})

        # ---- 2. extract (typed elements + provenance) ---------------------
        t = time.time()
        yield ev("extract", "started", 1)
        (jd / "extraction.json").write_text(
            json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
        counts = result.counts()
        with_bbox = sum(1 for e in result.elements if e.bbox)
        yield ev("extract", "done", 1, elapsed_ms=_ms(t),
                 detail={"elements": len(result.elements), "by_kind": counts,
                         "with_bbox": with_bbox,
                         "page_dims": bool(result.stats.get("page_dims"))})

        # ---- 3. enrich (roles / claims / citations / quantities [+ VLM]) --
        t = time.time()
        yield ev("enrich", "started", 2)
        vlm_detail = {}
        if want_vlm:
            figs = sum(1 for e in result.elements if e.kind == "figure")
            try:
                from docparse.enrich import describe_figures, make_anthropic_describer
                n = describe_figures(result, make_anthropic_describer())
                vlm_detail = {"vlm": "claude", "figures_described": n,
                              "figures_total": figs}
            except Exception as exc:  # noqa: BLE001 — missing key / package / crops
                # transparent skip: report WHY, never silently omit
                vlm_detail = {"vlm": "skipped", "reason": str(exc)[:120],
                              "figures_total": figs}
        roles = {}
        for e in result.elements:
            r = e.metadata.get("role")
            if r:
                roles[r] = roles.get(r, 0) + 1
        yield ev("enrich", "done", 2, elapsed_ms=_ms(t),
                 detail={"roles": roles,
                         "claims": len(result.stats.get("claims", []) or []),
                         "citations": len(result.stats.get("citation_graph", {}) or {}),
                         **vlm_detail})

        # ---- 4. chunk (structure-aware typed chunks) ----------------------
        t = time.time()
        yield ev("chunk", "started", 3)
        chunks = chunk_result(result)
        chunks_payload = [chunks.to_dict()]
        (jd / "chunks.json").write_text(
            json.dumps(chunks_payload, ensure_ascii=False), encoding="utf-8")
        yield ev("chunk", "done", 3, elapsed_ms=_ms(t),
                 detail={"chunks": chunks.stats.get("n_chunks"),
                         "by_type": chunks.stats.get("by_type"),
                         "tokens_by_type": chunks.stats.get("tokens_by_type")})

        # ---- 5. embed (dense + sparse; portable JSONL export) -------------
        t = time.time()
        yield ev("embed", "started", 4)
        norm = [normalize_chunk(c) for c in chunks.to_dict()["chunks"]]
        sparse = BM25SparseEncoder().fit(c["canonical_text"] for c in norm)
        doc_id = chunks.doc_id
        records = []
        texts = [c["canonical_text"] for c in norm]
        vecs = emb.encode(texts)
        for c, v in zip(norm, vecs):
            composite = (f"{self.tenant}:{doc_id}:{c['chunk_type']}:"
                         f"{c['chunk_id']}:{emb.name}:{self.embed_version}")
            si, sv = sparse.encode(c["canonical_text"])
            records.append({
                "id": point_uuid(composite), "composite_id": composite,
                "collection": COLLECTION_OF.get(c["chunk_type"], "sections"),
                "vector": v,
                "sparse": {"indices": si, "values": sv},
                "payload": payload_for(c, chunks.to_dict(), emb.name,
                                       self.embed_version, self.tenant),
            })
        # portable, DB-agnostic export
        with (jd / "embeddings.jsonl").open("w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        sparse.save(jd / "sparse_encoder.json")
        (jd / "embeddings_meta.json").write_text(json.dumps({
            "model": emb.name, "dim": emb.dim, "metric": "cosine",
            "count": len(records), "version": self.embed_version,
            "vector_field": "text_dense", "sparse_field": "text_sparse",
        }), encoding="utf-8")
        yield ev("embed", "done", 4, elapsed_ms=_ms(t),
                 detail={"vectors": len(records), "model": emb.name,
                         "dim": emb.dim, "metric": "cosine"})

        # ---- 6. graph (knowledge graph + exports) -------------------------
        t = time.time()
        yield ev("graph", "started", 5)
        kg = KnowledgeGraph()
        doc_norm = dict(chunks.to_dict())
        doc_norm["chunks"] = norm
        gstats = kg.build_from_chunks(doc_norm, self.tenant, emb.name,
                                      self.embed_version)
        exports = kg.export(jd)
        yield ev("graph", "done", 5, elapsed_ms=_ms(t),
                 detail={"nodes": gstats["nodes"], "edges": gstats["edges"],
                         "node_types": gstats["node_types"],
                         "edge_types": gstats["edge_types"]})

        # ---- manifest: what the UI/consumer can fetch ---------------------
        manifest = {
            "job_id": job_id, "doc_id": doc_id, "source": Path(source_path).name,
            "stages_completed": STAGES,
            "artifacts": {
                "extraction": "extraction.json",
                "chunks": "chunks.json",
                "embeddings": "embeddings.jsonl",
                "embeddings_meta": "embeddings_meta.json",
                "sparse_encoder": "sparse_encoder.json",
                "graph_json": "knowledge_graph.json",
                "graph_graphml": "knowledge_graph.graphml",
                "graph_cypher": "knowledge_graph.cypher",
            },
            "summary": {
                "pages": result.page_count,
                "elements": len(result.elements),
                "chunks": chunks.stats.get("n_chunks"),
                "vectors": len(records),
                "graph_nodes": gstats["nodes"], "graph_edges": gstats["edges"],
            },
        }
        (jd / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False,
                                                     indent=2), encoding="utf-8")


def _ms(t0: float) -> int:
    return int((time.time() - t0) * 1000)
