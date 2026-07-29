"""CLI: python -m docparse.stage5 chunks.json --out stage5_out [--embedder hash]
Ingests typed chunks into a local Qdrant DB + knowledge graph, runs
validation, and optionally answers ad-hoc queries with --query.
"""
import argparse, json, sys
from docparse.vectorization.ingest import IngestConfig, hybrid_query, ingest


def main() -> int:
    ap = argparse.ArgumentParser(prog="docparse.stage5")
    ap.add_argument("chunks", help="chunks.json from Stage 4")
    ap.add_argument("--out", default="stage5_out",
                    help="output dir: qdrant db + graph exports + sparse encoder")
    ap.add_argument("--embedder", default="auto",
                    help="auto | hash | bge-m3 | st:<model_name>")
    ap.add_argument("--tenant", default="default")
    ap.add_argument("--query", action="append", default=[],
                    help="run hybrid queries after ingest (repeatable)")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    cfg = IngestConfig(tenant=args.tenant, embedder=args.embedder,
                       db_location=f"{args.out}/qdrant", graph_outdir=args.out)
    res, store, kg, sparse = ingest(args.chunks, cfg)
    from docparse.vectorization.ingest import resolve_embedder
    emb = resolve_embedder(args.embedder)

    print(json.dumps({"doc_id": res.doc_id, "doc_ids": res.doc_ids,
                      "warnings": res.warnings, "collections": res.collections,
                      "graph": res.graph_stats, "validation": res.validation,
                      "exports": res.exports, "timings": res.timings},
                     ensure_ascii=False, indent=2))
    for q in args.query:
        print(f"\n=== query: {q}")
        for h in hybrid_query(store, emb, sparse, q, limit=args.limit)[:args.limit]:
            print(f"  [{h.chunk_type:>15}] {h.score:.3f}  "
                  f"{(h.payload.get('raw_text') or '')[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
