"""Stage 6 evaluation — measure retrieval quality, don't guess it.

The report is emphatic: pick indexing/rerank settings by measuring recall@k,
MRR, and nDCG on a labeled query set, not by intuition. This harness does that
against a persisted index, so every knob (embedder, reranker, alpha, expand,
hops) becomes a measured decision.

Gold format (JSON): a list of queries, each with the chunk_ids that should be
retrieved. Field-agnostic — you write these for whatever corpus you loaded.

    [
      {"query": "how are cold days defined",
       "relevant": ["climate_paper::sec::0007", "..."]},
      ...
    ]

CLI:
    python -m docparse.vectorization.evaluate INDEX_DIR gold.json \\
        --embedder hash --reranker lexical --k 10 [--set rerank_alpha=0.5]

Run twice with different --set overrides and diff the numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from docparse.vectorization.retriever import RetrieveConfig, load_retriever


def _dcg(rels: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(rels))


def evaluate(index_dir: str, gold_path: str, embedder: str = "auto",
             reranker: str = "auto", k: int = 10, overrides: dict | None = None
             ) -> dict:
    gold = json.loads(Path(gold_path).read_text(encoding="utf-8"))
    r = load_retriever(index_dir, embedder=embedder, reranker=reranker)

    base = RetrieveConfig(limit=k, expand=False)
    for key, val in (overrides or {}).items():
        setattr(base, key, _coerce(val))

    recalls, mrrs, ndcgs, hits_at1 = [], [], [], []
    per_query = []
    for item in gold:
        relevant = set(item["relevant"])
        if not relevant:
            continue
        bundles = r.retrieve(item["query"], base)
        got = [b.hit.chunk_id for b in bundles]

        found = [1 if cid in relevant else 0 for cid in got]
        recall = sum(found) / len(relevant)
        rr = next((1 / (i + 1) for i, f in enumerate(found) if f), 0.0)
        idcg = _dcg(sorted([1] * len(relevant), reverse=True)[:k]) or 1.0
        ndcg = _dcg(found) / idcg

        recalls.append(recall); mrrs.append(rr); ndcgs.append(ndcg)
        hits_at1.append(found[0] if found else 0)
        per_query.append({"query": item["query"][:60], "recall": round(recall, 3),
                          "mrr": round(rr, 3), "ndcg": round(ndcg, 3),
                          "top": got[:3]})

    n = len(recalls) or 1
    return {
        "n_queries": len(recalls),
        "config": {"embedder": embedder, "reranker": reranker, "k": k,
                   **(overrides or {})},
        "metrics": {
            f"recall@{k}": round(sum(recalls) / n, 4),
            "mrr": round(sum(mrrs) / n, 4),
            f"ndcg@{k}": round(sum(ndcgs) / n, 4),
            "hit@1": round(sum(hits_at1) / n, 4),
        },
        "per_query": per_query,
    }


def _coerce(v: str):
    if isinstance(v, str):
        low = v.lower()
        if low in ("true", "false"):
            return low == "true"
        try:
            return int(v)
        except ValueError:
            try:
                return float(v)
            except ValueError:
                return v
    return v


def main() -> int:
    ap = argparse.ArgumentParser(prog="docparse.vectorization.evaluate")
    ap.add_argument("index_dir")
    ap.add_argument("gold")
    ap.add_argument("--embedder", default="auto")
    ap.add_argument("--reranker", default="auto")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--set", action="append", default=[],
                    metavar="key=val", help="override a RetrieveConfig field")
    ap.add_argument("--json", help="write full report here")
    args = ap.parse_args()

    overrides = dict(kv.split("=", 1) for kv in args.set)
    report = evaluate(args.index_dir, args.gold, args.embedder,
                      args.reranker, args.k, overrides)
    print(json.dumps(report["metrics"], indent=2))
    print(f"over {report['n_queries']} queries | config {report['config']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
