"""Stage 6 — reranking.

First-stage hybrid retrieval optimizes recall (get the right chunk into the
candidate pool); reranking optimizes precision (put it at the top). The report
is explicit: rerank the top 20-100 candidates with a cross-encoder or
late-interaction model, because that reordering is where nDCG@10 is won.

Design for a GENERIC corpus:

* The reranker is pluggable behind one protocol. A real cross-encoder
  (sentence-transformers CrossEncoder, e.g. BAAI/bge-reranker or an MS-MARCO
  MiniLM) is the production path; when weights aren't present we fall back to
  a deterministic lexical reranker so the service always runs and stays
  testable offline.

* Nothing here knows about science, climate, or any field. The reranker scores
  (query, candidate_text) pairs. Field-specific behavior comes only from the
  chunk metadata already attached upstream — never from hard-coded topic words.

* Fusion with the first-stage score is explicit and tunable (`alpha`), so you
  can A/B "pure rerank" vs "rerank blended with retrieval score" on a gold set
  rather than guessing.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Protocol

from docparse.vectorization.vectorstore import SearchHit

_WORD = re.compile(r"[a-z0-9]+")


def _terms(text: str) -> list[str]:
    return _WORD.findall(text.lower())


class Reranker(Protocol):
    name: str

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return one relevance score per passage (higher = more relevant)."""
        ...


@dataclass
class LexicalReranker:
    """Deterministic offline fallback: token-overlap relevance with idf-ish
    rarity weighting and a length penalty. Not a substitute for a trained
    cross-encoder, but domain-agnostic and good enough to keep the pipeline
    honest and testable without downloads."""
    name: str = "lexical-rerank-v1"

    def score(self, query: str, passages: list[str]) -> list[float]:
        q = _terms(query)
        if not q:
            return [0.0] * len(passages)
        # corpus-local document frequency over the candidate set -> rare query
        # terms that appear in a passage count for more (poor-man's idf)
        df: dict[str, int] = {}
        tokenized = [_terms(p) for p in passages]
        for toks in tokenized:
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        n = len(passages) or 1
        out = []
        for toks in tokenized:
            tf = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            s = 0.0
            for t in set(q):
                if t in tf:
                    idf = math.log(1 + n / (df.get(t, 0) + 0.5))
                    s += idf * (1 + math.log(tf[t]))
            s /= math.sqrt(len(toks) + 1)          # length normalization
            out.append(s)
        return out


class CrossEncoderReranker:
    """Trained cross-encoder via sentence-transformers.
    Recommended weights (domain-neutral): 'BAAI/bge-reranker-v2-m3' or
    'cross-encoder/ms-marco-MiniLM-L-6-v2'."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        from sentence_transformers import CrossEncoder     # lazy heavy import
        self._m = CrossEncoder(model_name)
        self.name = model_name.rsplit("/", 1)[-1]

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        pairs = [[query, p] for p in passages]
        return [float(s) for s in self._m.predict(pairs)]


def resolve_reranker(spec: str = "auto") -> Reranker:
    """'auto' | 'lexical' | 'ce:<model_name>'."""
    if spec == "lexical":
        return LexicalReranker()
    if spec.startswith("ce:"):
        return CrossEncoderReranker(spec[3:])
    if spec == "auto":
        try:
            return CrossEncoderReranker()
        except Exception:
            return LexicalReranker()
    raise ValueError(f"unknown reranker spec: {spec}")


def _passage_text(hit: SearchHit) -> str:
    """What the reranker reads. Prefer the canonical (breadcrumbed) text so the
    section/type context contributes to relevance, fall back to raw."""
    p = hit.payload
    return (p.get("canonical_text") or p.get("raw_text") or "")[:2000]


def rerank(query: str, hits: list[SearchHit], reranker: Reranker,
           top_k: int | None = None, alpha: float = 1.0) -> list[SearchHit]:
    """Reorder hits by reranker score, optionally blended with the normalized
    first-stage score.

    alpha = 1.0  -> pure reranker order (report's default intent)
    alpha = 0.0  -> ignore reranker, keep retrieval order
    0 < a < 1    -> convex blend of min-max-normalized rerank + retrieval scores
    """
    if not hits:
        return []
    scores = reranker.score(query, [_passage_text(h) for h in hits])

    if alpha < 1.0:
        r_norm = _minmax(scores)
        v_norm = _minmax([h.score for h in hits])
        final = [alpha * r + (1 - alpha) * v for r, v in zip(r_norm, v_norm)]
    else:
        final = scores

    ranked = sorted(zip(hits, final, scores), key=lambda t: -t[1])
    out = []
    for hit, blended, rscore in ranked:
        # preserve provenance; expose both scores for evaluation/debugging
        hit.payload = {**hit.payload, "retrieval_score": hit.score,
                       "rerank_score": rscore}
        hit.score = float(blended)
        out.append(hit)
    return out[:top_k] if top_k else out


def _minmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    lo, hi = min(xs), max(xs)
    if hi - lo < 1e-9:
        return [0.5] * len(xs)
    return [(x - lo) / (hi - lo) for x in xs]
