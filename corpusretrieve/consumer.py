"""Reference consumer for a CorpusRetrieve bundle.

Loads a bundle ZIP (or an output directory) without a database and answers
hybrid queries with matrix operations: one matrix product scores every unit,
where a per-unit Python loop takes seconds per question on a large corpus.

    index = PortableIndex.load("bundle.zip")
    hits = index.search("typpilannoitus ohralle", k=10)
"""
from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from docparse.vectorization.embedding import BM25SparseEncoder, resolve_embedder

LEXICAL_MODELS = ("hashing-v1", "hashing-v2")


@dataclass
class Hit:
    id: str
    score: float
    dense: float
    sparse: float
    payload: dict[str, Any]
    matched_query: str = ""                      # which query version scored best
    query_scores: dict[str, float] = field(default_factory=dict)


class _Source:
    """Uniform read access to a bundle ZIP or an unpacked output directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._zip = zipfile.ZipFile(self.path) if self.path.is_file() else None

    def exists(self, name: str) -> bool:
        if self._zip is not None:
            return name in self._zip.namelist()
        return (self.path / name).exists()

    def read(self, name: str) -> bytes:
        if self._zip is not None:
            return self._zip.read(name)
        return (self.path / name).read_bytes()


class PortableIndex:
    def __init__(self, ids: list[str], payloads: list[dict], dense: np.ndarray,
                 sparse_rows: list[tuple[list[int], list[float]]],
                 encoder: BM25SparseEncoder, metadata: dict, collections: list[str]):
        self.ids = ids
        self.payloads = payloads
        self.dense = dense
        self.encoder = encoder
        self.metadata = metadata
        self.collections = np.asarray(collections)
        vocab = max(len(encoder.vocab), 1)
        self.sparse = _csr(sparse_rows, vocab)
        self._embedder = None

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: str | Path) -> PortableIndex:
        src = _Source(path)
        metadata = json.loads(src.read("index/metadata.json"))
        encoder = BM25SparseEncoder.from_dict(json.loads(src.read("index/sparse_encoder.json")))
        ids, payloads, sparse_rows, collections, inline = [], [], [], [], []
        for line in src.read("index/vectors.jsonl").decode("utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            ids.append(rec["id"])
            payloads.append(rec["payload"])
            collections.append(rec.get("collection", "sections"))
            sp = rec.get("sparse") or {}
            sparse_rows.append((sp.get("indices", []), sp.get("values", [])))
            if "vector" in rec:
                inline.append(rec["vector"])
        dense_name = metadata.get("dense_file", "index/dense.npy")
        if src.exists(dense_name):
            dense = np.load(io.BytesIO(src.read(dense_name)), allow_pickle=False)
        else:                                   # older bundle: vectors only inline
            dense = np.asarray(inline, dtype=np.float32)
        if dense.shape[0] != len(ids):
            raise ValueError(f"dense matrix has {dense.shape[0]} rows for {len(ids)} records")
        return cls(ids, payloads, dense.astype(np.float32, copy=False), sparse_rows,
                   encoder, metadata, collections)

    # --------------------------------------------------------------- search
    def embedder(self):
        if self._embedder is None:
            self._embedder = resolve_embedder(embedder_spec_for(self.metadata))
        return self._embedder

    def dense_scores(self, query: str) -> np.ndarray:
        embedder = self.embedder()
        encode = getattr(embedder, "encode_queries", embedder.encode)
        q = np.asarray(encode([query])[0], dtype=np.float32)
        return self.dense @ q

    def sparse_scores(self, query: str) -> np.ndarray:
        idx, val = self.encoder.encode_query(query)
        q = np.zeros(self.sparse["n_cols"], dtype=np.float32)
        q[np.asarray(idx, dtype=np.int64)] = np.asarray(val, dtype=np.float32)
        return _csr_dot(self.sparse, q)

    def _fused(self, query: str, dense_weight: float):
        dense = self.dense_scores(query)
        sparse = self.sparse_scores(query)
        top = sparse.max() if sparse.size else 0.0
        sparse_n = sparse / top if top > 0 else sparse
        return dense_weight * dense + (1.0 - dense_weight) * sparse_n, dense, sparse

    def search(self, query: str, k: int = 10, dense_weight: float = 0.55,
               collections: list[str] | None = None,
               include_excluded: bool = False,
               extra_queries: Iterable[str] | None = None,
               translator: Callable[[str], Iterable[str]] | None = None,
               languages: list[str] | None = None,
               where: Callable[[dict[str, Any]], bool] | None = None) -> list[Hit]:
        """Hybrid search: convex combination of dense cosine and max-normalised BM25.

        Cross-lingual retrieval: a lexical index (BM25, hashing
        embedder) cannot match a Finnish question to English evidence. Pass
        the question in other languages as ``extra_queries``, or a
        ``translator`` callable that returns them; every version is scored and
        each unit keeps its best score (max fusion), so evidence in either
        language can rank. No translation model is bundled: the caller chooses
        one (an LLM call, a glossary, a machine-translation service).
        """
        queries = [query]
        if translator is not None:
            queries += [q for q in translator(query) if q]
        if extra_queries:
            queries += [q for q in extra_queries if q]
        queries = list(dict.fromkeys(q.strip() for q in queries if q.strip()))

        best = dense_best = sparse_best = None
        which = None
        per_query: list[np.ndarray] = []
        for n, q in enumerate(queries):
            score, dense, sparse = self._fused(q, dense_weight)
            per_query.append(score)
            if best is None:
                best, dense_best, sparse_best = score, dense, sparse
                which = np.zeros(len(score), dtype=np.int64)
            else:
                better = score > best
                best = np.where(better, score, best)
                dense_best = np.where(better, dense, dense_best)
                sparse_best = np.where(better, sparse, sparse_best)
                which = np.where(better, n, which)

        mask = np.ones(len(self.ids), dtype=bool)
        if collections:
            mask &= np.isin(self.collections, collections)
        if not include_excluded:
            mask &= np.asarray([p.get("retrieval_default") != "exclude" for p in self.payloads])
        if languages:
            mask &= np.asarray([p.get("language") in languages for p in self.payloads])
        if where is not None:
            mask &= np.asarray([bool(where(p)) for p in self.payloads])
        score = np.where(mask, best, -np.inf)
        k = min(k, int(mask.sum()))
        if k <= 0:
            return []
        order = np.argpartition(-score, k - 1)[:k]
        order = order[np.argsort(-score[order], kind="stable")]
        return [Hit(self.ids[i], float(score[i]), float(dense_best[i]), float(sparse_best[i]),
                    self.payloads[i], matched_query=queries[int(which[i])],
                    query_scores={q: float(s[i]) for q, s in zip(queries, per_query,
                                                                 strict=True)})
                for i in order]

    @property
    def languages(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for p in self.payloads:
            lang = p.get("language") or "unknown"
            counts[lang] = counts.get(lang, 0) + 1
        return counts

    def crosslingual_risk(self, min_share: float = 0.1) -> bool:
        """True when a lexical index serves a corpus with two or more languages.

        Then a single-language query misses evidence in the other language
        unless ``extra_queries`` or ``translator`` is used.
        """
        return cross_lingual_risk(self.metadata.get("model", ""), self.languages, min_share)


def cross_lingual_risk(model: str, languages: dict[str, int], min_share: float = 0.1) -> bool:
    total = sum(v for k, v in languages.items() if k != "unknown")
    if not total or model not in LEXICAL_MODELS:
        return False
    return sum(1 for k, v in languages.items() if k != "unknown" and v / total >= min_share) >= 2


def recall_at_k(exact: list[str], approximate: list[str], k: int = 10) -> float:
    """Top-k agreement of an approximate index with exact search (RQ3 check)."""
    gold = set(exact[:k])
    return len(gold & set(approximate[:k])) / max(1, len(gold))


# ------------------------------------------------------------ tiny CSR helpers

def _csr(rows: list[tuple[list[int], list[float]]], n_cols: int) -> dict:
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    for i, (idx, _) in enumerate(rows):
        indptr[i + 1] = indptr[i] + len(idx)
    indices = np.fromiter((j for idx, _ in rows for j in idx), dtype=np.int64,
                          count=int(indptr[-1]))
    data = np.fromiter((v for _, val in rows for v in val), dtype=np.float32,
                       count=int(indptr[-1]))
    return {"indptr": indptr, "indices": indices, "data": data, "n_cols": n_cols}


def _csr_dot(m: dict, q: np.ndarray) -> np.ndarray:
    if m["indices"].size == 0:
        return np.zeros(len(m["indptr"]) - 1, dtype=np.float32)
    contrib = m["data"] * q[m["indices"]]
    # sum contributions per row; reduceat needs non-empty segments, so add a pad
    sums = np.add.reduceat(np.append(contrib, 0.0), m["indptr"][:-1])
    empty = m["indptr"][1:] == m["indptr"][:-1]
    sums[empty] = 0.0
    return sums.astype(np.float32)


def embedder_spec_for(metadata: dict) -> str:
    """The embedder spec that reproduces an index's dense model for queries."""
    model = metadata.get("model", "hashing-v1")
    known = {"hashing-v1": "hash", "hashing-v2": "hash-v2", "bge-m3": "bge-m3"}
    if model in known:
        return known[model]
    model_id = metadata.get("model_id") or metadata.get("embedder_requested", "")
    if model_id.startswith("st:"):
        return model_id
    if "/" in model_id:
        return f"st:{model_id}"
    raise ValueError(
        f"index model {model!r} has no full model id; rebuild the index with "
        "'corpusretrieve reindex' or pass an embedder to PortableIndex")
