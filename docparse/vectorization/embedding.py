"""Stage 5 embedding layer.

Design (follows the Stage-5 report, adapted to reality):

* Pluggable dense embedders behind one protocol. BGE-M3 is the recommended
  self-host primary (dense mode = normalized CLS + inner product), but model
  weights can't be assumed present in every environment, so the resolver
  degrades: FlagEmbedding BGE-M3 -> sentence-transformers -> HashingEmbedder,
  a deterministic offline fallback good enough for tests, demos, and
  plumbing validation (NOT for production relevance).
* One sparse path: a BM25-style encoder fitted on the corpus at ingest and
  persisted (vocab + idf) so query-time encoding matches ingest-time. This is
  the "hybrid sparse+dense first stage" the evidence base insists on, without
  requiring a learned sparse model.
* Everything emits L2-normalized dense vectors, so cosine == dot product and
  the vector DB metric choice is uncontroversial.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


# --------------------------------------------------------------------- dense

@dataclass
class HashingEmbedder:
    """Deterministic offline embedder: hashed char 3-5-grams + word unigrams,
    L2-normalized. Lexical-overlap similarity only — a plumbing fallback that
    keeps the whole Stage 5 pipeline runnable and testable with no downloads."""
    dim: int = 512
    name: str = "hashing-v1"

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            t_low = re.sub(r"\s+", " ", t.lower()).strip()
            feats: list[str] = re.findall(r"[a-z0-9°%]+", t_low)
            for n in (3, 4, 5):
                feats += [t_low[i:i + n] for i in range(0, max(0, len(t_low) - n), 2)]
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 63) else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class BGEM3Embedder:
    """BGE-M3 dense mode via FlagEmbedding (pip install FlagEmbedding).
    1024 dims, multilingual, 8k context — the report's self-host primary."""
    name = "bge-m3"
    dim = 1024

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool = True):
        from FlagEmbedding import BGEM3FlagModel      # lazy heavy import
        self._m = BGEM3FlagModel(model_name, use_fp16=use_fp16)

    def encode(self, texts: list[str]) -> list[list[float]]:
        res = self._m.encode(texts, return_dense=True, return_sparse=False,
                             return_colbert_vecs=False)
        return [list(map(float, v)) for v in res["dense_vecs"]]


class SentenceTransformerEmbedder:
    """Any sentence-transformers checkpoint (e5, jina-v3, minilm, ...)."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)
        self.name = model_name.rsplit("/", 1)[-1]
        self.dim = int(self._m.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in
                self._m.encode(texts, normalize_embeddings=True)]


def resolve_embedder(spec: str = "auto") -> Embedder:
    """'auto' | 'hash' | 'bge-m3' | 'st:<model_name>'."""
    if spec == "hash":
        return HashingEmbedder()
    if spec == "bge-m3":
        return BGEM3Embedder()
    if spec.startswith("st:"):
        return SentenceTransformerEmbedder(spec[3:])
    if spec == "auto":
        for maker in (lambda: BGEM3Embedder(), lambda: SentenceTransformerEmbedder()):
            try:
                return maker()
            except Exception:
                continue
        return HashingEmbedder()
    raise ValueError(f"unknown embedder spec: {spec}")


# --------------------------------------------------------------------- sparse

_TOKEN = re.compile(r"[a-z][a-z0-9_-]+|\d+(?:\.\d+)?(?:°c|%)?")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass
class BM25SparseEncoder:
    """Corpus-fitted BM25 term weights as qdrant sparse vectors.

    Fit at ingest on canonical texts, persist vocab+idf next to the DB, reload
    at query time — the ranking is only meaningful if both sides share the
    same statistics.
    """
    k1: float = 1.4
    b: float = 0.75
    vocab: dict[str, int] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    avg_len: float = 0.0
    name: str = "bm25-local-v1"

    def fit(self, corpus: Iterable[str]) -> "BM25SparseEncoder":
        df: dict[str, int] = {}
        n, total = 0, 0
        for text in corpus:
            toks = set(_tokens(text))
            total += len(_tokens(text))
            n += 1
            for t in toks:
                df[t] = df.get(t, 0) + 1
        self.avg_len = (total / n) if n else 1.0
        for t, d in df.items():
            self.vocab[t] = len(self.vocab)
            self.idf[t] = math.log(1 + (n - d + 0.5) / (d + 0.5))
        return self

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        tf: dict[str, int] = {}
        toks = _tokens(text)
        for t in toks:
            if t in self.vocab:
                tf[t] = tf.get(t, 0) + 1
        L = len(toks) or 1
        idx, val = [], []
        for t, f in tf.items():
            w = self.idf[t] * (f * (self.k1 + 1)) / (
                f + self.k1 * (1 - self.b + self.b * L / self.avg_len))
            if w > 0:
                idx.append(self.vocab[t])
                val.append(w)
        return idx, val

    # ---- persistence -------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"k1": self.k1, "b": self.b, "avg_len": self.avg_len,
             "vocab": self.vocab, "idf": self.idf, "name": self.name}), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "BM25SparseEncoder":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        enc = cls(k1=d["k1"], b=d["b"])
        enc.avg_len, enc.vocab, enc.idf, enc.name = d["avg_len"], d["vocab"], d["idf"], d["name"]
        return enc
