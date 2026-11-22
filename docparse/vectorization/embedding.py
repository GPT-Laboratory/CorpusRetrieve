"""Dense and sparse embedding implementations.

* Pluggable dense embedders behind one protocol. BGE-M3 is the recommended
  self-host primary (dense mode = normalized CLS + inner product), but model
  weights can't be assumed present in every environment, so the resolver
  degrades: FlagEmbedding BGE-M3 -> sentence-transformers -> HashingEmbedder,
  a deterministic offline fallback for tests and plumbing validation.
* One sparse path: a BM25-style encoder fitted on the corpus at ingest and
  persisted (vocab + idf) so query-time encoding matches ingest-time. This is
  the "hybrid sparse+dense first stage" the evidence base insists on, without
  requiring a learned sparse model.
* Dense vectors are L2-normalized, allowing cosine similarity through a dot
  product.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Protocol

from docparse.text import normalize_token_text


class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str]) -> list[list[float]]: ...


@dataclass
class HashingEmbedder:
    """Deterministic offline embedder: hashed char 3-5-grams + word unigrams,
    L2-normalized. Lexical-overlap similarity only — a plumbing fallback that
    keeps the indexing pipeline runnable and testable with no downloads."""
    dim: int = 512
    name: str = "hashing-v1"
    fallback_from: str | None = None          # set when "auto" could not load a model
    query_prefix: str = ""
    passage_prefix: str = ""

    @property
    def model_id(self) -> str:
        return self.name

    def encode(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            t_low = re.sub(r"\s+", " ", t.lower()).strip()
            # hashing-v1 keeps ASCII word features so existing bundles stay
            # comparable; hashing-v2 uses Unicode word features
            word_rx = r"[a-z0-9°%]+" if self.name == "hashing-v1" else r"[^\W_]+|[°%]"
            feats: list[str] = re.findall(word_rx, t_low)
            for n in (3, 4, 5):
                feats += [t_low[i:i + n] for i in range(0, max(0, len(t_low) - n), 2)]
            for f in feats:
                h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=8).digest(), "big")
                v[h % self.dim] += 1.0 if (h >> 63) else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch means no GPU
        return False


class BGEM3Embedder:
    """BGE-M3 dense mode via FlagEmbedding (pip install FlagEmbedding).
    1024 dimensions, multilingual, with long-context support."""
    name = "bge-m3"
    model_id = "BAAI/bge-m3"
    dim = 1024
    query_prefix = ""
    passage_prefix = ""

    def __init__(self, model_name: str = "BAAI/bge-m3", use_fp16: bool | None = None):
        from FlagEmbedding import BGEM3FlagModel  # lazy heavy import
        # half precision only helps (and only works reliably) on a GPU
        fp16 = _cuda_available() if use_fp16 is None else use_fp16
        self._m = BGEM3FlagModel(model_name, use_fp16=fp16)
        self.model_id = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        res = self._m.encode(texts, return_dense=True, return_sparse=False,
                             return_colbert_vecs=False)
        return [list(map(float, v)) for v in res["dense_vecs"]]


def _e5_prefixes(model_name: str) -> tuple[str, str]:
    """E5 models are trained with these prefixes and score worse without them."""
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


class SentenceTransformerEmbedder:
    """Any sentence-transformers checkpoint (e5, jina-v3, minilm, ...).

    ``model_id`` keeps the full Hugging Face id so a consumer can load the same
    model for queries; ``name`` is the short form used in point IDs.
    """

    # multilingual by default: all-MiniLM-L6-v2 is English-only
    def __init__(self, model_name: str = "intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer
        self._m = SentenceTransformer(model_name)
        self.model_id = model_name
        self.name = model_name.rsplit("/", 1)[-1]
        self.dim = int(self._m.get_sentence_embedding_dimension())
        self.query_prefix, self.passage_prefix = _e5_prefixes(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        """Encode documents (passages)."""
        return self._encode([self.passage_prefix + t for t in texts])

    def encode_queries(self, texts: list[str]) -> list[list[float]]:
        return self._encode([self.query_prefix + t for t in texts])

    def _encode(self, texts: list[str]) -> list[list[float]]:
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
    if spec == "hash-v2":
        return HashingEmbedder(name="hashing-v2")
    if spec == "auto":
        for maker in (lambda: BGEM3Embedder(), lambda: SentenceTransformerEmbedder()):
            try:
                return maker()
            except Exception:
                continue
        # recorded, never silent: the index metadata and quality block report it
        return HashingEmbedder(fallback_from="auto")
    raise ValueError(f"unknown embedder spec: {spec}")


# ASCII tokenizer, kept so older encoders load and score exactly as before. It only
# accepts ASCII letters, so "määrä" is dropped and "maaperän" becomes "maaper".
_TOKEN_ASCII = re.compile(r"[a-z][a-z0-9_-]+|\d+(?:\.\d+)?(?:°c|%)?")
# Unicode tokenizer (default): any Unicode letter, NFKC + casefold, decimal comma or point
_TOKEN_UNICODE = re.compile(r"[^\W\d_][\w-]*|\d+(?:[.,]\d+)?(?:°c|%)?")
TOKENIZERS = {"ascii-v1": _TOKEN_ASCII, "unicode-v2": _TOKEN_UNICODE}
DEFAULT_TOKENIZER = "unicode-v2"


def _tokens(text: str, tokenizer: str = DEFAULT_TOKENIZER) -> list[str]:
    if tokenizer == "ascii-v1":
        return _TOKEN_ASCII.findall(text.lower())
    return [t.strip("-_") for t in _TOKEN_UNICODE.findall(normalize_token_text(text))
            if t.strip("-_")]


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
    name: str = "bm25-local-v2"
    tokenizer: str = DEFAULT_TOKENIZER
    n_docs: int = 0
    corpus_fingerprint: str | None = None

    def fit(self, corpus: Iterable[str]) -> "BM25SparseEncoder":
        df: dict[str, int] = {}
        n, total = 0, 0
        fingerprint = hashlib.sha256()
        for text in corpus:
            toks = _tokens(text, self.tokenizer)
            total += len(toks)
            n += 1
            fingerprint.update(hashlib.sha256(text.encode("utf-8")).digest())
            for t in set(toks):
                df[t] = df.get(t, 0) + 1
        self.avg_len = (total / n) if n else 1.0
        self.n_docs = n
        # the encoder is only valid for the corpus it was fitted on
        self.corpus_fingerprint = fingerprint.hexdigest()
        # sorted vocabulary: term indices no longer depend on PYTHONHASHSEED
        for t in sorted(df):
            self.vocab[t] = len(self.vocab)
            d = df[t]
            self.idf[t] = math.log(1 + (n - d + 0.5) / (d + 0.5))
        return self

    def encode_query(self, text: str) -> tuple[list[int], list[float]]:
        """Query-side vector: one weight per distinct known term.

        Document vectors already carry idf and BM25 saturation/length
        normalisation, so the dot product of this vector with a document vector
        is that document's BM25 score. Encoding a query with ``encode`` (as older
        consumers had to) applies idf twice and normalises the query's length.
        """
        terms = sorted({t for t in _tokens(text, self.tokenizer) if t in self.vocab},
                       key=lambda t: self.vocab[t])
        return [self.vocab[t] for t in terms], [1.0] * len(terms)

    def encode(self, text: str) -> tuple[list[int], list[float]]:
        """Document-side BM25 term weights."""
        tf: dict[str, int] = {}
        toks = _tokens(text, self.tokenizer)
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
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    def to_dict(self) -> dict:
        return {"k1": self.k1, "b": self.b, "avg_len": self.avg_len,
                "vocab": self.vocab, "idf": self.idf, "name": self.name,
                "tokenizer": self.tokenizer, "n_docs": self.n_docs,
                "corpus_fingerprint": self.corpus_fingerprint,
                "query_encoding": "binary-terms; score = dot(query, document)"}

    @classmethod
    def from_dict(cls, d: dict) -> "BM25SparseEncoder":
        # encoders written before 2.0.0 carry no tokenizer field: they used ASCII tokens
        enc = cls(k1=d["k1"], b=d["b"], tokenizer=d.get("tokenizer", "ascii-v1"))
        enc.avg_len, enc.vocab, enc.idf = d["avg_len"], d["vocab"], d["idf"]
        enc.name = d.get("name", "bm25-local-v1")
        enc.n_docs = d.get("n_docs", 0)
        enc.corpus_fingerprint = d.get("corpus_fingerprint")
        return enc

    @classmethod
    def load(cls, path: str | Path) -> "BM25SparseEncoder":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
