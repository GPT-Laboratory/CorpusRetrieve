"""Stage 6 — graph-augmented retrieval (evidence expansion).

A vector hit is a single span. But the answer's *grounding* often lives in
neighbours the retriever didn't rank: the figure a claim cites, the table a
row-group belongs to, the section a paragraph sits in, the parent a panel
hangs off. The knowledge graph already encodes these; this module walks it to
turn each hit into a provenance-complete evidence bundle.

Engineered to stay GENERIC:

* The walk is driven by an EdgePolicy — a set of relation names to follow and
  a direction — not by any field-specific logic. The defaults follow the
  structural relations every document has (support, membership, hierarchy).
  A caller in law, medicine, finance, or engineering passes a different policy
  without touching this code.

* Node -> chunk resolution is by convention (`{ntype}:{chunk_id}`), so the
  expander needs only the graph and the chunk lookup, not the vector store.

* Everything is bounded: max hops, max neighbours per hit, and a global cap,
  so a pathological hub node can't explode the bundle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from docparse.vectorization.graph import KnowledgeGraph
from docparse.vectorization.vectorstore import SearchHit


@dataclass
class EdgePolicy:
    """Which relations to traverse, and how far. Relation names match the KG's
    edge keys; the defaults are the domain-neutral structural ones."""
    # relations that carry supporting/grounding evidence (claim -> its support)
    support: tuple[str, ...] = (
        "CLAIM_SUPPORTED_BY_CHUNK", "CLAIM_SUPPORTED_BY_FIGURE",
        "CLAIM_SUPPORTED_BY_TABLE", "CLAIM_SUPPORTED_BY_PANEL",
    )
    # membership/hierarchy relations (row-group -> table, panel -> figure,
    # chunk -> section); followed in BOTH directions so a table hit pulls its
    # rows and a row hit pulls its table
    structure: tuple[str, ...] = (
        "TABLE_HAS_ROW_GROUP", "FIGURE_HAS_PANEL", "SECTION_HAS_CHUNK",
        "ROW_GROUP_HAS_KV", "FORMULA_DEFINES_SYMBOL",
    )
    # citation relations (chunk -> the work it cites)
    citation: tuple[str, ...] = ("CHUNK_CITES_WORK", "PAPER_CITES_WORK")
    max_hops: int = 1
    max_neighbours_per_hit: int = 6
    max_total_expansion: int = 24
    include_citations: bool = False   # off by default: references are noisy context


@dataclass
class EvidenceBundle:
    """A ranked hit plus the graph neighbours that ground it."""
    hit: SearchHit
    neighbours: list[dict] = field(default_factory=list)   # {chunk_id, ntype, rel, hops, text}

    def to_dict(self) -> dict:
        return {"chunk_id": self.hit.chunk_id, "chunk_type": self.hit.chunk_type,
                "score": self.hit.score, "payload": self.hit.payload,
                "evidence": self.neighbours}


def _followable(policy: EdgePolicy) -> set[str]:
    rels = set(policy.support) | set(policy.structure)
    if policy.include_citations:
        rels |= set(policy.citation)
    return rels


def _node_id_for_chunk(kg: KnowledgeGraph, chunk_id: str) -> str | None:
    """Chunk nodes are id'd `{ntype.lower()}:{chunk_id}`; find the node whose
    id ends with the chunk id and carries a chunk_type attribute."""
    suffix = f":{chunk_id}"
    for nid, d in kg.g.nodes(data=True):
        if nid.endswith(suffix) and d.get("chunk_type"):
            return nid
    return None


def expand_hits(hits: list[SearchHit], kg: KnowledgeGraph,
                chunk_lookup: dict[str, dict],
                policy: EdgePolicy | None = None) -> list[EvidenceBundle]:
    """Attach graph neighbours to each hit. `chunk_lookup` maps chunk_id ->
    the chunk dict (for neighbour text); pass {} to omit neighbour text."""
    policy = policy or EdgePolicy()
    follow = _followable(policy)
    bundles: list[EvidenceBundle] = []
    seen_global: set[str] = set()
    budget = policy.max_total_expansion

    for hit in hits:
        bundle = EvidenceBundle(hit=hit)
        start = _node_id_for_chunk(kg, hit.chunk_id)
        if start is None or budget <= 0:
            bundles.append(bundle)
            continue

        # bounded BFS over the followable relations, both directions
        frontier: list[tuple[str, int]] = [(start, 0)]
        local_seen = {start}
        per_hit = 0
        while frontier and per_hit < policy.max_neighbours_per_hit and budget > 0:
            node, hops = frontier.pop(0)
            if hops >= policy.max_hops:
                continue
            for nbr, rel, direction in _neighbours(kg, node, follow):
                if nbr in local_seen:
                    continue
                local_seen.add(nbr)
                d = kg.g.nodes[nbr]
                nbr_chunk = _chunk_id_of(nbr, d)
                if nbr_chunk and nbr_chunk not in seen_global:
                    ck = chunk_lookup.get(nbr_chunk, {})
                    bundle.neighbours.append({
                        "chunk_id": nbr_chunk, "ntype": d.get("ntype"),
                        "rel": rel, "direction": direction, "hops": hops + 1,
                        "text": (ck.get("text") or d.get("text") or "")[:400],
                        "pages": ck.get("pages"),
                        "label": (ck.get("metadata") or {}).get("label"),
                    })
                    seen_global.add(nbr_chunk)
                    per_hit += 1
                    budget -= 1
                frontier.append((nbr, hops + 1))
                if per_hit >= policy.max_neighbours_per_hit or budget <= 0:
                    break
        bundles.append(bundle)
    return bundles


def _neighbours(kg: KnowledgeGraph, node: str, follow: set[str]
                ) -> Iterable[tuple[str, str, str]]:
    """Yield (neighbour_id, relation, direction) for followable edges."""
    for _, v, k in kg.g.out_edges(node, keys=True):
        if k in follow:
            yield v, k, "out"
    for u, _, k in kg.g.in_edges(node, keys=True):
        if k in follow:
            yield u, k, "in"


def _chunk_id_of(node_id: str, attrs: dict) -> str | None:
    """A node maps back to a chunk only if it *is* a chunk node (has a
    chunk_type). Structural-only nodes (Section, Unit, VectorPoint) return
    None — they're graph scaffolding, not retrievable evidence."""
    if not attrs.get("chunk_type"):
        return None
    return node_id.split(":", 1)[1] if ":" in node_id else node_id
