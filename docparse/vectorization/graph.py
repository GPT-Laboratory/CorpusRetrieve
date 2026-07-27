"""Knowledge graph projection (Stage 5).

Boundary rule from the report, implemented literally:
  vector DB  <- evidence spans people retrieve semantically
  graph      <- stable identities and exact relations (paper, sections,
                citations, quantities+units, figures/panels, table facts,
                formula symbols), each linked back to its vector point.

Storage is a networkx MultiDiGraph — vendor-neutral by design ("graph model
matters more than vendor"). Exports: GraphML, node/edge JSON, and a Cypher
script for Neo4j/Memgraph.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import networkx as nx

from docparse.vectorization.vectorstore import COLLECTION_OF, point_uuid

_CHUNK_NODE_TYPE = {
    "claim": "Claim", "section_text": "Chunk", "summary": "Chunk",
    "table_summary": "Table", "table_row_group": "TableRowGroup",
    "figure": "Figure", "figure_panel": "FigurePanel",
    "formula": "Formula", "reference": "ReferenceWork",
}


class KnowledgeGraph:
    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    # ---- helpers ---------------------------------------------------------------
    def _node(self, nid: str, ntype: str, **attrs) -> str:
        clean = {k: v for k, v in attrs.items() if v not in (None, [], {})}
        if self.g.has_node(nid):
            self.g.nodes[nid].update(clean)
        else:
            self.g.add_node(nid, ntype=ntype, **clean)
        return nid

    def _edge(self, a: str, b: str, rel: str, **attrs) -> None:
        if not self.g.has_edge(a, b, key=rel):
            self.g.add_edge(a, b, key=rel, rel=rel, **attrs)

    # ---- build ------------------------------------------------------------------
    def build_from_chunks(self, doc: dict, tenant: str, embed_model: str,
                          embed_version: str) -> dict[str, int]:
        doc_id = doc["doc_id"]
        title = (doc.get("stats") or {}).get("doc_title") or doc_id
        paper = self._node(f"paper:{doc_id}", "Paper", title=title,
                           source=doc.get("source"))
        sections: dict[str, str] = {}
        table_nodes: dict[str, str] = {}
        figure_nodes: dict[str, str] = {}
        chunk_nodes: dict[str, str] = {}

        # pre-pass: register figure/table nodes by label so claims processed
        # earlier in document order can still resolve their support edges
        for c in doc["chunks"]:
            lab = (c.get("metadata") or {}).get("label")
            if not lab:
                continue
            if c["chunk_type"] == "figure":
                figure_nodes[lab] = self._node(
                    f"figure:{c['chunk_id']}", "Figure", label=lab)
            elif c["chunk_type"] == "table_summary":
                table_nodes[lab] = self._node(
                    f"table:{c['chunk_id']}", "Table", label=lab)

        for c in doc["chunks"]:
            ctype = c["chunk_type"]
            cid = c["chunk_id"]
            ntype = _CHUNK_NODE_TYPE.get(ctype, "Chunk")
            node = self._node(f"{ntype.lower()}:{cid}", ntype,
                              chunk_type=ctype, role=c.get("role"),
                              pages=",".join(map(str, c.get("pages", []))),
                              label=(c.get("metadata") or {}).get("label"),
                              text=(c.get("text") or "")[:300])
            chunk_nodes[cid] = node

            # provenance + vector linkage (NODE_LINKS_VECTOR_POINT)
            composite = f"{tenant}:{doc_id}:{ctype}:{cid}:{embed_model}:{embed_version}"
            coll = COLLECTION_OF.get(ctype, "sections")
            self._edge(node, self._node(
                f"vec:{point_uuid(composite)}", "VectorPoint",
                collection=coll, composite_id=composite), "NODE_LINKS_VECTOR_POINT")

            # section hierarchy
            sp = c.get("section_path") or []
            parent = paper
            for depth, sec in enumerate(sp):
                sid = sections.setdefault(
                    f"section:{doc_id}:{depth}:{sec}",
                    self._node(f"section:{doc_id}:{depth}:{sec}", "Section",
                               title=sec, depth=depth))
                self._edge(paper, sid, "PAPER_HAS_SECTION")
                if parent != paper:
                    self._edge(parent, sid, "SECTION_HAS_SUBSECTION")
                parent = sid
            if sp:
                self._edge(parent, node, "SECTION_HAS_CHUNK")
            else:
                self._edge(paper, node, "PAPER_HAS_CHUNK")

            # typed structure
            if ntype == "Table":
                lab = (c.get("metadata") or {}).get("label") or cid
                table_nodes[lab] = node
                self._edge(paper, node, "PAPER_HAS_TABLE")
            if ntype == "TableRowGroup":
                lab = (c.get("metadata") or {}).get("label")
                if lab and lab in table_nodes:
                    self._edge(table_nodes[lab], node, "TABLE_HAS_ROW_GROUP")
                # cell facts live in the GRAPH, not as vectors (report: table_kv)
                for i, f in enumerate((c.get("metadata") or {}).get("facts") or []):
                    kv = self._node(f"tablekv:{cid}:{i}", "TableKV",
                                    row_key=f.get("row_key"), column=f.get("column"),
                                    value=f.get("value"))
                    self._edge(node, kv, "ROW_GROUP_HAS_KV")
            if ntype == "Figure":
                lab = (c.get("metadata") or {}).get("label") or cid
                figure_nodes[lab] = node
                self._edge(paper, node, "PAPER_HAS_FIGURE")
            if ntype == "FigurePanel":
                lab = (c.get("metadata") or {}).get("label")
                if lab and lab in figure_nodes:
                    self._edge(figure_nodes[lab], node, "FIGURE_HAS_PANEL")
                elif c.get("parent_chunk_id") in chunk_nodes:
                    self._edge(chunk_nodes[c["parent_chunk_id"]], node, "FIGURE_HAS_PANEL")
            if ntype == "Formula":
                num = (c.get("metadata") or {}).get("equation_number")
                expr = (c.get("metadata") or {}).get("expression") or c.get("text", "")
                for sym in _symbols(expr):
                    s = self._node(f"entity:{doc_id}:{sym}", "Entity", name=sym)
                    self._edge(node, s, "FORMULA_DEFINES_SYMBOL")
                if num is not None:
                    self.g.nodes[node]["equation_number"] = num
            if ntype == "ReferenceWork":
                key = (c.get("metadata") or {}).get("citation_key")
                if key:
                    self.g.nodes[node]["citation_key"] = key
                self._edge(paper, node, "PAPER_CITES_WORK")

            # claims: support + quantities
            if ntype == "Claim":
                if c.get("parent_chunk_id") in chunk_nodes:
                    self._edge(node, chunk_nodes[c["parent_chunk_id"]],
                               "CLAIM_SUPPORTED_BY_CHUNK")
                for lab in c.get("linked_figures", []):
                    if lab in figure_nodes:
                        self._edge(node, figure_nodes[lab], "CLAIM_SUPPORTED_BY_FIGURE")
                for lab in c.get("linked_tables", []):
                    if lab in table_nodes:
                        self._edge(node, table_nodes[lab], "CLAIM_SUPPORTED_BY_TABLE")

            # quantities + units for every fact-bearing chunk
            for i, q in enumerate(c.get("quantities_norm") or c.get("quantities") or []):
                qn = self._node(f"quantity:{cid}:{i}", "Quantity",
                                value_raw=str(q.get("value")),
                                value_norm=q.get("value_norm"),
                                value_norm_min=q.get("value_norm_min"),
                                value_norm_max=q.get("value_norm_max"),
                                qtype=q.get("type"))
                self._edge(node, qn, "CHUNK_REPORTS_QUANTITY")
                if q.get("unit"):
                    self._edge(qn, self._node(f"unit:{q['unit']}", "Unit",
                                              name=q["unit"]), "QUANTITY_HAS_UNIT")

            # citation edges from content chunks to reference works
            for cit in c.get("citations", []):
                key = cit.get("key")
                if key:
                    ref = self._node(f"citekey:{doc_id}:{key}", "ReferenceWork",
                                     citation_key=key)
                    self._edge(node, ref, "CHUNK_CITES_WORK")
                    self._edge(paper, ref, "PAPER_CITES_WORK")

        return self.stats()

    # ---- queries used by validation --------------------------------------------
    def supports_of(self, claim_node: str) -> list[str]:
        return [v for _, v, k in self.g.out_edges(claim_node, keys=True)
                if k.startswith("CLAIM_SUPPORTED_BY")]

    def stats(self) -> dict[str, int]:
        by_type: dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            by_type[d.get("ntype", "?")] = by_type.get(d.get("ntype", "?"), 0) + 1
        by_rel: dict[str, int] = {}
        for _, _, k in self.g.edges(keys=True):
            by_rel[k] = by_rel.get(k, 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "node_types": by_type, "edge_types": by_rel}

    # ---- exports -------------------------------------------------------------------
    def export(self, outdir: str | Path) -> dict[str, str]:
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        gml = out / "knowledge_graph.graphml"
        g2 = nx.MultiDiGraph()
        for n, d in self.g.nodes(data=True):        # graphml needs scalar attrs
            g2.add_node(n, **{k: ("" if v is None else str(v)) for k, v in d.items()})
        for a, b, k, d in self.g.edges(keys=True, data=True):
            g2.add_edge(a, b, key=k, **{kk: str(vv) for kk, vv in d.items()})
        nx.write_graphml(g2, gml)

        js = out / "knowledge_graph.json"
        js.write_text(json.dumps({
            "nodes": [{"id": n, **d} for n, d in self.g.nodes(data=True)],
            "edges": [{"source": a, "target": b, "rel": k, **d}
                      for a, b, k, d in self.g.edges(keys=True, data=True)],
        }, ensure_ascii=False, indent=1), encoding="utf-8")

        cy = out / "knowledge_graph.cypher"
        with cy.open("w", encoding="utf-8") as fh:
            fh.write("// Generated by docparse stage5 — load into Neo4j/Memgraph\n")
            for n, d in self.g.nodes(data=True):
                props = {k: v for k, v in d.items() if k != "ntype" and v not in (None, "")}
                fh.write(f"MERGE (n:{d.get('ntype','Node')} {{id: {json.dumps(n)}}}) "
                         f"SET n += {_cyprops(props)};\n")
            for a, b, k, _ in self.g.edges(keys=True, data=True):
                fh.write(f"MATCH (a {{id: {json.dumps(a)}}}), (b {{id: {json.dumps(b)}}}) "
                         f"MERGE (a)-[:{re.sub(r'[^A-Z_]', '_', k.upper())}]->(b);\n")
        return {"graphml": str(gml), "json": str(js), "cypher": str(cy)}


def _cyprops(props: dict[str, Any]) -> str:
    items = ", ".join(f"{re.sub(r'[^A-Za-z0-9_]', '_', k)}: {json.dumps(v)}"
                      for k, v in props.items())
    return "{" + items + "}"


_SYM = re.compile(r"\b([A-Za-z](?:_?[A-Za-z0-9]{0,6})?)\b")
_SYM_STOP = {"max", "min", "the", "and", "for", "sum", "log", "exp", "where"}


def _symbols(expr: str) -> list[str]:
    return [s for s in dict.fromkeys(_SYM.findall(expr))
            if len(s) <= 6 and s.lower() not in _SYM_STOP][:8]
