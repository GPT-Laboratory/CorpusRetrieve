from __future__ import annotations

from typing import Any

MAX_ELEMENT_TEXT = 600
MAX_NODE_TEXT = 280
# shown on request from the legend: a paper cites dozens of works and every
# unit links to its vector, so drawing them by default hides the outline
DEFAULT_HIDDEN_NODE_TYPES = ("VectorPoint", "Quantity", "Unit", "TableKV", "ReferenceWork",
                             "Entity")


def document_elements(
    extraction: dict[str, Any],
    page: int | None = None,
    kind: str | None = None,
) -> dict[str, Any]:
    """Typed elements with bounding boxes, sized for a provenance overlay."""
    elements = extraction.get("elements", [])
    if page is not None:
        elements = [item for item in elements if item.get("page") == page]
    if kind:
        elements = [item for item in elements if item.get("kind") == kind]
    return {
        "doc_id": extraction.get("doc_id"),
        "source": extraction.get("source"),
        "doc_type": extraction.get("doc_type"),
        "page_count": extraction.get("page_count"),
        "page_dims": extraction.get("stats", {}).get("page_dims", {}),
        "page_previews": extraction.get("page_previews", {}),
        "count": len(elements),
        "elements": [
            {
                "id": item.get("id"),
                "kind": item.get("kind"),
                "page": item.get("page"),
                "bbox": item.get("bbox"),
                "role": (item.get("metadata") or {}).get("role"),
                "text": (item.get("text") or "")[:MAX_ELEMENT_TEXT],
            }
            for item in elements
        ],
    }


def document_chunks(document: dict[str, Any]) -> dict[str, Any]:
    """Chunks reduced to the provenance fields the workbench renders."""
    return {
        "doc_id": document.get("doc_id"),
        "source": document.get("source"),
        "stats": document.get("stats", {}),
        "chunks": [
            {
                "chunk_id": chunk.get("chunk_id"),
                "chunk_type": chunk.get("chunk_type"),
                "role": chunk.get("role"),
                "pages": chunk.get("pages", []),
                "section_path": chunk.get("section_path", []),
                "element_ids": chunk.get("element_ids", []),
                "char_start": chunk.get("char_start"),
                "char_end": chunk.get("char_end"),
                "token_estimate": chunk.get("token_estimate"),
                "label": (chunk.get("metadata") or {}).get("label"),
                "parent_chunk_id": chunk.get("parent_chunk_id"),
                "text": chunk.get("text", ""),
            }
            for chunk in document.get("chunks", [])
        ],
    }


GRAPH_NODE_LIMIT = 1500
# kept first when a document graph is larger than GRAPH_NODE_LIMIT
GRAPH_TYPE_PRIORITY = ("Paper", "Section", "Chunk", "Table", "Figure", "Claim",
                       "FigurePanel", "TableRowGroup", "Formula", "ReferenceWork",
                       "Entity", "Quantity", "Unit", "TableKV", "VectorPoint")
# edges that make the document's outline: they decide where a node is drawn
_TREE_RELATIONS = (
    ("PAPER_HAS_SECTION", False), ("SECTION_HAS_SUBSECTION", False),
    ("SECTION_HAS_CHUNK", False), ("PAPER_HAS_CHUNK", False),
    ("TABLE_HAS_ROW_GROUP", False), ("FIGURE_HAS_PANEL", False),
    ("CLAIM_SUPPORTED_BY_CHUNK", True), ("PAPER_HAS_TABLE", False),
    ("PAPER_HAS_FIGURE", False), ("PAPER_CITES_WORK", False),
    ("CHUNK_CITES_WORK", False), ("FORMULA_DEFINES_SYMBOL", False),
    ("CHUNK_REPORTS_QUANTITY", False), ("QUANTITY_HAS_UNIT", False),
    ("ROW_GROUP_HAS_KV", False), ("NODE_LINKS_VECTOR_POINT", False),
)


def document_graph(document: dict[str, Any], corpus_name: str, embed_model: str,
                   embed_version: str = "v1",
                   limit: int = GRAPH_NODE_LIMIT) -> dict[str, Any]:
    """One document's knowledge graph, laid out for the console.

    The graph is rebuilt from the document's chunks with the same builder that
    wrote the corpus exports, so it matches them exactly, but it is small
    enough to draw: the corpus graph of a large build has over a million
    nodes. Positions come from a radial layout of the document outline (paper
    in the centre, then sections, then their units), so the same document is
    drawn the same way every time.
    """
    from docparse.vectorization.graph import _CHUNK_NODE_TYPE, KnowledgeGraph

    builder = KnowledgeGraph()
    builder.build_from_chunks(document, corpus_name, embed_model, embed_version)
    graph = builder.g
    all_types: dict[str, int] = {}
    for _, data in graph.nodes(data=True):
        all_types[data.get("ntype", "unknown")] = all_types.get(data.get("ntype", "unknown"), 0) + 1

    rank = {name: index for index, name in enumerate(GRAPH_TYPE_PRIORITY)}
    ordered = sorted(graph.nodes(data=True),
                     key=lambda item: rank.get(item[1].get("ntype"), len(rank)))
    kept = {node for node, _ in ordered[:limit]}

    chunk_of: dict[str, str] = {}
    doc_id = document.get("doc_id", "")
    for chunk in document.get("chunks", []):
        ntype = _CHUNK_NODE_TYPE.get(chunk.get("chunk_type"), "Chunk")
        key = (chunk.get("metadata") or {}).get("citation_key")
        node = (f"citekey:{doc_id}:{key}" if chunk.get("chunk_type") == "reference" and key
                else f"{ntype.lower()}:{chunk['chunk_id']}")
        chunk_of.setdefault(node, chunk["chunk_id"])

    edges = [{"source": a, "target": b, "rel": rel}
             for a, b, rel in graph.edges(keys=True) if a in kept and b in kept]
    # lay out the outline only; nodes that start hidden (cited works, vector
    # points, values) are placed beside the outline node they belong to, so
    # they neither take up space on the rings nor move the outline when shown
    ntype = {node: data.get("ntype") for node, data in graph.nodes(data=True)}
    outline = {n for n in kept if ntype.get(n) not in DEFAULT_HIDDEN_NODE_TYPES}
    positions, depth = _radial_layout(
        f"paper:{doc_id}", outline,
        [e for e in edges if e["source"] in outline and e["target"] in outline])
    _place_satellites(kept - outline, edges, positions, depth)
    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    nodes = []
    for node, data in ordered[:limit]:
        x, y = positions.get(node, (0.0, 0.0))
        nodes.append({
            "id": node,
            "type": data.get("ntype"),
            "label": _node_label(node, data),
            "role": data.get("role"),
            "text": (data.get("text") or "")[:MAX_NODE_TEXT],
            "pages": data.get("pages"),
            "degree": degree.get(node, 0),
            "depth": depth.get(node),
            "chunk_id": chunk_of.get(node),
            "x": round(x, 4), "y": round(y, 4),
        })
    return {
        "doc_id": doc_id,
        "nodes": nodes,
        "edges": edges,
        "all_types": all_types,
        "default_hidden": list(DEFAULT_HIDDEN_NODE_TYPES),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "total_nodes": graph.number_of_nodes(),
        "truncated": graph.number_of_nodes() > len(nodes),
    }


def _radial_layout(root: str, nodes: set[str], edges: list[dict[str, Any]]
                   ) -> tuple[dict[str, tuple[float, float]], dict[str, int]]:
    """Tree layout on concentric rings: each subtree gets an angle in
    proportion to its number of leaves. Nodes outside the outline (a cited
    work reached only through a citation) hang from the first node that
    links to them; anything still unreached hangs from the root."""
    import math

    parent: dict[str, str] = {}
    by_rel: dict[str, list[dict[str, Any]]] = {}
    for edge in edges:
        by_rel.setdefault(edge["rel"], []).append(edge)
    for rel, reverse in _TREE_RELATIONS:
        for edge in by_rel.get(rel, []):
            up, down = ((edge["target"], edge["source"]) if reverse
                        else (edge["source"], edge["target"]))
            if down != root and down not in parent and up != down:
                parent[down] = up
    # break any cycle the relations could form, then attach the remainder
    for node in list(parent):
        seen, walk = {node}, parent.get(node)
        while walk is not None and walk != root:
            if walk in seen:
                parent[node] = root
                break
            seen.add(walk)
            walk = parent.get(walk)
    for node in sorted(nodes):
        if node != root and node not in parent:
            parent[node] = root
    children: dict[str, list[str]] = {}
    for node, up in parent.items():
        children.setdefault(up, []).append(node)
    for kids in children.values():
        kids.sort()

    leaves: dict[str, int] = {}

    def count(node: str, guard: int = 0) -> int:
        if guard > 64:
            return 1
        kids = children.get(node, [])
        leaves[node] = sum(count(k, guard + 1) for k in kids) if kids else 1
        return leaves[node]

    count(root)
    positions: dict[str, tuple[float, float]] = {root: (0.0, 0.0)}
    depth: dict[str, int] = {root: 0}
    stack = [(root, 0.0, 2 * math.pi, 0)]
    while stack:
        node, start, span, level = stack.pop()
        cursor = start
        for kid in children.get(node, []):
            share = span * leaves.get(kid, 1) / max(leaves.get(node, 1), 1)
            angle = cursor + share / 2
            radius = float(level + 1)
            positions[kid] = (radius * math.cos(angle), radius * math.sin(angle))
            depth[kid] = level + 1
            stack.append((kid, cursor, share, level + 1))
            cursor += share
    return positions, depth


def _place_satellites(nodes: set[str], edges: list[dict[str, Any]],
                      positions: dict[str, tuple[float, float]],
                      depth: dict[str, int]) -> None:
    """Put each remaining node on a small circle around a node already placed."""
    import math

    anchor: dict[str, str] = {}
    pending = sorted(nodes)
    for _ in range(4):                       # values hang off quantities, etc.
        for edge in edges:
            for a, b in ((edge["source"], edge["target"]), (edge["target"], edge["source"])):
                if a in nodes and a not in anchor and b in positions:
                    anchor[a] = b
        grouped: dict[str, list[str]] = {}
        for node in pending:
            if node in anchor and node not in positions:
                grouped.setdefault(anchor[node], []).append(node)
        for host, members in grouped.items():
            hx, hy = positions[host]
            scale = math.hypot(hx, hy) or 1.0
            ox, oy = hx / scale, hy / scale           # outward from the centre
            for index, node in enumerate(members):
                angle = 2 * math.pi * index / max(len(members), 1)
                radius = 0.35 + 0.02 * (index // 12)
                positions[node] = (hx + 0.3 * ox + radius * math.cos(angle),
                                   hy + 0.3 * oy + radius * math.sin(angle))
                depth[node] = depth.get(host, 0) + 1
    outer = max(depth.values(), default=0) + 1
    rest = [n for n in pending if n not in positions]
    for index, node in enumerate(rest):
        angle = 2 * math.pi * index / max(len(rest), 1)
        positions[node] = (outer * math.cos(angle), outer * math.sin(angle))
        depth[node] = outer


def _node_label(node_id: str, node: dict[str, Any]) -> str:
    ntype = node.get("ntype")
    if ntype == "TableKV":
        parts = [node.get("row_key"), node.get("column")]
        head = " / ".join(str(p) for p in parts if p)
        return f"{head} = {node.get('value')}" if head else str(node.get("value"))
    if ntype == "Quantity":
        return str(node.get("value_raw") or node.get("value_norm") or "quantity")
    if ntype == "VectorPoint":
        return f"vector ({node.get('collection')})"
    if ntype == "Paper" and node.get("title") == node_id.split(":", 1)[-1]:
        return str(node.get("source") or node.get("title"))
    for key in ("title", "label", "name", "citation_key"):
        value = node.get(key)
        if value:
            return str(value)
    text = " ".join(str(node.get("text") or "").split())
    if text:
        return text[:60] + ("…" if len(text) > 60 else "")
    return node_id.split(":")[-1]


ARTIFACT_GROUPS = (
    ("index/", "Index"),
    ("knowledge_graph/", "Knowledge graph"),
    ("documents/", "Documents"),
    ("corpus/", "Corpus"),
)
DEFAULT_ARTIFACT_GROUP = "Package"


def artifact_group(path: str) -> str:
    """Classify an artifact path into the section the console renders it under."""
    for prefix, name in ARTIFACT_GROUPS:
        if path.startswith(prefix):
            return name
    return DEFAULT_ARTIFACT_GROUP


def grouped_artifacts(job_id: str, artifacts: list[dict[str, Any]],
                      item_limit: int = 200) -> list[dict[str, Any]]:
    """Artifacts by output family, with counts and sizes.

    A large build has well over 100,000 per-document files; each group lists
    at most ``item_limit`` of them, and the bundle or the API serves the rest.
    """
    groups: dict[str, dict[str, Any]] = {}
    for item in artifacts:
        name = artifact_group(item["path"])
        group = groups.setdefault(name, {"name": name, "count": 0, "size_bytes": 0,
                                         "items": []})
        group["count"] += 1
        group["size_bytes"] += int(item.get("size_bytes") or 0)
        if len(group["items"]) < item_limit:
            group["items"].append({
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "media_type": item["media_type"],
                "download": f"/v1/jobs/{job_id}/artifacts/download?path={item['path']}",
            })
    return [groups[name] for _, name in ARTIFACT_GROUPS + (("", DEFAULT_ARTIFACT_GROUP),)
            if name in groups]
