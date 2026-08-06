from __future__ import annotations

from typing import Any

MAX_ELEMENT_TEXT = 600
MAX_NODE_TEXT = 280
DEFAULT_HIDDEN_NODE_TYPES = ("VectorPoint", "Quantity", "Unit", "TableKV")


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


def graph_view(graph: dict[str, Any], hidden: set[str]) -> dict[str, Any]:
    """Knowledge graph shaped for in-browser exploration.

    Bookkeeping node types are hidden by default so the layout stays readable,
    and every surviving node carries its degree and attributes to avoid a second
    request when the user inspects it.
    """
    kept = {node["id"] for node in graph["nodes"] if node.get("ntype") not in hidden}
    edges = [
        {"source": edge["source"], "target": edge["target"], "rel": edge.get("rel")}
        for edge in graph["edges"]
        if edge["source"] in kept and edge["target"] in kept
    ]

    degree: dict[str, int] = {}
    for edge in edges:
        degree[edge["source"]] = degree.get(edge["source"], 0) + 1
        degree[edge["target"]] = degree.get(edge["target"], 0) + 1

    nodes = [
        {
            "id": node["id"],
            "type": node.get("ntype"),
            "label": _node_label(node),
            "role": node.get("role"),
            "text": (node.get("text") or "")[:MAX_NODE_TEXT],
            "pages": node.get("pages"),
            "degree": degree.get(node["id"], 0),
            "attrs": {
                key: value
                for key, value in node.items()
                if key not in {"id", "ntype", "text"} and value not in (None, "", [])
            },
        }
        for node in graph["nodes"]
        if node["id"] in kept
    ]

    all_types: dict[str, int] = {}
    for node in graph["nodes"]:
        all_types[node.get("ntype", "unknown")] = (
            all_types.get(node.get("ntype", "unknown"), 0) + 1
        )

    return {
        "nodes": nodes,
        "edges": edges,
        "hidden_types": sorted(hidden),
        "all_types": all_types,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }


def _node_label(node: dict[str, Any]) -> str:
    for key in ("label", "title", "name", "citation_key"):
        value = node.get(key)
        if value:
            return str(value)
    return str(node["id"]).split(":")[-1]


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


def grouped_artifacts(job_id: str, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in artifacts:
        groups.setdefault(artifact_group(item["path"]), []).append(
            {
                "path": item["path"],
                "size_bytes": item["size_bytes"],
                "media_type": item["media_type"],
                "download": f"/v1/jobs/{job_id}/artifacts/download?path={item['path']}",
            }
        )
    order = [name for _, name in ARTIFACT_GROUPS] + [DEFAULT_ARTIFACT_GROUP]
    return [{"name": name, "items": groups[name]} for name in order if groups.get(name)]
