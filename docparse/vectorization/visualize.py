"""Interactive knowledge-graph visualization.

Renders the Stage-5 graph export (knowledge_graph.json) as a self-contained
HTML file (pyvis / vis-network) that opens in any browser — no Neo4j needed.

The full graph is dominated by bookkeeping nodes (VectorPoint, Quantity,
TableKV, ReferenceWork), so the default view hides those and shows the
skeleton: Paper -> Sections -> Chunks/Claims/Figures/Tables/Formulas plus
support and citation edges. Flags bring the rest back.

CLI:
    python -m docparse.vectorization.visualize index/knowledge_graph.json
    python -m docparse.vectorization.visualize index/knowledge_graph.json \\
        --out kg.html --include Quantity Unit
    python -m docparse.vectorization.visualize index/knowledge_graph.json --all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_HIDDEN = {"VectorPoint", "Quantity", "Unit", "TableKV", "ReferenceWork",
                  "Entity"}

COLORS = {
    "Paper": "#e63946", "Section": "#457b9d", "Chunk": "#a8dadc",
    "Claim": "#f4a261", "Figure": "#2a9d8f", "FigurePanel": "#8ab17d",
    "Table": "#6d597a", "TableRowGroup": "#b56576", "TableKV": "#e0afa0",
    "Formula": "#ffb703", "ReferenceWork": "#adb5bd", "Quantity": "#cdb4db",
    "Unit": "#ffc8dd", "Entity": "#bde0fe", "VectorPoint": "#dee2e6",
}
SIZES = {"Paper": 40, "Section": 22, "Figure": 18, "Table": 18, "Claim": 16,
         "Formula": 16}


def build_html(graph_json: str | Path, out_html: str | Path,
               hidden: set[str] | None = None, physics: bool = True) -> str:
    from pyvis.network import Network

    hidden = DEFAULT_HIDDEN if hidden is None else hidden
    data = json.loads(Path(graph_json).read_text(encoding="utf-8"))
    keep = {n["id"]: n for n in data["nodes"] if n.get("ntype") not in hidden}

    net = Network(height="920px", width="100%", directed=True,
                  bgcolor="#ffffff", cdn_resources="in_line",
                  select_menu=True, filter_menu=True)
    for nid, n in keep.items():
        ntype = n.get("ntype", "Node")
        label = (n.get("label") or n.get("title") or n.get("name")
                 or n.get("citation_key") or nid.split(":")[-1])
        tip = [f"<b>{ntype}</b>", nid]
        if n.get("role"):
            tip.append(f"role: {n['role']}")
        if n.get("text"):
            tip.append(str(n["text"])[:220] + "…")
        net.add_node(nid, label=str(label)[:40], title="\n".join(tip),
                     color=COLORS.get(ntype, "#999999"),
                     size=SIZES.get(ntype, 10), group=ntype)
    shown_edges = 0
    for e in data["edges"]:
        if e["source"] in keep and e["target"] in keep:
            net.add_edge(e["source"], e["target"], title=e.get("rel", ""),
                         color="#c0c0c0", arrows="to", width=1)
            shown_edges += 1
    if physics:
        net.barnes_hut(gravity=-2500, central_gravity=0.25,
                       spring_length=140, damping=0.85)
    # pyvis' write_html uses the platform default encoding (cp1252 on Windows),
    # which chokes on °C, en-dashes, Greek symbols, etc. Generate the HTML and
    # write it as UTF-8 ourselves so it works cross-platform.
    html = net.generate_html(notebook=False)
    Path(out_html).write_text(html, encoding="utf-8")
    return (f"{len(keep)} nodes / {shown_edges} edges rendered "
            f"(hidden types: {', '.join(sorted(hidden)) or 'none'})")


def main() -> int:
    ap = argparse.ArgumentParser(prog="docparse.vectorization.visualize")
    ap.add_argument("graph_json", help="knowledge_graph.json from the index dir")
    ap.add_argument("--out", default="knowledge_graph.html")
    ap.add_argument("--include", nargs="*", default=[],
                    help="node types to un-hide, e.g. Quantity Unit TableKV")
    ap.add_argument("--all", action="store_true", help="show every node type")
    args = ap.parse_args()
    hidden = set() if args.all else DEFAULT_HIDDEN - set(args.include)
    try:
        msg = build_html(args.graph_json, args.out, hidden)
    except ImportError:
        print("pyvis is required: pip install pyvis", file=sys.stderr)
        return 1
    print(msg)
    print(f"open {args.out} in a browser")
    return 0


if __name__ == "__main__":
    sys.exit(main())
