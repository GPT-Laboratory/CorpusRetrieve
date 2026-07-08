"""Verify every enrichment layer against a real paper, with printed evidence.

Usage:
    python verify_layers.py [path/to/paper.pdf]      (default: mydocs/climate_paper.pdf)

Each check maps 1:1 to a row of the requirements table and prints PASS/FAIL
plus the actual extracted evidence, so you can eyeball quality — not just counts.
"""
from __future__ import annotations

import json
import sys

from docparse import ExtractionConfig, extract

GREEN, RED, YELLOW, END = "\033[92m", "\033[91m", "\033[93m", "\033[0m"
results: list[tuple[str, bool | None]] = []


def check(name: str, ok: bool | None, evidence: str = "") -> None:
    tag = (f"{GREEN}PASS{END}" if ok else f"{RED}FAIL{END}") if ok is not None else f"{YELLOW}HOOK{END}"
    print(f"\n[{tag}] {name}")
    if evidence:
        for line in evidence.splitlines():
            print(f"       {line}")
    results.append((name, ok))


def main(path: str) -> int:
    cfg = ExtractionConfig(table_serializations=("markdown", "kv"), assets_dir="assets")
    r = extract(path, cfg)
    print(f"parsed {path}: {r.page_count} pages, {len(r.elements)} elements, "
          f"counts={r.counts()}")

    # 1 ── Structured table extraction ────────────────────────────────────────
    tables = [e for e in r.elements if e.kind == "table" and e.structured]
    t1 = tables[0] if tables else None
    ok = bool(t1 and len(t1.structured["rows"]) >= 10 and len(t1.structured["header"]) >= 4)
    ev = ""
    if t1:
        ev = (f"tables={len(tables)}  |  Table@p{t1.page}: {len(t1.structured['rows'])} rows x "
              f"{len(t1.structured['header'])} cols\nheader: {t1.structured['header']}\n"
              f"row 1 : {t1.structured['rows'][0]}\n"
              f"kv    : {t1.metadata.get('serializations', {}).get('kv', '').splitlines()[0] if t1.metadata.get('serializations') else '(add kv to table_serializations)'}")
        # the query the critique said must become answerable:
        try:
            hdr = t1.structured["header"]
            col = next(i for i, h in enumerate(hdr) if "2.0" in h)
            name_col = next(i for i, h in enumerate(hdr) if "model" in h.lower())
            earliest = min(t1.structured["rows"], key=lambda row: int(row[col]))
            ev += f"\nQUERY 'which model reaches 2.0°C earliest?' -> {earliest[name_col]} ({earliest[col]})"
        except (StopIteration, ValueError):
            pass
    check("Structured table extraction (Table 1 as rows/columns)", ok, ev)

    # 2 ── Figure understanding (VLM hook) ────────────────────────────────────
    figs = [e for e in r.elements if e.kind == "figure"]
    described = r.stats.get("figures_described", 0)
    crops = sum(1 for e in figs if (e.visual_ref or {}).get("image_path"))
    check("Figure understanding (panel-by-panel)",
          True if described else None,
          f"figures={len(figs)}, crops saved={crops}, described by VLM={described}\n"
          "to fill: pip install anthropic; set ANTHROPIC_API_KEY; then\n"
          "  from docparse.enrich import describe_figures, make_anthropic_describer\n"
          "  describe_figures(result, make_anthropic_describer())")

    # 3 ── Equation extraction ────────────────────────────────────────────────
    eqs = [e for e in r.elements if e.kind == "formula"]
    ev = "\n".join(f"eq#{e.structured.get('number')}: {e.structured.get('expression','')[:70]}"
                   for e in eqs[:5])
    check("Equation extraction (number + expression + context ids)", len(eqs) >= 2, ev)

    # 4 ── Citation graph ─────────────────────────────────────────────────────
    cit = r.stats.get("citations", {})
    graph = r.stats.get("citation_graph", {})
    sample = next(iter(graph.items()), None)
    ev = (f"reference entries={r.stats.get('reference_entries')}  "
          f"unique inline keys={cit.get('unique_keys')}  linked={cit.get('linked_to_reference')}")
    if sample:
        k, v = sample
        ev += f"\nexample: '{k}' -> ref element {v['ref_id']} ({len(v['mentions'])} mention(s))"
    check("Citation graph (inline -> reference entry)",
          bool(cit.get("linked_to_reference", 0) >= 10), ev)

    # 5 ── Metadata schema ────────────────────────────────────────────────────
    paper = r.stats.get("paper", {})
    need = ["doi", "title", "year"]
    check("Metadata schema (DOI/title/journal/year/authors/dates)",
          all(k in paper for k in need),
          json.dumps({k: paper.get(k) for k in
                      ("doi", "title", "authors_raw", "journal", "year",
                       "received", "accepted", "keywords")}, ensure_ascii=False, indent=1))

    # 6 ── Claim extraction ───────────────────────────────────────────────────
    claims = r.stats.get("claims", [])
    ev = "\n".join(f"[{c['role']} p{c['page']}] {c['text'][:95]}" for c in claims[:3])
    check("Claim extraction (searchable findings with provenance)", len(claims) >= 3, ev)

    # 7 ── Method extraction ──────────────────────────────────────────────────
    meth = [e for e in r.elements if e.metadata.get("role") == "methods" and e.text]
    meth_q = [q for e in meth for q in e.metadata.get("quantities", [])]
    check("Method extraction (methods-role elements + their thresholds)",
          len(meth) >= 5,
          f"methods-role elements={len(meth)}, quantities inside methods={len(meth_q)}\n"
          f"sample thresholds: {meth_q[:6]}\n"
          "note: a structured method schema (datasets/models/assumptions) is the one layer\n"
          "best filled by an LLM over exactly these methods-role chunks")

    # 8 ── Result extraction ──────────────────────────────────────────────────
    qty_els = [e for e in r.elements if e.metadata.get("quantities")]
    flat = [q for e in qty_els for q in e.metadata["quantities"]]
    ranges = [q for q in flat if "–" in q or "-" in q or "once in" in q.lower()]
    check("Result extraction (60–80%, once-in-N-years, °C ranges)",
          len(qty_els) >= 10 and len(ranges) >= 3,
          f"elements with quantities={len(qty_els)}; samples: {ranges[:5] or flat[:5]}")

    # 9 ── Uncertainty extraction ─────────────────────────────────────────────
    unc = [e for e in r.elements if e.metadata.get("uncertainty")]
    check("Uncertainty extraction (±, inter-model spread, confidence)",
          len(unc) >= 2,
          f"uncertainty-flagged elements={len(unc)}\n"
          + "\n".join(f"p{e.page}: {e.text[:90]}" for e in unc[:2]))

    # 10 ── Noise removal ─────────────────────────────────────────────────────
    residual = sum(1 for e in r.elements
                   if "Wiley Online" in e.text or "Terms and Conditions" in e.text)
    check("Noise removal (license/download/footer boilerplate)",
          residual == 0,
          f"license noise removed={r.stats.get('license_noise_removed', 0)}, "
          f"running headers/footers removed={r.stats.get('boilerplate_removed', 0)}, "
          f"residual license text={residual}\n"
          f"hyphenation fixes={r.stats.get('hyphenation_fixes', 0)}")

    # 11 ── Chunking by scientific role ───────────────────────────────────────
    dist = r.stats.get("role_distribution", {})
    core = {"introduction", "methods", "conclusion"} | ({"results"} if "results" in dist else {"discussion"})
    check("Chunking by scientific role (role tag on every element)",
          core.issubset(dist.keys()),
          f"distribution: {dist}\n"
          "downstream: group elements by el.metadata['role'] to chunk per section type")

    # ── summary ──────────────────────────────────────────────────────────────
    passed = sum(1 for _, ok in results if ok)
    hooks = sum(1 for _, ok in results if ok is None)
    failed = sum(1 for _, ok in results if ok is False)
    print(f"\n{'=' * 70}\n{passed} PASS   {failed} FAIL   {hooks} HOOK (needs VLM/API key)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "mydocs/climate_paper.pdf"))
