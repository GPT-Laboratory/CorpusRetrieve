"""Extraction fidelity harness — measure parsing quality, don't guess it.

Ground truth: next to each document ``X`` place ``X.gt.json``:

    {
      "reading_order": ["first paragraph ...", "second ...", ...],   # ordered text fragments
      "headings":      ["Introduction", "2. Methods", ...],
      "tables":        [{"page": 1, "rows": [["h1","h2"],["a","b"]]}, ...],
      "text":          "full expected text (optional, for OCR CER)"
    }

Metrics reported per document and aggregated:
  * heading_precision / heading_recall / heading_f1   (normalized text match)
  * reading_order_tau      Kendall's tau of matched gt fragments vs emitted order
  * table_cell_f1          best-pairing cell multiset F1 over structured views
  * ocr_cer                character error rate vs "text" (when given)

CLI:
    python -m docparse.bench golden_dir/ [--set pdf_backend=pdfplumber] [--json out.json]

Run it twice with different --set overrides and diff the numbers — every knob in
ExtractionConfig becomes a measured decision instead of a guess.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import fields
from pathlib import Path
from typing import Any

from docparse.models import ExtractionConfig, ExtractionResult
from docparse.pipeline import extract

_TEXT_KINDS = {"title", "heading", "narrative", "list_item", "caption"}


# --------------------------------------------------------------------------- utils

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s)).strip()


def _kendall_tau(seq: list[int]) -> float | None:
    """Tau over an observed ranking of items expected in ascending order."""
    n = len(seq)
    if n < 2:
        return None
    conc = disc = 0
    for i in range(n):
        for j in range(i + 1, n):
            if seq[i] < seq[j]:
                conc += 1
            elif seq[i] > seq[j]:
                disc += 1
    total = n * (n - 1) / 2
    return round((conc - disc) / total, 4) if total else None


def _cer(pred: str, gt: str) -> float | None:
    """Character error rate via Levenshtein distance (O(len*len), fine for pages)."""
    a, b = _norm(pred), _norm(gt)
    if not b:
        return None
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return round(prev[-1] / len(b), 4)


# --------------------------------------------------------------------------- metrics

def heading_prf(result: ExtractionResult, gt_headings: list[str]) -> dict[str, float | None]:
    pred = [_norm(e.text) for e in result.elements if e.kind in ("heading", "title") and e.text]
    gt = [_norm(h) for h in gt_headings]
    if not gt and not pred:
        return {"heading_precision": None, "heading_recall": None, "heading_f1": None}
    gt_left = list(gt)
    tp = 0
    for p in pred:
        if p in gt_left:
            gt_left.remove(p)
            tp += 1
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"heading_precision": round(prec, 4), "heading_recall": round(rec, 4),
            "heading_f1": round(f1, 4)}


def reading_order_tau(result: ExtractionResult, gt_fragments: list[str]) -> dict[str, Any]:
    """Match each gt fragment to the first element containing it; tau over positions."""
    elements = [(i, _norm(e.text)) for i, e in enumerate(result.elements)
                if e.kind in _TEXT_KINDS and e.text]
    positions, used = [], set()
    matched = 0
    for frag in gt_fragments:
        nf = _norm(frag)
        if not nf:
            continue
        hit = next((i for i, t in elements if i not in used and nf in t), None)
        if hit is not None:
            used.add(hit)
            positions.append(hit)
            matched += 1
    tau = _kendall_tau(positions)
    cov = round(matched / len(gt_fragments), 4) if gt_fragments else None
    return {"reading_order_tau": tau, "reading_order_coverage": cov}


def _cells(rows: list[list[Any]]) -> list[str]:
    return [c for r in rows for c in (_norm(str(x)) for x in r) if c]


def _pair_f1(pred_rows, gt_rows) -> float:
    p, g = _cells(pred_rows), _cells(gt_rows)
    if not p and not g:
        return 1.0
    g_left = list(g)
    tp = 0
    for c in p:
        if c in g_left:
            g_left.remove(c)
            tp += 1
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0


def table_cell_f1(result: ExtractionResult, gt_tables: list[dict]) -> dict[str, Any]:
    if not gt_tables:
        return {"table_cell_f1": None, "tables_found": None}
    preds = []
    for e in result.elements:
        if e.kind == "table" and e.structured:
            preds.append((e.page, [e.structured.get("header", [])] + e.structured.get("rows", [])))
    scores, taken = [], set()
    for gt in gt_tables:
        gt_rows = gt.get("rows", [])
        gt_page = gt.get("page")
        best, best_i = 0.0, None
        for i, (page, rows) in enumerate(preds):
            if i in taken or (gt_page is not None and page is not None and page != gt_page):
                continue
            f1 = _pair_f1(rows, gt_rows)
            if f1 > best:
                best, best_i = f1, i
        if best_i is not None:
            taken.add(best_i)
        scores.append(best)
    return {"table_cell_f1": round(sum(scores) / len(scores), 4),
            "tables_found": len(preds), "tables_expected": len(gt_tables)}


def score(result: ExtractionResult, gt: dict) -> dict[str, Any]:
    out: dict[str, Any] = {"source": result.source}
    out.update(heading_prf(result, gt.get("headings", [])))
    out.update(reading_order_tau(result, gt.get("reading_order", [])))
    out.update(table_cell_f1(result, gt.get("tables", [])))
    if gt.get("text"):
        pred = " ".join(e.text for e in result.elements if e.kind in _TEXT_KINDS and e.text)
        out["ocr_cer"] = _cer(pred, gt["text"])
    return out


# --------------------------------------------------------------------------- runner

_METRIC_KEYS = ("heading_precision", "heading_recall", "heading_f1",
                "reading_order_tau", "reading_order_coverage", "table_cell_f1", "ocr_cer")


def run(golden_dir: str, cfg: ExtractionConfig) -> dict[str, Any]:
    rows = []
    for gt_path in sorted(Path(golden_dir).rglob("*.gt.json")):
        doc = gt_path.with_name(gt_path.name[:-len(".gt.json")])
        if not doc.exists():
            continue
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        rows.append(score(extract(str(doc), cfg), gt))
    agg = {}
    for k in _METRIC_KEYS:
        vals = [r[k] for r in rows if r.get(k) is not None]
        agg[k] = round(sum(vals) / len(vals), 4) if vals else None
    return {"documents": rows, "aggregate": agg, "n_documents": len(rows)}


def _apply_sets(cfg: ExtractionConfig, sets: list[str]) -> ExtractionConfig:
    valid = {f.name: f.type for f in fields(ExtractionConfig)}
    for kv in sets:
        k, _, v = kv.partition("=")
        if k not in valid:
            raise SystemExit(f"unknown config field: {k}")
        cur = getattr(cfg, k)
        if isinstance(cur, bool):
            v = v.lower() in ("1", "true", "yes", "on")
        elif isinstance(cur, int):
            v = int(v)
        elif isinstance(cur, float):
            v = float(v)
        elif isinstance(cur, tuple):
            v = tuple(x.strip() for x in v.split(","))
        setattr(cfg, k, v)
    return cfg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docparse.bench",
                                 description="Extraction fidelity benchmark.")
    ap.add_argument("golden_dir", help="directory of documents with X.gt.json ground truth")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override an ExtractionConfig field (repeatable)")
    ap.add_argument("--json", default=None, help="write full report to this file")
    args = ap.parse_args(argv)

    cfg = _apply_sets(ExtractionConfig(), args.set)
    report = run(args.golden_dir, cfg)
    report["config_overrides"] = args.set

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                   encoding="utf-8")
    agg = report["aggregate"]
    print(f"documents: {report['n_documents']}"
          + (f"   overrides: {' '.join(args.set)}" if args.set else ""))
    for k in _METRIC_KEYS:
        v = agg.get(k)
        print(f"  {k:24s} {'-' if v is None else v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
