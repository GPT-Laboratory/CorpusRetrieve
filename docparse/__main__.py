"""CLI:  python -m docparse <path...> [--summary|--json] [options]"""
from __future__ import annotations

import argparse
import json
import sys

from docparse.models import ExtractionConfig
from docparse.pipeline import extract_path


def _build_config(args) -> ExtractionConfig:
    return ExtractionConfig(
        pdf_backend=args.backend, ocr=args.ocr, ocr_langs=args.ocr_langs,
        emit_page_images=args.emit_page_images, assets_dir=args.assets,
        image_dpi=args.dpi, max_pages=args.pages,
        table_serializations=tuple(args.table_serializations.split(",")),
        remove_boilerplate=not args.keep_boilerplate,
        column_aware=not args.no_columns,
        per_element_language=args.per_element_language,
        scientific=not args.no_scientific,
        table_fallback=args.table_fallback,
    )


def _maybe_describe(result, args):
    if not args.describe_figures:
        return
    from docparse.enrich import describe_figures, make_anthropic_describer
    try:
        n = describe_figures(result, make_anthropic_describer(model=args.vlm_model),
                             kinds=("figure", "page_image"))
        print(f"[vlm] {result.source}: {n} figure(s) described", file=sys.stderr)
    except Exception as exc:
        print(f"[vlm] skipped ({exc}) — is ANTHROPIC_API_KEY set and 'anthropic' installed?",
              file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docparse", description="Generic document extractor.")
    ap.add_argument("paths", nargs="+", help="files, directories, or .zip archives")
    ap.add_argument("--json", action="store_true", help="emit full JSON")
    ap.add_argument("--summary", action="store_true", help="emit per-file summary (default)")
    ap.add_argument("--out", default=None, help="write output to this file as UTF-8 (bypasses console encoding)")
    ap.add_argument("--backend", default="auto", choices=["auto", "pymupdf", "pdfplumber"])
    ap.add_argument("--ocr", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--ocr-langs", default="eng")
    ap.add_argument("--emit-page-images", action="store_true")
    ap.add_argument("--assets", default=None, help="directory for page/figure images")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pages", type=int, default=None, help="max pages per PDF")
    ap.add_argument("--table-serializations", default="markdown",
                    help="comma list of markdown,json,kv (first fills Element.text)")
    ap.add_argument("--keep-boilerplate", action="store_true",
                    help="do not strip repeated page headers/footers")
    ap.add_argument("--no-columns", action="store_true",
                    help="disable XY-cut column-aware reading order")
    ap.add_argument("--per-element-language", action="store_true")
    ap.add_argument("--no-scientific", action="store_true",
                    help="disable scientific enrichment (dehyphenation, roles, citations, claims)")
    ap.add_argument("--table-fallback", default="text", choices=["text", "none"],
                    help="caption-anchored detection for unruled tables (default: text)")
    ap.add_argument("--describe-figures", action="store_true",
                    help="describe figure crops panel-by-panel with a VLM "
                         "(needs ANTHROPIC_API_KEY and 'pip install anthropic'; implies --assets)")
    ap.add_argument("--vlm-model", default="claude-sonnet-4-6")
    args = ap.parse_args(argv)

    # Make console output UTF-8 regardless of the OS default code page (fixes Windows cp1252).
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if args.describe_figures and not args.assets:
        args.assets = "assets"                    # crops must exist on disk for the VLM
    cfg = _build_config(args)
    all_results = []
    for path in args.paths:
        all_results.extend(extract_path(path, cfg))
    for r in all_results:
        _maybe_describe(r, args)

    if args.json:
        payload = json.dumps([r.to_dict() for r in all_results], ensure_ascii=False, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(payload + "\n")
        else:
            sys.stdout.write(payload + "\n")
    else:
        lines = []
        for r in all_results:
            counts = ", ".join(f"{k}:{v}" for k, v in sorted(r.counts().items()))
            lines.append(f"{r.source}")
            lines.append(f"    type={r.doc_type} pages={r.page_count} lang={r.language} "
                         f"backend={r.stats.get('backend', '-')}")
            lines.append(f"    elements: {counts or '(none)'}")
            if r.warnings:
                lines.append(f"    warnings: {len(r.warnings)} (e.g. {r.warnings[0][:80]})")
        text = "\n".join(lines) + "\n"
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
