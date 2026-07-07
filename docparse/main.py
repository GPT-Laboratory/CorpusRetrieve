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
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="docparse", description="Generic document extractor.")
    ap.add_argument("paths", nargs="+", help="files, directories, or .zip archives")
    ap.add_argument("--json", action="store_true", help="emit full JSON")
    ap.add_argument("--summary", action="store_true", help="emit per-file summary (default)")
    ap.add_argument("--backend", default="auto", choices=["auto", "pymupdf", "pdfplumber"])
    ap.add_argument("--ocr", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--ocr-langs", default="eng")
    ap.add_argument("--emit-page-images", action="store_true")
    ap.add_argument("--assets", default=None, help="directory for page/figure images")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--pages", type=int, default=None, help="max pages per PDF")
    args = ap.parse_args(argv)

    cfg = _build_config(args)
    all_results = []
    for path in args.paths:
        all_results.extend(extract_path(path, cfg))

    if args.json:
        json.dump([r.to_dict() for r in all_results], sys.stdout, ensure_ascii=False, indent=2)
        sys.stdout.write("\n")
    else:
        for r in all_results:
            counts = ", ".join(f"{k}:{v}" for k, v in sorted(r.counts().items()))
            print(f"{r.source}")
            print(f"    type={r.doc_type} pages={r.page_count} lang={r.language} "
                  f"backend={r.stats.get('backend', '-')}")
            print(f"    elements: {counts or '(none)'}")
            if r.warnings:
                print(f"    warnings: {len(r.warnings)} (e.g. {r.warnings[0][:80]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
