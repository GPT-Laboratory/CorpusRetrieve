"""docparse — a generic, optimizable, reusable document-extraction component.

One call, any supported format, one provenance-bearing Element schema:

    from docparse import extract, extract_path, ExtractionConfig
    result = extract("report.pdf", ExtractionConfig(ocr="auto"))
    for el in result.elements:
        print(el.kind, el.page, el.text[:80])

Supported out of the box: PDF (PyMuPDF/pdfplumber + OCR fallback), DOCX, XLSX, CSV/TSV,
HTML, plain text/markdown, and images (OCR). Register your own backend via docparse.base.
"""
from __future__ import annotations

from docparse import extractors as _extractors  # noqa: F401  (registers built-ins on import)
from docparse.base import get_extractor, register, register_fn, registered_types
from docparse.models import Element, ExtractionConfig, ExtractionResult
from docparse.pipeline import extract, extract_path, extract_zip

__version__ = "0.1.0"
__all__ = [
    "extract", "extract_path", "extract_zip",
    "Element", "ExtractionResult", "ExtractionConfig",
    "register", "register_fn", "get_extractor", "registered_types",
]
