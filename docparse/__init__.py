"""docparse — a generic, optimizable, reusable document-extraction component.

One call, any supported format, one provenance-bearing Element schema:

    from docparse import extract, extract_path, ExtractionConfig
    result = extract("document.pdf", ExtractionConfig(ocr="auto"))
    for el in result.elements:
        print(el.kind, el.page, el.text[:80])

Supported out of the box: PDF (PyMuPDF/pdfplumber + OCR fallback), DOCX, XLSX, CSV/TSV,
HTML, plain text/markdown, and images (OCR).
"""
from __future__ import annotations

from docparse import extractors as _extractors  # noqa: F401  (registers built-ins on import)
from docparse.chunking import Chunk, ChunkingConfig, ChunkingResult, chunk_result
from docparse.models import Element, ExtractionConfig, ExtractionResult
from docparse.pipeline import extract, extract_path, extract_zip
from docparse.router import get_extractor, register, register_fn, registered_types

__version__ = "1.0.0"
__all__ = [
    "extract", "extract_path", "extract_zip",
    "chunk_result", "Chunk", "ChunkingConfig", "ChunkingResult",
    "Element", "ExtractionResult", "ExtractionConfig",
    "register", "register_fn", "get_extractor", "registered_types",
]
