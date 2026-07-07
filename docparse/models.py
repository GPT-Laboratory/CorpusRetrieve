"""Generic, format-agnostic extraction schema.

Every extractor — PDF, DOCX, XLSX, CSV, HTML, image — emits the *same* list of
`Element`s. An Element is a provenance-bearing unit: it knows what it is (kind),
where it came from (source, page, bbox), what language it is in, and how confident
the extractor is. This is intentionally aligned with a knowledge-base ingestion
contract so the output drops straight into a RAG / evidence pipeline.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

# Controlled, domain-neutral element kinds.
ElementKind = Literal[
    "title",
    "heading",
    "narrative",
    "list_item",
    "table",
    "figure",
    "caption",
    "page_image",
    "reference",
    "formula",
    "other",
]


@dataclass
class Element:
    kind: ElementKind = "narrative"
    text: str = ""
    page: int | None = None                 # 1-based
    bbox: list[float] | None = None          # [x0, y0, x1, y1] in source coordinates
    language: str | None = None              # ISO-639-1 best guess
    confidence: float | None = None          # 0..1 (extractor's confidence)
    order: int = 0                           # reading order within the document
    extractor: str | None = None             # which extractor/backend produced it
    parent_id: int | None = None             # e.g. a caption -> its figure's order
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != {} and v != []}


@dataclass
class ExtractionResult:
    source: str
    mime: str = ""
    doc_type: str = ""                       # pdf | docx | xlsx | csv | html | text | image | ...
    page_count: int | None = None
    language: str | None = None
    elements: list[Element] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.elements:
            out[e.kind] = out.get(e.kind, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "mime": self.mime,
            "doc_type": self.doc_type,
            "page_count": self.page_count,
            "language": self.language,
            "counts": self.counts(),
            "warnings": self.warnings,
            "stats": self.stats,
            "elements": [e.to_dict() for e in self.elements],
        }


@dataclass
class ExtractionConfig:
    """Everything optimizable lives here; each backend reads only what it needs."""

    # dispatch / backends
    pdf_backend: Literal["auto", "pymupdf", "pdfplumber"] = "auto"

    # what to pull out
    extract_tables: bool = True
    extract_figures: bool = True
    emit_page_images: bool = False           # render full-page rasters as page_image elements

    # OCR (scanned pages / image files)
    ocr: Literal["auto", "on", "off"] = "auto"   # auto = only pages with < ocr_min_chars of text
    ocr_langs: str = "eng"                        # e.g. "eng+fin+swe" once traineddata is present
    ocr_min_chars: int = 20

    # rendering
    image_dpi: int = 200
    assets_dir: str | None = None            # where page-image / figure crops are written

    # language
    detect_language: bool = True

    # limits / heuristics
    max_pages: int | None = None
    heading_size_ratio: float = 1.15         # span font size vs page median -> heading
    table_max_rows: int = 200                # cap rows serialized to markdown (docx/xlsx/csv)

    # caption detection (FI/EN/SV), extend freely
    caption_regex: str = r"^(fig(?:ure)?|kuva|kuvio|taulukko|table|tabell|liite|appendix)\b"

    parameters: dict[str, Any] = field(default_factory=dict)  # escape hatch for custom backends
