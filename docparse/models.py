"""Generic, format-agnostic extraction schema.

Every extractor — PDF, DOCX, XLSX, CSV, HTML, image — emits the *same* list of
`Element`s. An Element is a provenance-bearing unit: it knows what it is (kind),
where it came from (source, page, bbox, char offsets), what language it is in,
which section of the document it belongs to, and how confident the extractor is.

Complex elements (tables, figures) carry up to three linked views:
  * ``text``       — normalized text view (markdown for tables)
  * ``structured`` — machine-usable view (header + rows for tables)
  * ``visual_ref`` — provenance pointer to the source region (bbox, crop path)
"""
from __future__ import annotations

import uuid
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
    "page_header",     # running header (boilerplate; suppressed by default)
    "page_footer",     # running footer / page number (boilerplate)
    "reference",
    "formula",
    "other",
]


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Element:
    kind: ElementKind = "narrative"
    text: str = ""
    page: int | None = None                  # 1-based
    bbox: list[float] | None = None          # [x0, y0, x1, y1] in source coordinates
    language: str | None = None              # ISO-639-1 best guess
    confidence: float | None = None          # 0..1 (extractor's confidence)
    order: int = 0                           # reading order within the document
    extractor: str | None = None             # which extractor/backend produced it

    # identity & structure
    id: str = field(default_factory=_new_id)  # stable handle for cross-references
    parent_id: str | None = None              # e.g. caption -> its figure/table id
    level: int | None = None                  # heading level (1 = top); None otherwise
    section_path: list[str] = field(default_factory=list)  # heading ancestry

    # three-view contract for complex elements
    structured: dict[str, Any] | None = None  # e.g. {"header": [...], "rows": [[...]]}
    visual_ref: dict[str, Any] | None = None  # e.g. {"page": 3, "bbox": [...], "image_path": ...}

    # late-chunking support: offsets into the concatenated document text
    char_start: int | None = None
    char_end: int | None = None

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

    def document_text(self) -> str:
        """The concatenated text that char_start/char_end offsets index into."""
        return "\n\n".join(e.text for e in self.elements if e.text)

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

    # table serializations — "markdown" fills Element.text; extra ones land in
    # metadata["serializations"] so retrieval A/B tests over representations are trivial.
    table_serializations: tuple[str, ...] = ("markdown",)   # of: markdown | json | kv

    # boilerplate (running headers/footers/page numbers) — PDF only
    remove_boilerplate: bool = True
    boilerplate_zone: float = 0.12           # top/bottom fraction of page height scanned
    boilerplate_min_pages: int = 3           # min repetitions before a block counts as boilerplate
    boilerplate_min_ratio: float = 0.4       # fraction of pages the block must appear on

    # reading order
    column_aware: bool = True                # recursive XY-cut instead of naive y,x sort

    # vector figures: cluster vector drawings (charts/maps/diagrams drawn as paths,
    # invisible to raster-image detection) into figure regions with crops
    vector_figures: bool = True
    vector_figure_min_area: float = 20000.0   # pt² — ignore rules/ornaments
    vector_figure_min_dim: float = 60.0       # pt  — ignore thin separators
    figure_max_text_coverage: float = 0.25    # region mostly covered by text = a text box, not a figure
    # figure-dominated pages (schematics, maps, posters, engineering drawings):
    # a page that is mostly vector ink with little running text is one big
    # figure, not a table. When detected, we keep the whole-page figure (for a
    # VLM to read) and don't let phantom "tables" from grid/border lines
    # suppress it. Heuristic: many vector paths, few text chars, and the paths
    # span most of the page.
    figure_dominant_detection: bool = True
    figure_dominant_min_paths: int = 150      # dense line-art (schematics have 100s)
    figure_dominant_max_chars: int = 4000     # little running text (labels only)
    figure_dominant_min_coverage: float = 0.45  # vector cluster spans ≥45% of page

    # publisher/license noise — a DATA list, not logic; extend per corpus
    noise_patterns: tuple[str, ...] = (
        r"wiley online library", r"terms and conditions", r"onlinelibrary\.wiley\.com",
        r"downloaded (?:from|by)", r"creative ?commons licen[cs]e", r"creativecommons\.org",
        r"rules of use.*oa articles", r"governed by the applicable", r"see the terms",
        r"©\s*(?:19|20)\d{2}\s+the\s+authors?\b", r"open access article under the terms",
        r"all rights reserved", r"authorized licensed use",          # IEEE
        r"content may change prior to final publication",            # IEEE early access
        r"springer(?:link)? (?:nature|terms)", r"elsevier (?:b\.v\.|ltd|inc)",
        r"provided by .{0,40}repository", r"^\s*\d{7,},?\s*(19|20)\d{2}",
        r"open access article under the terms", r"©\s*(19|20)\d{2}\s*the author",
        r"^\s*\d+\s+of\s+\d+\s*$",                        # "1 of 17" page footer
        r"https?://doi\.org/\S+",
        r"^\s*(?:international\s+)?journal of [a-z]+[a-z ]{0,40}$",  # journal footer line
    )

    # unruled tables: caption-anchored text-strategy detection when ruled detection finds none
    table_fallback: str = "text"             # "text" | "none"
    table_caption_regex: str = r"^\s*tab(?:le|ell|ulukko)\s*\d"

    # scientific-document enrichment (dehyphenation, license-noise removal, equations,
    # roles, paper metadata, citation graph, quantities/claims) — see docparse.scientific
    scientific: bool = True

    # captions
    link_captions: bool = True
    caption_max_gap: float = 80.0            # max pt distance between caption and figure/table

    # OCR (scanned pages / image files)
    ocr: Literal["auto", "on", "off"] = "auto"   # auto = only pages with < ocr_min_chars of text
    ocr_langs: str = "eng"                        # e.g. "eng+fin+swe" once traineddata is present
    ocr_min_chars: int = 20
    ocr_preprocess: bool = True              # grayscale + autocontrast before Tesseract
    ocr_auto_orient: bool = True             # use Tesseract OSD to fix rotated scans
    ocr_line_elements: bool = True           # emit per-line OCR elements with word bboxes

    # rendering
    image_dpi: int = 200
    assets_dir: str | None = None            # where page-image / figure / table crops are written

    # language
    detect_language: bool = True
    per_element_language: bool = False       # detect language per element (code-switched docs)

    # limits / heuristics
    max_pages: int | None = None
    heading_size_ratio: float = 1.15         # span font size vs page median -> heading
    table_max_rows: int = 200                # cap rows serialized to markdown (docx/xlsx/csv)

    # caption detection (FI/EN/SV), extend freely
    caption_regex: str = r"^(fig(?:ure)?\.?|kuva|kuvio|taulukko|table|tabell|liite|appendix)\b"

    # list detection (bullets / numbered)
    list_regex: str = r"^\s*(?:[-*•‣▪]|\(?\d{1,3}[.)]|\(?[a-z][.)])\s+"

    parameters: dict[str, Any] = field(default_factory=dict)  # escape hatch for custom backends
