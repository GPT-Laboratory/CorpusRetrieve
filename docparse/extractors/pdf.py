"""PDF extraction. Primary backend PyMuPDF (text+tables+images+render), pdfplumber fallback.

Per page it produces reading-order elements: headings, narrative blocks, tables (markdown),
figures (bbox + linked caption), and — for scanned pages or on request — page-image rasters
with OCR text. Everything is provenance-bearing (page + bbox) and confidence-tagged.
"""
from __future__ import annotations

import io
import os
import re
from statistics import median

from docparse.base import register
from docparse.lang import detect_language
from docparse.models import Element, ExtractionConfig, ExtractionResult

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False


def _center_in(bbox, box) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _ocr(png_bytes: bytes, langs: str) -> tuple[str, float]:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        try:
            data = pytesseract.image_to_data(img, lang=langs, output_type=pytesseract.Output.DICT)
        except Exception:
            data = pytesseract.image_to_data(img, lang="eng", output_type=pytesseract.Output.DICT)
        words, confs = [], []
        for w, c in zip(data.get("text", []), data.get("conf", [])):
            if w and w.strip():
                words.append(w)
                try:
                    ci = float(c)
                    if ci >= 0:
                        confs.append(ci)
                except ValueError:
                    pass
        text = " ".join(words)
        conf = round(sum(confs) / len(confs) / 100.0, 3) if confs else None
        return text, conf
    except Exception:
        return "", None


def _pymupdf_extract(path: str, cfg: ExtractionConfig, result: ExtractionResult) -> None:
    doc = fitz.open(path)
    result.page_count = doc.page_count
    caption_re = re.compile(cfg.caption_regex, re.IGNORECASE)
    order = 0
    doc_text_sample: list[str] = []
    assets = cfg.assets_dir
    if assets:
        os.makedirs(assets, exist_ok=True)

    n_pages = doc.page_count if cfg.max_pages is None else min(doc.page_count, cfg.max_pages)
    for pno in range(n_pages):
        page = doc[pno]
        pd = page.get_text("dict")
        raw_text = page.get_text("text")

        # --- scanned page? OCR fallback ---
        is_scanned = len(raw_text.strip()) < cfg.ocr_min_chars
        do_ocr = cfg.ocr == "on" or (cfg.ocr == "auto" and is_scanned)

        # --- tables first (so we can suppress their text blocks) ---
        table_boxes = []
        if cfg.extract_tables:
            try:
                for t in page.find_tables().tables:
                    md = t.to_markdown()
                    bbox = list(t.bbox)
                    table_boxes.append(bbox)
                    order += 1
                    result.elements.append(Element(
                        kind="table", text=md.strip(), page=pno + 1, bbox=bbox,
                        order=order, extractor="pymupdf",
                        metadata={"rows": getattr(t, "row_count", None), "cols": getattr(t, "col_count", None)}))
            except Exception as exc:
                result.warnings.append(f"table extraction failed on page {pno + 1}: {exc}")

        # --- text blocks (skip those inside tables) ---
        sizes = [s["size"] for b in pd["blocks"] if b.get("type") == 0
                 for ln in b.get("lines", []) for s in ln.get("spans", [])]
        med = median(sizes) if sizes else 0.0
        items = []  # (y0, x0, Element)
        for b in pd["blocks"]:
            if b.get("type") != 0:
                continue
            bbox = list(b["bbox"])
            if any(_center_in(bbox, tb) for tb in table_boxes):
                continue
            spans = [s for ln in b.get("lines", []) for s in ln.get("spans", [])]
            line_texts = []
            for ln in b.get("lines", []):
                lt = "".join(s["text"] for s in ln.get("spans", []))
                if lt.strip():
                    line_texts.append(lt.strip())
            btext = " ".join(line_texts).strip()
            if not btext:
                continue
            max_size = max((s["size"] for s in spans), default=0.0)
            kind = "heading" if med and max_size >= med * cfg.heading_size_ratio and len(btext) < 200 else "narrative"
            if caption_re.match(btext):
                kind = "caption"
            items.append((bbox[1], bbox[0], Element(
                kind=kind, text=btext, page=pno + 1, bbox=bbox, extractor="pymupdf")))
            if len(doc_text_sample) < 40 and kind == "narrative":
                doc_text_sample.append(btext)

        # --- figures / images ---
        fig_items = []
        if cfg.extract_figures:
            seen = set()
            for img in page.get_images(full=True):
                xref = img[0]
                if xref in seen:
                    continue
                seen.add(xref)
                try:
                    rects = page.get_image_rects(xref)
                except Exception:
                    rects = []
                for r in rects:
                    bbox = [r.x0, r.y0, r.x1, r.y1]
                    if (r.x1 - r.x0) * (r.y1 - r.y0) < 2500:  # skip tiny logos/rules
                        continue
                    meta = {"xref": xref}
                    if assets:
                        try:
                            crop = page.get_pixmap(clip=r, dpi=cfg.image_dpi)
                            fp = os.path.join(assets, f"p{pno + 1}_fig{xref}.png")
                            crop.save(fp)
                            meta["image_path"] = fp
                        except Exception:
                            pass
                    fig_items.append((bbox[1], bbox[0], Element(
                        kind="figure", text="", page=pno + 1, bbox=bbox, extractor="pymupdf", metadata=meta)))

        # --- merge into reading order (top-to-bottom, left-to-right) ---
        page_items = sorted(items + fig_items, key=lambda it: (round(it[0] / 3), it[1]))
        emitted_by_order = {}
        for _, _, el in page_items:
            order += 1
            el.order = order
            result.elements.append(el)
            emitted_by_order[order] = el

        # link captions to nearest figure above/below
        figs = [el for _, _, el in fig_items]
        for el in result.elements:
            if el.kind == "caption" and el.page == pno + 1 and el.bbox:
                best, bestd = None, 1e9
                for f in figs:
                    if not f.bbox:
                        continue
                    d = abs(el.bbox[1] - f.bbox[3])
                    if d < bestd:
                        best, bestd = f, d
                if best is not None and bestd < 80:
                    el.parent_id = best.order

        # --- OCR / page image ---
        if do_ocr or cfg.emit_page_images:
            pix = page.get_pixmap(dpi=cfg.image_dpi)
            png = pix.tobytes("png")
            img_path = None
            if assets:
                img_path = os.path.join(assets, f"page_{pno + 1}.png")
                with open(img_path, "wb") as fh:
                    fh.write(png)
            if do_ocr:
                text, conf = _ocr(png, cfg.ocr_langs)
                if text.strip():
                    order += 1
                    result.elements.append(Element(
                        kind="narrative", text=text, page=pno + 1, order=order,
                        confidence=conf, extractor="tesseract",
                        metadata={"ocr": True, "langs": cfg.ocr_langs, **({"image_path": img_path} if img_path else {})}))
                    if len(doc_text_sample) < 40:
                        doc_text_sample.append(text)
            if cfg.emit_page_images:
                order += 1
                result.elements.append(Element(
                    kind="page_image", text="", page=pno + 1, order=order, extractor="pymupdf",
                    metadata={"dpi": cfg.image_dpi, **({"image_path": img_path} if img_path else {})}))

    doc.close()
    if cfg.detect_language:
        lang = detect_language(" ".join(doc_text_sample))
        result.language = lang
        for el in result.elements:
            if el.text and el.language is None:
                el.language = lang


def _pdfplumber_extract(path: str, cfg: ExtractionConfig, result: ExtractionResult) -> None:
    import pdfplumber
    order = 0
    sample: list[str] = []
    with pdfplumber.open(path) as pdf:
        result.page_count = len(pdf.pages)
        n = len(pdf.pages) if cfg.max_pages is None else min(len(pdf.pages), cfg.max_pages)
        for pno in range(n):
            page = pdf.pages[pno]
            if cfg.extract_tables:
                for tbl in page.extract_tables() or []:
                    rows = ["| " + " | ".join((c or "").replace("\n", " ") for c in row) + " |" for row in tbl]
                    order += 1
                    result.elements.append(Element(kind="table", text="\n".join(rows), page=pno + 1,
                                                    order=order, extractor="pdfplumber"))
            text = page.extract_text() or ""
            if text.strip():
                order += 1
                result.elements.append(Element(kind="narrative", text=text, page=pno + 1,
                                                order=order, extractor="pdfplumber"))
                if len(sample) < 40:
                    sample.append(text)
    if cfg.detect_language:
        result.language = detect_language(" ".join(sample))


class PdfExtractor:
    doc_types = ("pdf",)

    def extract(self, path: str, config: ExtractionConfig) -> ExtractionResult:
        result = ExtractionResult(source=path, doc_type="pdf", mime="application/pdf")
        backend = config.pdf_backend
        if backend == "auto":
            backend = "pymupdf" if _HAS_FITZ else "pdfplumber"
        if backend == "pymupdf" and _HAS_FITZ:
            _pymupdf_extract(path, config, result)
            result.stats["backend"] = "pymupdf"
        else:
            _pdfplumber_extract(path, config, result)
            result.stats["backend"] = "pdfplumber"
        return result


register(PdfExtractor())
