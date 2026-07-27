"""PDF extraction. Primary backend PyMuPDF, structure-preserving pdfplumber fallback.

Per page, in column-aware reading order (recursive XY-cut):
  headings (with levels from font-size clusters), narrative, tables (markdown text
  + structured header/rows view + optional crop), figures (bbox + optional crop)
  with captions linked to the nearest figure OR table above/below, and — for
  scanned pages — preprocessed, orientation-corrected OCR with per-line bboxes.

Running headers/footers/page numbers are detected by cross-page repetition and
suppressed (kind page_header/page_footer, dropped unless configured otherwise).
"""
from __future__ import annotations

import io
import os
import re
from collections import Counter
from statistics import median

from docparse.router import register
from docparse.models import Element, ExtractionConfig, ExtractionResult
from docparse.serialize import serialize_table

try:
    import fitz  # PyMuPDF
    _HAS_FITZ = True
except Exception:
    _HAS_FITZ = False


# --------------------------------------------------------------------------- helpers

def _center_in(bbox, box) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return box[0] <= cx <= box[2] and box[1] <= cy <= box[3]


def _norm_boiler(text: str) -> str:
    """Normalize a block for repetition matching: page numbers vary, the rest doesn't."""
    t = re.sub(r"\d+", "#", text.strip().lower())
    return re.sub(r"\s+", " ", t)


def _vertical_gap(caption_bbox, target_bbox) -> float:
    """Distance between a caption and a figure/table, valid above OR below."""
    below = caption_bbox[1] - target_bbox[3]   # caption under the target
    above = target_bbox[1] - caption_bbox[3]   # caption over the target
    gaps = [g for g in (below, above) if g >= -5]  # small overlap tolerated
    return min(gaps) if gaps else 1e9


def _xy_cut(items: list[tuple[list[float], object]], min_gap: float = 14.0) -> list[object]:
    """Recursive XY-cut reading order. items = [(bbox, payload)].

    Splits on the widest horizontal whitespace gap first (columns, left→right),
    then vertical gaps (top→bottom); falls back to (y, x) sort at the leaves.
    """
    if len(items) <= 1:
        return [p for _, p in items]

    def best_gap(axis: int) -> tuple[float, float] | None:
        lo, hi = axis, axis + 2
        ivs = sorted((b[lo], b[hi]) for b, _ in items)
        merged = [list(ivs[0])]
        for s, e in ivs[1:]:
            if s <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        best = None
        for (a0, a1), (b0, _) in zip(merged, merged[1:]):
            gap = b0 - a1
            if gap >= min_gap and (best is None or gap > best[0]):
                best = (gap, (a1 + b0) / 2)
        return best

    for axis in (0, 1):  # x first: columns beat bands
        g = best_gap(axis)
        if g:
            cut = g[1]
            lo_items = [(b, p) for b, p in items if (b[axis] + b[axis + 2]) / 2 < cut]
            hi_items = [(b, p) for b, p in items if (b[axis] + b[axis + 2]) / 2 >= cut]
            if lo_items and hi_items:
                return _xy_cut(lo_items, min_gap) + _xy_cut(hi_items, min_gap)
    return [p for _, p in sorted(items, key=lambda it: (round(it[0][1] / 3), it[0][0]))]


def _heading_levels(all_sizes: list[float], med: float, ratio: float) -> dict[float, int]:
    """Cluster distinct 'large' font sizes into heading levels 1..3."""
    big = sorted({round(s, 1) for s in all_sizes if med and s >= med * ratio}, reverse=True)
    return {s: min(i + 1, 3) for i, s in enumerate(big)}




_SOFT_CELL = str.maketrans({"\u00ad": "", "\u202f": " ", "\u00a0": " ",
                            "\u2010": "-", "\u2011": "-"})
_IDX_CELL = re.compile(r"^\d{1,3}[.)]?$")


def _clean_text_grid(rows, row_bboxes):
    """Filter a text-strategy grid: drop margin-junk rows, empty columns, prose tails.

    Returns (rows, bbox) or (None, None) if what remains does not look like a table.
    """
    if not rows:
        return None, None
    width = max(len(r) for r in rows)
    keep_idx, kept = [], []
    for i, r in enumerate(rows):
        cells = [("" if c is None else str(c).translate(_SOFT_CELL).strip()) for c in r] \
            + [""] * (width - len(r))
        nonempty = [c for c in cells if c]
        if len(nonempty) < 3:                     # rotated-sidebar fragments, page furniture
            continue
        if max(len(c) for c in nonempty) > 90:    # body prose swallowed below the table
            break
        # dense rows of alphabetic word-fragments = shredded two-column body text
        if len(kept) > 3 and len(nonempty) >= max(6, int(0.8 * width)) \
                and sum(1 for c in nonempty if any(ch.isdigit() for ch in c)) < 0.3 * len(nonempty):
            break
        keep_idx.append(i)
        kept.append(cells)
    if len(kept) < 3:
        return None, None
    # index-anchored trim: if rows carry a "1. / 2. / 3." counter column, the table
    # is exactly the contiguous indexed span (plus the header rows above it)
    idx_rows = [j for j, r in enumerate(kept) if _IDX_CELL.match(r[0])]
    if len(idx_rows) >= 3:
        first, last = idx_rows[0], idx_rows[-1]
        keep_idx = keep_idx[:last + 1]
        kept = kept[:last + 1]
    # heal word fragments split across column boundaries by the text strategy:
    # "H|adGEM3", "U|nited States", "Eu||ropean Consortium", "EC-E|arth3-Veg".
    # A fragment continues into the next non-empty cell when that cell starts
    # lowercase and the left piece bears a truncation signature.
    def _is_fragment(left: str) -> bool:
        if not left or left[-1] in ".)]:%":            # index cells, complete tokens
            return False
        if len(left) <= 3 and left.isalpha():
            return True
        return len(left) <= 6 and any(ch.isalpha() for ch in left) \
            and (left[-1].isupper() or "-" in left)    # "EC-E" style truncation

    _intra = re.compile(r"\b([A-Za-z0-9]*(?:-[A-Za-z0-9]*)*[A-Z-]) ([a-z][A-Za-z0-9-]*)")

    def _intra_fix(m: re.Match) -> str:
        l, rr = m.group(1), m.group(2)
        if "-" in l or "-" in rr or any(ch.isdigit() for ch in l + rr):
            return l + rr                      # "EC-E arth3-Veg" -> "EC-Earth3-Veg"
        return m.group(0)                      # "United States" stays two words

    for r in kept:
        for i, c in enumerate(r):
            if c and " " in c:
                r[i] = _intra.sub(_intra_fix, c)
        for i in range(len(r) - 2, -1, -1):
            left = r[i]
            if not _is_fragment(left):
                continue
            j = next((k for k in range(i + 1, len(r)) if r[k]), None)
            if j is not None and r[j][0].isalpha() and r[j][0].islower():
                r[j] = left + r[j]
                r[i] = ""

    # drop columns that are empty in (almost) every kept row
    fill = [sum(1 for r in kept if r[c]) for c in range(width)]
    cols = [c for c in range(width) if fill[c] >= max(2, int(0.2 * len(kept)))]
    if len(cols) < 3:
        return None, None
    kept = [[r[c] for c in cols] for r in kept]
    first_data = next((i for i, r in enumerate(kept)
                       if re.match(r"^\d+\.?$", r[0]) or sum(c.replace(".", "").isdigit() for c in r if c) >= 3), None)
    if first_data and 0 < first_data <= 3:      # collapse wrapped header lines into one
        header = [" ".join(x for x in (kept[j][c] for j in range(first_data)) if x).strip()
                  for c in range(len(kept[0]))]
        kept = [header] + kept[first_data:]
    boxes = [row_bboxes[i] for i in keep_idx if i < len(row_bboxes) and row_bboxes[i]]
    bbox = None
    if boxes:
        bbox = [min(b[0] for b in boxes), min(b[1] for b in boxes),
                max(b[2] for b in boxes), max(b[3] for b in boxes)]
    return kept, bbox


def _caption_anchored_tables(page, cfg, existing_boxes):
    """For each 'TABLE n' caption without a ruled table below it, detect an unruled
    table via PyMuPDF's text strategy clipped under the caption, then clean the grid."""
    out = []
    cap_re = re.compile(cfg.table_caption_regex, re.IGNORECASE)
    W, H = page.rect.width, page.rect.height
    for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
        if not cap_re.match(text.strip()):
            continue
        if any(tb[1] >= y1 - 5 and tb[1] <= y1 + 300 for tb in existing_boxes):
            continue                               # a ruled table already covers this caption
        # in two-column layouts the table lives in the caption's column
        cx0, cx1 = 24, W - 30
        if x1 < W * 0.55:
            cx1 = W * 0.52                        # left-column table
        elif x0 > W * 0.45:
            cx0 = W * 0.48                        # right-column table
        try:
            found = page.find_tables(clip=fitz.Rect(cx0, y1 - 2, cx1, H), strategy="text").tables
        except Exception:
            continue
        if not found:
            continue
        t = max(found, key=lambda t: (t.bbox[3] - t.bbox[1]) * (t.bbox[2] - t.bbox[0]))
        try:
            row_bboxes = [list(r.bbox) for r in t.rows]
        except Exception:
            row_bboxes = []
        rows, bbox = _clean_text_grid(t.extract(), row_bboxes)
        if rows:
            out.append((rows, bbox or list(t.bbox)))
    return out

# --------------------------------------------------------------------------- OCR

def _preprocess(img, cfg: ExtractionConfig):
    from PIL import ImageOps
    if not cfg.ocr_preprocess:
        return img
    g = ImageOps.autocontrast(img.convert("L"))
    return g


def _auto_orient(img, result: ExtractionResult, cfg: ExtractionConfig):
    if not cfg.ocr_auto_orient:
        return img
    try:
        import pytesseract
        osd = pytesseract.image_to_osd(img)
        m = re.search(r"Rotate:\s*(\d+)", osd)
        rot = int(m.group(1)) if m else 0
        s = re.search(r"Script:\s*(\w+)", osd)
        if s:
            result.stats.setdefault("ocr_scripts", []).append(s.group(1))
        if rot:
            return img.rotate(-rot, expand=True)
    except Exception:
        pass
    return img


def _ocr_lines(png_bytes: bytes, cfg: ExtractionConfig, result: ExtractionResult,
               scale: float) -> list[tuple[str, float | None, list[float]]]:
    """OCR a page raster -> [(line_text, confidence, bbox_in_source_coords)]."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(png_bytes))
        img = _auto_orient(img, result, cfg)
        img = _preprocess(img, cfg)
        try:
            data = pytesseract.image_to_data(img, lang=cfg.ocr_langs,
                                             output_type=pytesseract.Output.DICT)
        except Exception:
            data = pytesseract.image_to_data(img, lang="eng",
                                             output_type=pytesseract.Output.DICT)
        lines: dict[tuple, dict] = {}
        n = len(data.get("text", []))
        for i in range(n):
            w = (data["text"][i] or "").strip()
            if not w:
                continue
            key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
            box = [data["left"][i], data["top"][i],
                   data["left"][i] + data["width"][i], data["top"][i] + data["height"][i]]
            ln = lines.setdefault(key, {"words": [], "confs": [], "bbox": box})
            ln["words"].append(w)
            ln["bbox"] = [min(ln["bbox"][0], box[0]), min(ln["bbox"][1], box[1]),
                          max(ln["bbox"][2], box[2]), max(ln["bbox"][3], box[3])]
            try:
                c = float(data["conf"][i])
                if c >= 0:
                    ln["confs"].append(c)
            except (ValueError, TypeError):
                pass
        out = []
        for key in sorted(lines):
            ln = lines[key]
            conf = round(sum(ln["confs"]) / len(ln["confs"]) / 100.0, 3) if ln["confs"] else None
            bbox = [round(v * scale, 2) for v in ln["bbox"]]
            out.append((" ".join(ln["words"]), conf, bbox))
        return out
    except Exception as exc:
        result.warnings.append(f"OCR failed: {exc}")
        return []


# --------------------------------------------------------------------------- PyMuPDF

def _collect_boilerplate(doc, cfg: ExtractionConfig) -> set[str]:
    """Cheap first pass: normalized top/bottom-zone blocks repeated across pages."""
    if not cfg.remove_boilerplate or doc.page_count < cfg.boilerplate_min_pages:
        return set()
    counter: Counter[str] = Counter()
    n = doc.page_count if cfg.max_pages is None else min(doc.page_count, cfg.max_pages)
    for pno in range(n):
        page = doc[pno]
        h = page.rect.height
        top, bot = h * cfg.boilerplate_zone, h * (1 - cfg.boilerplate_zone)
        seen_on_page = set()
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            text = text.strip()
            if not text or len(text) > 120:
                continue
            if y1 <= top or y0 >= bot:
                seen_on_page.add(_norm_boiler(text))
        counter.update(seen_on_page)
    threshold = max(cfg.boilerplate_min_pages, int(n * cfg.boilerplate_min_ratio))
    return {t for t, c in counter.items() if c >= threshold}


def _pymupdf_extract(path: str, cfg: ExtractionConfig, result: ExtractionResult) -> None:
    doc = fitz.open(path)
    result.page_count = doc.page_count
    caption_re = re.compile(cfg.caption_regex, re.IGNORECASE)
    order = 0
    assets = cfg.assets_dir
    if assets:
        os.makedirs(assets, exist_ok=True)

    boiler = _collect_boilerplate(doc, cfg)
    result.stats["boilerplate_patterns"] = len(boiler)
    scale = 72.0 / cfg.image_dpi  # raster px -> PDF points

    # per-page dimensions (PDF points, top-left origin) so a UI overlay can
    # scale element bboxes onto a rendered page image without guessing.
    result.stats["page_dims"] = {
        str(p + 1): {"width": doc[p].rect.width, "height": doc[p].rect.height,
                     "unit": "pt", "origin": "top-left"}
        for p in range(doc.page_count)}

    n_pages = doc.page_count if cfg.max_pages is None else min(doc.page_count, cfg.max_pages)
    for pno in range(n_pages):
        page = doc[pno]
        pd = page.get_text("dict")
        raw_text = page.get_text("text")
        page_h = page.rect.height

        is_scanned = len(raw_text.strip()) < cfg.ocr_min_chars
        do_ocr = cfg.ocr == "on" or (cfg.ocr == "auto" and is_scanned)

        # --- figure-dominant page detection (schematics / maps / drawings) ---
        # Such pages are mostly vector ink with sparse label text; PyMuPDF's
        # find_tables() misreads their grid/border/wires as page-spanning
        # "tables", which then suppress the real (whole-page) figure. Detect
        # this up front and treat the page as figure-first instead.
        page_is_figure = False
        page_fig_bbox = None
        if cfg.figure_dominant_detection and cfg.extract_figures:
            try:
                n_paths = len(page.get_drawings())
            except Exception:
                n_paths = 0
            n_chars = len(raw_text.strip())
            if (n_paths >= cfg.figure_dominant_min_paths
                    and n_chars <= cfg.figure_dominant_max_chars):
                try:
                    clusters = page.cluster_drawings()
                except Exception:
                    clusters = []
                page_area = page.rect.width * page.rect.height or 1.0
                for cr in clusters:
                    cov = ((cr.x1 - cr.x0) * (cr.y1 - cr.y0)) / page_area
                    if cov >= cfg.figure_dominant_min_coverage:
                        page_is_figure = True
                        page_fig_bbox = [cr.x0, cr.y0, cr.x1, cr.y1]
                        break

        # --- tables first (their text blocks get suppressed) ---
        # ...skipped on figure-dominant pages so drawing lines aren't mistaken
        # for ruled tables (the small real title-block still surfaces as text).
        table_boxes: list[list[float]] = []
        table_els: list[Element] = []
        if cfg.extract_tables and not page_is_figure:
            try:
                for ti, t in enumerate(page.find_tables().tables):
                    rows = t.extract()
                    bbox = list(t.bbox)
                    table_boxes.append(bbox)
                    text, extras, structured, trunc = serialize_table(
                        rows, cfg.table_serializations, cfg.table_max_rows)
                    visual = {"page": pno + 1, "bbox": bbox}
                    if assets:
                        try:
                            crop = page.get_pixmap(clip=fitz.Rect(bbox), dpi=cfg.image_dpi)
                            fp = os.path.join(assets, f"p{pno + 1}_table{ti}.png")
                            crop.save(fp)
                            visual["image_path"] = fp
                        except Exception:
                            pass
                    meta = {"rows": getattr(t, "row_count", None),
                            "cols": getattr(t, "col_count", None), "truncated": trunc}
                    if extras:
                        meta["serializations"] = extras
                    table_els.append(Element(
                        kind="table", text=text, page=pno + 1, bbox=bbox,
                        extractor="pymupdf", structured=structured,
                        visual_ref=visual, metadata=meta))
            except Exception as exc:
                result.warnings.append(f"table extraction failed on page {pno + 1}: {exc}")
            if cfg.table_fallback == "text":
                for rows, bbox in _caption_anchored_tables(page, cfg, table_boxes):
                    table_boxes.append(bbox)
                    text, extras, structured, trunc = serialize_table(
                        rows, cfg.table_serializations, cfg.table_max_rows)
                    visual = {"page": pno + 1, "bbox": bbox}
                    if assets:
                        try:
                            crop = page.get_pixmap(clip=fitz.Rect(bbox), dpi=cfg.image_dpi)
                            fp = os.path.join(assets, f"p{pno + 1}_table{len(table_els)}.png")
                            crop.save(fp)
                            visual["image_path"] = fp
                        except Exception:
                            pass
                    meta = {"rows": len(rows), "cols": len(rows[0]),
                            "truncated": trunc, "strategy": "caption-anchored-text"}
                    if extras:
                        meta["serializations"] = extras
                    table_els.append(Element(
                        kind="table", text=text, page=pno + 1, bbox=bbox,
                        extractor="pymupdf-text", structured=structured,
                        visual_ref=visual, metadata=meta))

        # --- text blocks (skip table interiors and boilerplate) ---
        sizes = [s["size"] for b in pd["blocks"] if b.get("type") == 0
                 for ln in b.get("lines", []) for s in ln.get("spans", [])]
        med = median(sizes) if sizes else 0.0
        levels = _heading_levels(sizes, med, cfg.heading_size_ratio)

        text_els: list[Element] = []
        for b in pd["blocks"]:
            if b.get("type") != 0:
                continue
            bbox = list(b["bbox"])
            if any(_center_in(bbox, tb) for tb in table_boxes):
                continue
            spans = [s for ln in b.get("lines", []) for s in ln.get("spans", [])]
            line_texts = [lt for ln in b.get("lines", [])
                          if (lt := "".join(s["text"] for s in ln.get("spans", [])).strip())]
            btext = " ".join(line_texts).strip()
            if not btext:
                continue

            if boiler and _norm_boiler(btext) in boiler:
                zone = "page_header" if bbox[3] <= page_h * cfg.boilerplate_zone else "page_footer"
                result.stats["boilerplate_removed"] = result.stats.get("boilerplate_removed", 0) + 1
                if cfg.parameters.get("keep_boilerplate"):
                    text_els.append(Element(kind=zone, text=btext, page=pno + 1,
                                            bbox=bbox, extractor="pymupdf"))
                continue

            max_size = round(max((s["size"] for s in spans), default=0.0), 1)
            bold = any(s.get("flags", 0) & 16 for s in spans)  # bit 4 = bold
            kind, level = "narrative", None
            if med and len(btext) < 200:
                if max_size in levels:
                    kind, level = "heading", levels[max_size]
                elif bold and max_size >= med and len(btext) < 90 and not btext.endswith("."):
                    kind, level = "heading", (levels.get(max_size) or 3)
            if caption_re.match(btext):
                kind, level = "caption", None
            text_els.append(Element(kind=kind, text=btext, page=pno + 1, bbox=bbox,
                                    level=level, extractor="pymupdf"))

        # --- figures / images ---
        fig_els: list[Element] = []
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
                    if (r.x1 - r.x0) * (r.y1 - r.y0) < 2500:  # skip tiny logos/rules
                        continue
                    bbox = [r.x0, r.y0, r.x1, r.y1]
                    visual = {"page": pno + 1, "bbox": bbox, "xref": xref}
                    if assets:
                        try:
                            crop = page.get_pixmap(clip=r, dpi=cfg.image_dpi)
                            fp = os.path.join(assets, f"p{pno + 1}_fig{xref}.png")
                            crop.save(fp)
                            visual["image_path"] = fp
                        except Exception:
                            pass
                    fig_els.append(Element(kind="figure", text="", page=pno + 1, bbox=bbox,
                                           extractor="pymupdf", visual_ref=visual))

        # --- vector figures: charts/maps/diagrams drawn as paths, not raster images ---
        if page_is_figure and page_fig_bbox is not None:
            # figure-dominant page: emit the whole drawing as one figure and
            # skip the normal cluster filtering (which the phantom tables and
            # sheer size would otherwise trip). This is the element the VLM
            # describes — the entire schematic / diagram.
            visual = {"page": pno + 1, "bbox": page_fig_bbox, "source": "vector",
                      "full_page": True}
            if assets:
                try:
                    crop = page.get_pixmap(clip=fitz.Rect(page_fig_bbox), dpi=cfg.image_dpi)
                    fp = os.path.join(assets, f"p{pno + 1}_page_figure.png")
                    crop.save(fp)
                    visual["image_path"] = fp
                except Exception:
                    pass
            fig_els.append(Element(kind="figure", text="", page=pno + 1,
                                   bbox=page_fig_bbox, extractor="pymupdf",
                                   visual_ref=visual,
                                   metadata={"vector": True, "full_page": True}))
        elif cfg.extract_figures and cfg.vector_figures:
            try:
                clusters = page.cluster_drawings()
            except Exception:
                clusters = []
            for ci, cr in enumerate(clusters):
                w, h = cr.x1 - cr.x0, cr.y1 - cr.y0
                if w * h < cfg.vector_figure_min_area or min(w, h) < cfg.vector_figure_min_dim:
                    continue                                  # rules, underlines, ornaments
                bbox = [cr.x0, cr.y0, cr.x1, cr.y1]
                if any(_center_in(bbox, tb) or _center_in(tb, bbox) for tb in table_boxes):
                    continue                                  # ruled tables are drawings too
                # a decorated text block (abstract box, sidebar, callout) is not a figure:
                # real charts/maps/diagrams are mostly ink, not mostly words
                try:
                    words = page.get_text("words", clip=cr)
                    coverage = sum((x1 - x0) * (y1 - y0) for x0, y0, x1, y1, *_ in words) / (w * h)
                    if coverage > cfg.figure_max_text_coverage:
                        continue
                except Exception:
                    pass
                if any(f.bbox and _center_in(bbox, f.bbox) for f in fig_els):
                    continue                                  # already covered by a raster figure
                visual = {"page": pno + 1, "bbox": bbox, "source": "vector"}
                if assets:
                    try:
                        crop = page.get_pixmap(clip=cr, dpi=cfg.image_dpi)
                        fp = os.path.join(assets, f"p{pno + 1}_vfig{ci}.png")
                        crop.save(fp)
                        visual["image_path"] = fp
                    except Exception:
                        pass
                fig_els.append(Element(kind="figure", text="", page=pno + 1, bbox=bbox,
                                       extractor="pymupdf", visual_ref=visual,
                                       metadata={"vector": True}))

        # --- column-aware reading order over everything on the page ---
        page_items = [(el.bbox, el) for el in text_els + table_els + fig_els if el.bbox]
        ordered = (_xy_cut(page_items) if cfg.column_aware
                   else [el for _, el in sorted(page_items,
                                                key=lambda it: (round(it[0][1] / 3), it[0][0]))])
        for el in ordered:
            order += 1
            el.order = order
            result.elements.append(el)

        # --- link captions to nearest figure OR table (above or below) ---
        if cfg.link_captions:
            targets = fig_els + table_els
            for el in ordered:
                if el.kind != "caption" or not el.bbox:
                    continue
                best, bestd = None, cfg.caption_max_gap
                for tgt in targets:
                    d = _vertical_gap(el.bbox, tgt.bbox)
                    if d < bestd:
                        best, bestd = tgt, d
                if best is not None:
                    el.parent_id = best.id

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
                lines = _ocr_lines(png, cfg, result, scale)
                if cfg.ocr_line_elements:
                    for text, conf, bbox in lines:
                        order += 1
                        result.elements.append(Element(
                            kind="narrative", text=text, page=pno + 1, order=order,
                            bbox=bbox, confidence=conf, extractor="tesseract",
                            metadata={"ocr": True, "langs": cfg.ocr_langs}))
                elif lines:
                    confs = [c for _, c, _ in lines if c is not None]
                    order += 1
                    result.elements.append(Element(
                        kind="narrative", text=" ".join(t for t, _, _ in lines),
                        page=pno + 1, order=order,
                        confidence=round(sum(confs) / len(confs), 3) if confs else None,
                        extractor="tesseract", metadata={"ocr": True, "langs": cfg.ocr_langs}))
            if cfg.emit_page_images:
                order += 1
                result.elements.append(Element(
                    kind="page_image", text="", page=pno + 1, order=order, extractor="pymupdf",
                    visual_ref={"page": pno + 1, "dpi": cfg.image_dpi,
                                **({"image_path": img_path} if img_path else {})}))

    doc.close()


# --------------------------------------------------------------------------- pdfplumber

def _pdfplumber_extract(path: str, cfg: ExtractionConfig, result: ExtractionResult) -> None:
    """Fallback that keeps the contract: headings (font-size), bboxes, structured tables."""
    import pdfplumber
    order = 0
    with pdfplumber.open(path) as pdf:
        result.page_count = len(pdf.pages)
        caption_re = re.compile(cfg.caption_regex, re.IGNORECASE)
        n = len(pdf.pages) if cfg.max_pages is None else min(len(pdf.pages), cfg.max_pages)
        for pno in range(n):
            page = pdf.pages[pno]
            table_boxes: list[list[float]] = []
            table_els: list[Element] = []
            if cfg.extract_tables:
                try:
                    for ti, t in enumerate(page.find_tables()):
                        rows = t.extract()
                        bbox = list(t.bbox)
                        table_boxes.append(bbox)
                        text, extras, structured, trunc = serialize_table(
                            rows, cfg.table_serializations, cfg.table_max_rows)
                        meta = {"rows": len(rows), "truncated": trunc}
                        if extras:
                            meta["serializations"] = extras
                        table_els.append(Element(
                            kind="table", text=text, page=pno + 1, bbox=bbox,
                            extractor="pdfplumber", structured=structured,
                            visual_ref={"page": pno + 1, "bbox": bbox}, metadata=meta))
                except Exception as exc:
                    result.warnings.append(f"table extraction failed on page {pno + 1}: {exc}")

            # lines with font sizes -> paragraph blocks
            try:
                lines = page.extract_text_lines() or []
            except Exception:
                lines = []
            sizes = [c.get("size", 0.0) for ln in lines for c in ln.get("chars", [])]
            med = median(sizes) if sizes else 0.0
            levels = _heading_levels(sizes, med, cfg.heading_size_ratio)

            blocks: list[dict] = []
            for ln in sorted(lines, key=lambda l: (l["top"], l["x0"])):
                bbox = [ln["x0"], ln["top"], ln["x1"], ln["bottom"]]
                if any(_center_in(bbox, tb) for tb in table_boxes):
                    continue
                lsize = round(max((c.get("size", 0.0) for c in ln.get("chars", [])), default=0.0), 1)
                text = (ln.get("text") or "").strip()
                if not text:
                    continue
                prev = blocks[-1] if blocks else None
                gap = bbox[1] - prev["bbox"][3] if prev else 1e9
                same_kind = prev is not None and prev["size"] == lsize
                if prev and same_kind and gap < max(4.0, lsize * 0.9):
                    prev["text"] += " " + text
                    prev["bbox"] = [min(prev["bbox"][0], bbox[0]), prev["bbox"][1],
                                    max(prev["bbox"][2], bbox[2]), bbox[3]]
                else:
                    blocks.append({"text": text, "bbox": bbox, "size": lsize})

            text_els: list[Element] = []
            for b in blocks:
                kind, level = "narrative", None
                if b["size"] in levels and len(b["text"]) < 200:
                    kind, level = "heading", levels[b["size"]]
                if caption_re.match(b["text"]):
                    kind, level = "caption", None
                text_els.append(Element(kind=kind, text=b["text"], page=pno + 1,
                                        bbox=b["bbox"], level=level, extractor="pdfplumber"))

            page_items = [(el.bbox, el) for el in text_els + table_els if el.bbox]
            ordered = (_xy_cut(page_items) if cfg.column_aware
                       else [el for _, el in sorted(page_items,
                                                    key=lambda it: (round(it[0][1] / 3), it[0][0]))])
            for el in ordered:
                order += 1
                el.order = order
                result.elements.append(el)

            if cfg.link_captions:
                for el in ordered:
                    if el.kind != "caption" or not el.bbox:
                        continue
                    best, bestd = None, cfg.caption_max_gap
                    for tgt in table_els:
                        d = _vertical_gap(el.bbox, tgt.bbox)
                        if d < bestd:
                            best, bestd = tgt, d
                    if best is not None:
                        el.parent_id = best.id


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
