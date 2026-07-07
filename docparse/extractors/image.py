"""Image extractor: preprocessed, orientation-corrected OCR with per-line bboxes."""
from __future__ import annotations

from docparse.router import register
from docparse.models import Element, ExtractionConfig, ExtractionResult


class ImageExtractor:
    doc_types = ("image",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        from PIL import Image
        result = ExtractionResult(source=path, doc_type="image", mime="image/*")
        img = Image.open(path)
        result.stats.update({"size": list(img.size), "mode": img.mode, "format": img.format})
        result.elements.append(Element(
            kind="page_image", text="", order=1, extractor="pillow",
            visual_ref={"image_path": path, "width": img.size[0], "height": img.size[1]}))
        if cfg.ocr in ("auto", "on"):
            import io
            from docparse.extractors.pdf import _ocr_lines
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            lines = _ocr_lines(buf.getvalue(), cfg, result, scale=1.0)  # bbox in pixels
            order = 1
            if cfg.ocr_line_elements:
                for text, conf, bbox in lines:
                    order += 1
                    result.elements.append(Element(
                        kind="narrative", text=text, order=order, bbox=bbox,
                        confidence=conf, extractor="tesseract",
                        metadata={"ocr": True, "langs": cfg.ocr_langs, "bbox_unit": "px"}))
            elif lines:
                confs = [c for _, c, _ in lines if c is not None]
                order += 1
                result.elements.append(Element(
                    kind="narrative", text=" ".join(t for t, _, _ in lines), order=order,
                    confidence=round(sum(confs) / len(confs), 3) if confs else None,
                    extractor="tesseract", metadata={"ocr": True}))
        return result


register(ImageExtractor())
