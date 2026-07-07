"""Image extractor: OCR text out of a raster image file."""
from __future__ import annotations

from docparse.base import register
from docparse.lang import detect_language
from docparse.models import Element, ExtractionConfig, ExtractionResult


class ImageExtractor:
    doc_types = ("image",)

    def extract(self, path: str, cfg: ExtractionConfig) -> ExtractionResult:
        from PIL import Image
        result = ExtractionResult(source=path, doc_type="image", mime="image/*")
        img = Image.open(path)
        result.stats.update({"size": list(img.size), "mode": img.mode, "format": img.format})
        result.elements.append(Element(kind="page_image", text="", order=1, extractor="pillow",
                                       metadata={"width": img.size[0], "height": img.size[1]}))
        if cfg.ocr in ("auto", "on"):
            try:
                import pytesseract
                try:
                    text = pytesseract.image_to_string(img, lang=cfg.ocr_langs)
                except Exception:
                    text = pytesseract.image_to_string(img, lang="eng")
                if text.strip():
                    result.elements.append(Element(kind="narrative", text=text.strip(), order=2,
                                                   extractor="tesseract", metadata={"ocr": True}))
                    if cfg.detect_language:
                        result.language = detect_language(text)
            except Exception as exc:
                result.warnings.append(f"OCR failed: {exc}")
        return result


register(ImageExtractor())
