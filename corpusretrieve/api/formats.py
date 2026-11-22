from __future__ import annotations

from importlib.util import find_spec

SUPPORTED_FORMATS = {
    ".pdf": "PDF documents",
    ".docx": "Microsoft Word documents",
    ".xlsx": "Microsoft Excel workbooks",
    ".xlsm": "Macro-enabled Excel workbooks",
    ".csv": "Comma-separated tables",
    ".tsv": "Tab-separated tables",
    ".json": "JSON documents",
    ".jsonl": "JSON Lines documents",
    ".html": "HTML documents",
    ".htm": "HTML documents",
    ".txt": "Plain-text documents",
    ".md": "Markdown documents",
    ".log": "Log files",
    ".png": "PNG images",
    ".jpg": "JPEG images",
    ".jpeg": "JPEG images",
    ".webp": "WebP images",
    ".gif": "GIF images",
    ".tif": "TIFF images",
    ".tiff": "TIFF images",
    ".bmp": "Bitmap images",
}

SUPPORTED_EXTENSIONS = frozenset(SUPPORTED_FORMATS)
SUPPORTED_FORMATS_INLINE = ", ".join(sorted(SUPPORTED_EXTENSIONS))


EMBEDDER_CATALOG = (
    {"spec": "hash-v2", "kind": "lexical", "dimension": 512, "requires": None,
     "label": "Hashing (Unicode words) - no download, matches shared words only"},
    {"spec": "hash", "kind": "lexical", "dimension": 512, "requires": None,
     "label": "Hashing (ASCII words) - kept for comparing with older bundles"},
    {"spec": "st:intfloat/multilingual-e5-small", "kind": "semantic", "dimension": 384,
     "requires": "sentence_transformers",
     "label": "multilingual-e5-small - semantic, multilingual, fastest on CPU"},
    {"spec": "st:intfloat/multilingual-e5-base", "kind": "semantic", "dimension": 768,
     "requires": "sentence_transformers",
     "label": "multilingual-e5-base - semantic, multilingual, balanced"},
    {"spec": "bge-m3", "kind": "semantic", "dimension": 1024, "requires": "FlagEmbedding",
     "label": "BGE-M3 - semantic, multilingual, long inputs, best with a GPU"},
)


def _installed(module: str | None) -> bool:
    return module is None or find_spec(module) is not None


def embedder_catalog() -> list[dict]:
    """Every built-in embedder choice, marked with whether it can run here."""
    return [{**entry, "available": _installed(entry["requires"])} for entry in EMBEDDER_CATALOG]


def available_embedders() -> list[str]:
    return [entry["spec"] for entry in embedder_catalog() if entry["available"]] + ["auto"]


def embedder_available(spec: str) -> bool:
    """Built-in choices, plus any ``st:<model id>`` when sentence-transformers is installed."""
    if spec in available_embedders():
        return True
    return spec.startswith("st:") and len(spec) > 3 and _installed("sentence_transformers")


def ocr_preflight(langs: str) -> dict:
    """What OCR will be able to do for scanned pages, checked before a build."""
    from docparse.ocr import ocr_status

    status = ocr_status(langs)
    if not status["binary"]:
        message = ("Tesseract was not found, so scanned pages cannot be read. Install it, "
                   "or set TESSERACT_CMD to the full path of tesseract.exe.")
    elif status["missing"]:
        message = ("Tesseract is missing language data for "
                   + "+".join(status["missing"]) + "; scanned pages in those languages "
                   "will be read with the installed languages only.")
    else:
        message = None
    return {"requested": status["requested"], "installed": status["installed"],
            "missing": status["missing"], "binary": status["binary"],
            "version": status["version"], "ready": message is None, "message": message}
