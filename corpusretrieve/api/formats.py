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


def available_embedders() -> list[str]:
    embedders = ["hash"]
    if find_spec("FlagEmbedding") is not None:
        embedders.append("bge-m3")
    return embedders
