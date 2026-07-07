"""File-type detection: extension first, `file`/libmagic as backup."""
from __future__ import annotations

import mimetypes
import subprocess
from pathlib import Path

_EXT_TO_TYPE = {
    ".pdf": "pdf",
    ".docx": "docx",
    ".xlsx": "xlsx", ".xlsm": "xlsx",
    ".csv": "csv", ".tsv": "csv",
    ".html": "html", ".htm": "html",
    ".txt": "text", ".md": "text", ".log": "text",
    ".json": "json", ".jsonl": "json",
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".webp": "image",
    ".gif": "image", ".tif": "image", ".tiff": "image", ".bmp": "image",
}


def detect_type(path: str | Path) -> tuple[str, str]:
    """Return (doc_type, mime)."""
    p = Path(path)
    ext = p.suffix.lower()
    doc_type = _EXT_TO_TYPE.get(ext, "")
    mime = mimetypes.guess_type(str(p))[0] or ""
    if not doc_type:
        try:
            out = subprocess.run(["file", "--mime-type", "-b", str(p)],
                                 capture_output=True, text=True, timeout=10).stdout.strip()
            mime = out or mime
            if "pdf" in out:
                doc_type = "pdf"
            elif "word" in out or "officedocument.wordprocessing" in out:
                doc_type = "docx"
            elif "sheet" in out or "excel" in out:
                doc_type = "xlsx"
            elif out.startswith("image/"):
                doc_type = "image"
            elif out.startswith("text/"):
                doc_type = "text"
        except Exception:
            pass
    return doc_type or "unknown", mime
