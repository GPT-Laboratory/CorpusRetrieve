"""OCR preflight: is Tesseract installed, and is every requested language present?

``ocr_status`` answers once per process, and the extractors count every page
that needed OCR but could not get it, so a missing binary or language pack
makes the build fail its quality gate instead of silently skipping pages.

Set ``TESSERACT_CMD`` to the full path of the Tesseract executable when it is
not on ``PATH`` (the usual case on Windows, for example
``C:\\Program Files\\Tesseract-OCR\\tesseract.exe``).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from functools import lru_cache


def tesseract_binary() -> str | None:
    """``TESSERACT_CMD`` when it names an existing file, otherwise ``PATH``."""
    configured = os.getenv("TESSERACT_CMD", "").strip().strip('"')
    if configured:
        return configured if os.path.isfile(configured) else None
    return shutil.which("tesseract")


def configure_pytesseract() -> str | None:
    """Point pytesseract at the same binary the preflight found."""
    binary = tesseract_binary()
    if binary:
        try:
            import pytesseract

            pytesseract.pytesseract.tesseract_cmd = binary
        except ImportError:
            return None
    return binary


@lru_cache(maxsize=16)
def ocr_status(langs: str) -> dict:
    """Return availability, version, installed and missing languages."""
    binary = tesseract_binary()
    status = {"binary": binary, "available": False, "version": None,
              "requested": [lang for lang in (langs or "").split("+") if lang],
              "installed": [], "missing": [],
              "configured_path": os.getenv("TESSERACT_CMD", "").strip() or None}
    if not binary:
        status["missing"] = list(status["requested"])
        return status
    try:
        ver = subprocess.run([binary, "--version"], capture_output=True, text=True,
                             timeout=10)
        first = (ver.stdout or ver.stderr).splitlines()
        status["version"] = first[0].strip() if first else None
        out = subprocess.run([binary, "--list-langs"], capture_output=True, text=True,
                             timeout=10)
        lines = (out.stdout or "").splitlines()
        installed = [line.strip() for line in lines[1:] if line.strip()]
    except Exception:  # noqa: BLE001 - reported as "no languages installed"
        installed = []
    status["installed"] = installed
    status["missing"] = [lang for lang in status["requested"] if lang not in installed]
    status["available"] = bool(installed) and not status["missing"]
    return status


def usable_langs(langs: str) -> str | None:
    """The requested languages that are actually installed, or None."""
    st = ocr_status(langs)
    have = [lang for lang in st["requested"] if lang in st["installed"]]
    return "+".join(have) if have else None
