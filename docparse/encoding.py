"""Robust text decoding for arbitrary input files.

Strategy, in order:
  1. Byte-order marks (BOM) → exact encoding (UTF-8/16/32, with endianness).
  2. charset-normalizer, if installed → statistical detection.
  3. Fallback chain: utf-8 (strict) → cp1252 → latin-1 (latin-1 never fails).

Returns the decoded text AND the encoding used, so callers can record provenance.
Binary container formats (PDF/DOCX/XLSX) do NOT use this — their libraries decode internally.
"""
from __future__ import annotations

from pathlib import Path

# Longer BOMs first: utf-32-le (FF FE 00 00) must be tested before utf-16-le (FF FE).
_BOMS: list[tuple[bytes, str]] = [
    (b"\xef\xbb\xbf", "utf-8-sig"),
    (b"\xff\xfe\x00\x00", "utf-32"),
    (b"\x00\x00\xfe\xff", "utf-32"),
    (b"\xff\xfe", "utf-16"),
    (b"\xfe\xff", "utf-16"),
]

_FALLBACKS = ("utf-8", "cp1252", "latin-1")


def detect_encoding(raw: bytes) -> str:
    for bom, enc in _BOMS:
        if raw.startswith(bom):
            return enc
    try:
        from charset_normalizer import from_bytes
        best = from_bytes(raw).best()
        if best and best.encoding:
            return best.encoding
    except Exception:
        pass
    for enc in _FALLBACKS:
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "latin-1"  # decodes any byte sequence


def read_text(path: str | Path) -> tuple[str, str]:
    """Read a text file with best-effort encoding detection → (text, encoding_used)."""
    raw = Path(path).read_bytes()
    enc = detect_encoding(raw)
    return raw.decode(enc, errors="replace").lstrip("\ufeff"), enc
