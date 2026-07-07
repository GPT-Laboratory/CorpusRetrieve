"""Pluggable language detection.

Prefers lingua-py (markedly more accurate, especially on short text) when it is
installed; falls back to langdetect; degrades to None gracefully.
"""
from __future__ import annotations

from functools import lru_cache

_MIN_CHARS = 20


@lru_cache(maxsize=1)
def _lingua():
    try:
        from lingua import LanguageDetectorBuilder
        return LanguageDetectorBuilder.from_all_spoken_languages().build()
    except Exception:
        return None


def detect_language(text: str) -> str | None:
    text = (text or "").strip()
    if len(text) < _MIN_CHARS:
        return None
    det = _lingua()
    if det is not None:
        try:
            lang = det.detect_language_of(text)
            if lang is not None:
                return lang.iso_code_639_1.name.lower()
        except Exception:
            pass
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return None
