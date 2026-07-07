"""Pluggable language detection. Falls back gracefully when langdetect is absent."""
from __future__ import annotations


def detect_language(text: str) -> str | None:
    text = (text or "").strip()
    if len(text) < 20:
        return None
    try:
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return None
