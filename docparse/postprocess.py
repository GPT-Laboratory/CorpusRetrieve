"""Format-agnostic finalization applied to every ExtractionResult.

Runs after any extractor and provides, uniformly:
  * section ancestry  — every element gets section_path from the heading stack
  * char offsets      — char_start/char_end into result.document_text(), so a
                        downstream chunker can do hierarchical or *late* chunking
                        without re-parsing
  * per-element language (optional) and document-language fill-in
  * list-item promotion for narrative blocks that look like bullets
"""
from __future__ import annotations

import re

from docparse.lang import detect_language
from docparse.models import ExtractionConfig, ExtractionResult

_TEXT_KINDS = {"title", "heading", "narrative", "list_item", "caption", "reference"}


def _promote_list_items(result: ExtractionResult, cfg: ExtractionConfig) -> None:
    list_re = re.compile(cfg.list_regex)
    for el in result.elements:
        if el.kind == "narrative" and el.text and len(el.text) < 400 and list_re.match(el.text):
            el.kind = "list_item"


_MOJIBAKE = re.compile(r"[\uac00-\ud7af\ufffd\u25a1]")   # CP949-misdecoded Δ etc.


def _is_pseudo_heading(text: str) -> bool:
    """Table header rows, axis labels, and mojibake fragments that a font-size
    heuristic promoted to 'heading' — they must not become section titles."""
    t = text.strip()
    if not t:
        return True
    if _MOJIBAKE.search(t):
        return True
    toks = t.split()
    numish = sum(1 for w in toks if re.search(r"\d|°|%|±", w))
    # column-header shape: several short unit/number tokens, no sentence structure
    if len(toks) >= 3 and numish >= max(2, len(toks) // 2):
        return True
    if len(toks) >= 4 and all(len(w) <= 12 for w in toks) \
            and numish >= 2 and not re.search(r"\b(?:and|of|in|for|the)\b", t, re.I):
        return True
    return False


def _assign_section_paths(result: ExtractionResult) -> None:
    stack: list[tuple[int, str]] = []  # (level, heading text)
    for el in result.elements:
        if el.kind in ("title", "heading"):
            if el.kind == "heading" and _is_pseudo_heading(el.text):
                el.metadata["pseudo_heading"] = True
                el.section_path = [t for _, t in stack]
                continue                       # never enters the section stack
            level = el.level if el.level is not None else (0 if el.kind == "title" else 1)
            while stack and stack[-1][0] >= level:
                stack.pop()
            el.section_path = [t for _, t in stack]
            stack.append((level, el.text.strip()))
        else:
            el.section_path = [t for _, t in stack]


def _assign_char_offsets(result: ExtractionResult) -> None:
    pos = 0
    first = True
    for el in result.elements:
        if not el.text:
            continue
        if not first:
            pos += 2  # the "\n\n" joiner in document_text()
        el.char_start = pos
        pos += len(el.text)
        el.char_end = pos
        first = False


def _assign_languages(result: ExtractionResult, cfg: ExtractionConfig) -> None:
    if not cfg.detect_language:
        return
    if result.language is None:
        sample = " ".join(e.text for e in result.elements
                          if e.kind in _TEXT_KINDS and e.text)[:4000]
        result.language = detect_language(sample)
    for el in result.elements:
        if not el.text or el.kind not in _TEXT_KINDS:
            continue
        if cfg.per_element_language and len(el.text) >= 40:
            el.language = detect_language(el.text) or result.language
        elif el.language is None:
            el.language = result.language


def finalize(result: ExtractionResult, cfg: ExtractionConfig) -> ExtractionResult:
    _promote_list_items(result, cfg)
    _assign_section_paths(result)
    _assign_char_offsets(result)
    _assign_languages(result, cfg)
    result.stats.setdefault("element_count", len(result.elements))
    return result
