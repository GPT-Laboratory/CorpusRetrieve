"""Language-aware text helpers shared by extraction, enrichment, and chunking.

These rules are shared by every stage and make no ASCII-only assumptions:

* ``split_sentences`` recognises any Unicode uppercase letter as a sentence
  start (Finnish ``Ä``/``Ö``, Swedish ``Å``) and does not split after common
  Finnish, Swedish, and English abbreviations.
* ``is_suspended_hyphen`` protects coordinated compounds such as
  ``maa- ja metsätalous`` or ``pre- and post-processing`` from line-break repair.
* ``estimate_tokens`` no longer assumes English word length.
* ``garbled_ratio`` measures how much of a text layer is undecodable, so a
  page with a broken font encoding can be routed to OCR.
"""
from __future__ import annotations

import re
import unicodedata

# Words that follow a suspended (coordination) hyphen: "maa- ja metsätalous".
SUSPENDED_HYPHEN_CONNECTIVES = frozenset({
    # Finnish
    "ja", "tai", "eli", "sekä", "kuin", "että", "vai", "eikä",
    # Swedish
    "och", "eller", "samt",
    # English
    "and", "or", "to", "nor", "versus", "vs",
    # German (common in European reports)
    "und", "oder", "bis",
})

_ABBREVIATIONS = frozenset({
    # Finnish
    "esim", "mm", "ns", "yms", "jne", "ym", "n", "ks", "vrt", "eli", "ts", "tms",
    "s", "tk", "v", "vs", "milj", "mrd", "noin", "kpl", "prof", "toim", "julk",
    # Swedish
    "t.ex", "bl.a", "osv", "dvs", "ca", "resp",
    # English
    "e.g", "i.e", "et al", "al", "fig", "figs", "eq", "eqs", "no", "vol", "pp",
    "approx", "cf", "dr", "mr", "mrs", "ms", "st", "vs", "etc", "tab",
})

# A sentence ends at . ! ? followed by whitespace and an uppercase letter, a
# digit, an opening bracket, or a quotation mark.
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[0-9]|[^\W\d_])")


def _starts_upper_or_open(text: str) -> bool:
    if not text:
        return False
    ch = text[0]
    return ch.isupper() or ch.isdigit() or ch in "\"'“(["


def _ends_with_abbreviation(text: str) -> bool:
    tail = text.rstrip()
    if not tail.endswith("."):
        return False
    words = tail[:-1].split()
    if not words:
        return False
    last = words[-1].lower().lstrip("(\"'")
    two = " ".join(words[-2:]).lower() if len(words) >= 2 else last
    if last in _ABBREVIATIONS or two in _ABBREVIATIONS:
        return True
    # single initials such as "J." in author lists
    return len(last) == 1 and last.isalpha()


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences without breaking on known abbreviations."""
    text = (text or "").strip()
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    for m in _BOUNDARY.finditer(text):
        candidate = text[start:m.start()]
        nxt = text[m.end():m.end() + 1]
        if not _starts_upper_or_open(nxt) or _ends_with_abbreviation(candidate):
            continue
        pieces.append(candidate.strip())
        start = m.end()
    tail = text[start:].strip()
    if tail:
        pieces.append(tail)
    return [p for p in pieces if p]


def is_suspended_hyphen(right_word: str) -> bool:
    """True when ``left- right`` is a coordinated compound, not a line break."""
    return right_word.lower() in SUSPENDED_HYPHEN_CONNECTIVES


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate that does not undercount long-word languages.

    English prose averages about 1.33 subword tokens per word; agglutinative
    languages such as Finnish have long words that tokenise into many pieces.
    Taking the maximum of a word-based and a character-based estimate keeps the
    budget honest for both.
    """
    words = len(re.findall(r"\S+", text or ""))
    chars = len(text or "")
    return max(1, round(max(words * 4 / 3, chars / 4)))


_GARBLE_CATEGORIES = {"Co", "Cn", "Cs"}


def garbled_ratio(text: str) -> float:
    """Share of non-space characters that are replacement, private-use, or control."""
    chars = [c for c in (text or "") if not c.isspace()]
    if not chars:
        return 0.0
    bad = 0
    for c in chars:
        if c == "\ufffd":
            bad += 1
            continue
        cat = unicodedata.category(c)
        if cat in _GARBLE_CATEGORIES or (cat == "Cc" and c not in "\t\n\r"):
            bad += 1
    return bad / len(chars)


def normalize_token_text(text: str) -> str:
    """NFKC + casefold: the canonical form every lexical index should share."""
    return unicodedata.normalize("NFKC", text or "").casefold()
