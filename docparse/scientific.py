"""Scientific-document enrichment: the layers a research paper needs beyond parsing.

Two passes, both driven by ``ExtractionConfig.scientific``:

``clean(result, cfg)``   — runs BEFORE finalize (it edits text / removes elements):
  * strips publisher license/download boilerplate (Wiley sidebar & friends)
  * removes soft hyphens and repairs line-break hyphenation ("se- verity" -> "severity",
    "bias- corrected" -> "bias-corrected", decided against the document's own vocabulary)
  * promotes numbered / symbol-dense lines to kind="formula" with a structured view

``annotate(result, cfg)`` — runs AFTER finalize (it only adds metadata):
  * scientific role per element (front_matter/abstract/introduction/methods/results/
    discussion/conclusion/references/acknowledgements/appendix) from heading ancestry
  * paper metadata: DOI, title, journal, year, received/revised/accepted, keywords, funding
  * reference entries (kind="reference") + inline-citation linking -> stats["citation_graph"]
  * quantities (%, °C, ranges, return periods), uncertainty flags, and heuristic
    claim extraction from abstract/results/discussion -> stats["claims"]

Everything here is deterministic and heuristic — clearly useful for retrieval, but claims
and citation links are best-effort, not ground truth. Figure *understanding* (panel-by-panel
map/plot description) needs a vision model; see docparse.enrich for that hook.
"""
from __future__ import annotations

import re
from collections import Counter

from docparse.models import Element, ExtractionConfig, ExtractionResult
from docparse.text import is_suspended_hyphen, split_sentences

_TEXT_KINDS = {"title", "heading", "narrative", "list_item", "caption", "reference"}

# --------------------------------------------------------------------------- clean

from functools import lru_cache


@lru_cache(maxsize=8)
def _license_re(patterns: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)

_SOFT_CHARS = {"\u00ad": "", "\u2010": "-", "\u2011": "-", "\u202f": " ", "\u00a0": " "}
_LETTERS = r"[^\W\d_]"
_HYPH_BREAK = re.compile(rf"({_LETTERS}+)-\s+({_LETTERS}+)")

_MATH_SYMBOLS = set("≤≥±×÷∑∏∫√ΔδσμλνΦφθτ≈≠≡→←∂∞∝∈")
_EQ_NUM = re.compile(r"^\(\d{1,2}\)\s*|\s*\(\d{1,2}\)\s*$")
_PUB_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")


def _looks_like_reference_entry(text: str) -> bool:
    """True when a numbered short line looks like a bibliography/reference-
    list entry: either it has no real
    expression content once the "(N)" marker is stripped (a bare number is
    not a formula), or it contains a publication year and no math symbol at
    all, which is the fingerprint of a "(N) Author. Title. Year." citation
    line in essentially any language -- publication years are universal even
    where heading vocabulary is not."""
    stripped = _EQ_NUM.sub("", text).strip()
    if not stripped:
        return True
    has_symbol = any(c in _MATH_SYMBOLS for c in text)
    has_year = bool(_PUB_YEAR.search(text))
    return has_year and not has_symbol


def _strip_soft(text: str) -> str:
    for bad, good in _SOFT_CHARS.items():
        text = text.replace(bad, good)
    return text


def _build_vocab(result: ExtractionResult) -> set[str]:
    """Words that appear OUTSIDE hyphen-break spans — the broken fragments themselves
    ("tempera", "ture") must not vouch for keeping a bogus hyphen."""
    vocab: set[str] = set()
    for el in result.elements:
        if el.kind in _TEXT_KINDS and el.text:
            clean_text = _HYPH_BREAK.sub(" ", el.text)
            vocab.update(w.lower() for w in re.findall(rf"{_LETTERS}{{3,}}", clean_text))
    return vocab


def _dehyphenate(text: str, vocab: set[str]) -> tuple[str, int]:
    n = 0

    def fix(m: re.Match) -> str:
        nonlocal n
        left, right = m.group(1), m.group(2)
        if is_suspended_hyphen(right):
            # coordinated compound: "maa- ja metsätalous", "pre- and post-"
            return m.group(0)
        n += 1
        # the document's own vocabulary decides: merged word seen elsewhere -> join;
        # both halves are real standalone words -> true compound, keep the hyphen
        if (left + right).lower() in vocab:
            return f"{left}{right}"
        if left.lower() in vocab and right.lower() in vocab and len(left) >= 3 and len(right) >= 3:
            return f"{left}-{right}"
        return f"{left}{right}"

    return _HYPH_BREAK.sub(fix, text), n


def _looks_like_equation(text: str, cfg: ExtractionConfig) -> bool:
    if len(text) > 220:
        return False
    words = len(text.split())
    symbols = sum(text.count(s) for s in _MATH_SYMBOLS)
    numbered = bool(_EQ_NUM.search(text))
    if numbered and _looks_like_reference_entry(text):
        return False
    return (numbered and ("=" in text or symbols >= 1)) or (symbols >= 2 and words <= 30) \
        or ("=" in text and words <= 30)


_PAGE_MARKER = re.compile(r"^\s*\d{1,3}\s+of\s+\d{1,3}\s*$", re.I)
_DOI_LINE = re.compile(r"^\s*(?:https?://)?(?:dx\.)?doi\.org/\S+\s*$", re.I)


def _scrub_inline_noise(text: str, lic_re: re.Pattern) -> tuple[str, int]:
    """Remove license/footer sentences *inside* long elements instead of dropping
    the whole element (PDF extraction often fuses footers into body paragraphs)."""
    removed = 0
    lines: list[str] = []
    for line in text.split("\n"):
        if _PAGE_MARKER.match(line):
            removed += 1
            continue
        lines.append(line)
    parts = re.split(r"(?<=[.!?])\s+", "\n".join(lines))
    kept = []
    for p in parts:
        if p and lic_re.search(p) and len(p) < 400:
            removed += 1
            continue
        kept.append(p)
    return " ".join(k for k in kept if k).strip(), removed


_NUMISH = re.compile(r"\d|°C|%")
_NUMBERED_HEADING = re.compile(r"^\d+(?:\.\d+)*\s*[|.\u2012-\u2015-]?\s+\S")
_LOWER_WORD = re.compile(r"^[a-z][a-z-]+$")
_UNIT_WORDS = {"day", "days", "year", "years", "month", "months", "hour", "hours",
               "week", "weeks", "min", "max", "mean", "median", "std"}
_FRONT_MATTER_HEADING = re.compile(
    r"^\s*(?:(?:research|original|review)\s+article(?:\s+open\s+access)?"
    r"|open\s+access|(?:short|technical)\s+communication"
    r"|(?:international\s+)?journal\s+of\s+[a-z][a-z ]{0,40}"
    r"|author\s+contributions?|conflicts?\s+of\s+interest(?:\s+disclosures?)?"
    r"|data\s+availability(?:\s+statement)?|ethics\s+statement"
    r"|orcid|correspondence|keywords)\s*$", re.I)


def _fake_heading(text: str) -> bool:
    """Table header rows and broken-glyph fragments must not become sections.

    Real scientific headings may legitimately contain several numbers —
    "3.1 | Changes in Cold Days at 1.0°C, 1.5°C and 2.0°C Warming Levels" —
    so numeric density alone is NOT sufficient evidence. A table header has
    numbers AND no connective prose ("Model Country Npar 1.0°C 1.5°C 2.0°C"
    contains zero lowercase words). Numbered section headings and
    equation-like lines are always exempt."""
    if _EQ_NUM.search(text) or any(c in _MATH_SYMBOLS for c in text) or "=" in text:
        return False
    if _NUMBERED_HEADING.match(text.strip()):
        return False                                   # "2.3 | Cold Extreme Indices"
    toks = text.split()
    if len(toks) < 3:
        return False
    if sum(1 for t in toks if _LOWER_WORD.match(t) and t not in _UNIT_WORDS) >= 2:
        return False                                   # connective prose => real title
    numish = sum(1 for t in toks if _NUMISH.search(t))
    nonlatin = bool(re.search(r"[\u1100-\uD7FF\uF900-\uFFFF]", text))
    return numish >= 3 or numish / len(toks) > 0.5 or (nonlatin and numish >= 1)


_BACK_MATTER_HEADING = re.compile(
    r"^\s*(?:acknowledg(?:e)?ments?|appendix(?:\s+[A-Z0-9])?|funding|references?"
    r"|bibliography|supporting information|data availability(?:\s+statement)?"
    r"|author contributions?|conflicts? of interest)\b", re.I)
_CAPTION_LINE = re.compile(
    r"^\s*(FIGURE|TABLE|FIG\.?|KUVA|KUVIO|TAULUKKO|FIGUR|TABELL)\s+(\d+[A-Za-z]?)"
    r"\s*[|.:\u2012-\u2015-]?\s+", re.I)
_TABLE_WORDS = ("TAB", "TAULUKKO")


def _promote_caption_narratives(result: ExtractionResult) -> None:
    claimed = {e.parent_id for e in result.elements if e.kind == "caption" and e.parent_id}
    targets = [e for e in result.elements if e.kind in ("figure", "table")
               and e.id not in claimed]
    promoted = 0
    for el in result.elements:
        if el.kind not in ("narrative", "other") or not el.text:
            continue
        m = _CAPTION_LINE.match(el.text)
        if not m or len(el.text) > 900:                   # captions are shortish
            continue
        kind_word, num = m.group(1).upper(), m.group(2)
        is_table = kind_word.startswith(_TABLE_WORDS)
        label = ("Table " if is_table else "Figure ") + num
        want = "table" if is_table else "figure"
        near = [t for t in targets if t.kind == want and t.page == el.page]
        el.kind = "caption"
        el.metadata["label"] = label
        if near:
            tgt = min(near, key=lambda t: abs((t.order or 0) - (el.order or 0))
                      if hasattr(t, "order") else 0)
            el.parent_id = tgt.id
            tgt.metadata.setdefault("label", label)
            targets.remove(tgt)
        promoted += 1
    if promoted:
        result.stats["captions_promoted"] = promoted


def clean(result: ExtractionResult, cfg: ExtractionConfig) -> ExtractionResult:
    # 0) headings that are really table headers / visual fragments
    demoted = 0
    for el in result.elements:
        if el.kind != "heading" or not el.text:
            continue
        if _FRONT_MATTER_HEADING.match(el.text):
            el.kind = "other"                          # banner: metadata, not a section
            el.level = None
            el.metadata["front_matter_banner"] = True
            demoted += 1
        elif _fake_heading(el.text):
            el.kind = "other"
            el.level = None
            demoted += 1
    if demoted:
        result.stats["fake_headings_demoted"] = demoted

    # 0b) back-matter headings are top-level: Acknowledgements etc. must not
    # nest under "6 | Conclusions" just because they carry no section number
    for el in result.elements:
        if el.kind == "heading" and el.text and _BACK_MATTER_HEADING.match(el.text):
            el.level = 1

    # 0c) publisher caption blocks extracted as narratives ("FIGURE 1 | ..."):
    # promote to caption elements, derive the label, link to the nearest
    # unclaimed figure/table on the same page
    _promote_caption_narratives(result)

    # 1) publisher license / download boilerplate — anywhere on the page, any orientation
    kept: list[Element] = []
    removed = 0
    lic_re = _license_re(tuple(cfg.noise_patterns))
    for el in result.elements:
        if el.kind in _TEXT_KINDS and el.text and lic_re.search(el.text) \
                and (len(el.text) < 600 or len(lic_re.findall(el.text)) >= 2):
            removed += 1
            continue
        # long element with fused footer/license text: scrub inline, keep the science
        if el.kind in _TEXT_KINDS and el.text and (lic_re.search(el.text)
                                                   or "\n" in el.text):
            el.text, n = _scrub_inline_noise(el.text, lic_re)
            removed += 1 if n else 0
            if not el.text:
                continue
        kept.append(el)
    if removed:
        result.elements = kept
        result.stats["license_noise_removed"] = removed

    # 2) soft hyphens first (they hide inside words), then vocabulary, then line-break repair
    for el in result.elements:
        if el.text:
            el.text = _strip_soft(el.text)
    vocab = _build_vocab(result)
    fixes = 0
    nospace = re.compile(rf"({_LETTERS}{{2,}})-({_LETTERS}{{2,}})")

    def fix_nospace(m: re.Match) -> str:
        l, r = m.group(1), m.group(2)
        if not r[0].islower():
            return m.group(0)
        if (l + r).lower() in vocab and not (l.lower() in vocab and r.lower() in vocab):
            return l + r                       # "vir-tually" -> "virtually"
        return m.group(0)

    for el in result.elements:
        if el.kind not in _TEXT_KINDS or not el.text:
            continue
        if "- " in el.text:
            el.text, n = _dehyphenate(el.text, vocab)
            fixes += n
        el.text = nospace.sub(fix_nospace, el.text)
    if fixes:
        result.stats["hyphenation_fixes"] = fixes

    # 3) equations -> kind="formula" with a structured view and equation number
    n_eq = 0
    for i, el in enumerate(result.elements):
        if not el.text:
            continue
        is_eq = el.kind in ("narrative", "list_item") and _looks_like_equation(el.text, cfg)
        if is_eq and (len(el.text.strip()) < 6 or
                      (el.bbox and el.bbox[2] - el.bbox[0] > 360
                       and "=" not in el.text and el.text.count("±") >= 2)):
            is_eq = False
        if el.kind == "heading" and _EQ_NUM.search(el.text) and len(el.text) < 120 \
                and (sum(el.text.count(c) for c in _MATH_SYMBOLS) >= 1
                     or ("=" in el.text and len(el.text.split()) <= 12)):
            is_eq = True
        if is_eq:
            num = None
            m = re.search(r"\((\d{1,2})\)", el.text)
            if m:
                num = int(m.group(1))
            expression = _EQ_NUM.sub("", el.text).strip()
            prev = next((e.id for e in reversed(result.elements[:i])
                         if e.kind == "narrative" and len(e.text) > 60), None)
            nxt = next((e.id for e in result.elements[i + 1:]
                        if e.kind == "narrative" and len(e.text) > 60), None)
            el.kind = "formula"
            el.level = None
            el.structured = {"number": num, "expression": expression}
            el.metadata.update({"context_before": prev, "context_after": nxt})
            n_eq += 1
    if n_eq:
        # absorb stray fragments (summation limits, closing parens) that sit INSIDE or
        # right beside the formula's box on the page — order-independent, bbox-driven
        def _near(a, b, pad=26.0):
            return not (a[2] + pad < b[0] or b[2] + pad < a[0]
                        or a[3] + pad < b[1] or b[3] + pad < a[1])

        formulas = [e for e in result.elements if e.kind == "formula" and e.bbox]
        absorbed: set[str] = set()
        for f in formulas:
            frags = [e for e in result.elements
                     if e.id != f.id and e.id not in absorbed and e.page == f.page
                     and e.kind in ("narrative", "list_item") and e.bbox and e.text
                     and len(e.text) <= 40 and len(e.text.split()) <= 6
                     and not e.text.rstrip().endswith(".") and _near(f.bbox, e.bbox)]
            if not frags:
                continue
            frags.sort(key=lambda e: (round(e.bbox[1]), e.bbox[0]))
            f.text = " ".join([f.text] + [e.text for e in frags]).strip()
            f.structured["expression"] = _EQ_NUM.sub("", f.text).strip()
            f.bbox = [min(f.bbox[0], *(e.bbox[0] for e in frags)),
                      min(f.bbox[1], *(e.bbox[1] for e in frags)),
                      max(f.bbox[2], *(e.bbox[2] for e in frags)),
                      max(f.bbox[3], *(e.bbox[3] for e in frags))]
            absorbed.update(e.id for e in frags)
        if absorbed:
            result.elements = [e for e in result.elements if e.id not in absorbed]

        merged: list[Element] = []
        for el in result.elements:
            prev = merged[-1] if merged else None
            if (prev is not None and prev.kind == "formula" and el.page == prev.page
                    and el.kind in ("narrative", "list_item", "formula")
                    and len(el.text) <= 60 and not el.text.rstrip().endswith(".")
                    and (prev.text.rstrip().endswith(("=", "+", "-", "/", "(")) or len(prev.text) < 12
                         or el.kind == "formula")):
                prev.text = (prev.text + " " + el.text).strip()
                prev.structured["expression"] = _EQ_NUM.sub("", prev.text).strip()
                if prev.structured.get("number") is None and (m := re.search(r"\((\d{1,2})\)", el.text)):
                    prev.structured["number"] = int(m.group(1))
                continue
            merged.append(el)
        result.elements = merged
        result.stats["formulas_detected"] = sum(1 for e in result.elements if e.kind == "formula")
    return result


# --------------------------------------------------------------------------- annotate

_ROLE_RULES = [
    # English, Finnish, and Swedish section vocabularies. Word boundaries reduce
    # false matches inside compound words.
    ("abstract", r"abstract|summary of the (paper|study)|\btiivistelmä|\babstrakti"
                 r"|\bsammandrag|\breferat\b"),
    ("references", r"^references?$|^bibliography$|^literature cited$"
                   r"|^viitteet$|^lähteet$|^lähdeluettelo$|^kirjallisuus$"
                   r"|^kirjallisuusluettelo$|^käytetyt lähteet$|^referenser$|^litteratur$"),
    ("acknowledgements", r"acknowledg|\bkiitokset|^tack\b"),
    ("appendix", r"^appendi|supplementary|supporting information|^liite|^liitteet|^bilag"),
    ("funding", r"funding|\brahoitus|\bfinansiering"),
    ("methods", r"method|data and|materials|experimental setup|model description"
                r"|\bmenetelm|\baineisto|\bmaterial och|\bmetod"),
    ("results", r"result|finding|\btulokset|\btulos\b|\bresultat"),
    ("discussion", r"discussion|\bpohdinta|\btarkastelu|\bdiskussion"),
    ("conclusion", r"conclusion|concluding|summary$|\bjohtopäätök|\byhteenveto"
                   r"|\bslutsats|\bsammanfattning"),
    ("introduction", r"introduction|background|\bjohdanto|\btausta\b|\binledning"
                     r"|\bbakgrund"),
]

# All built-in role rules apply independently of document subject.
_PROFILE_ROLE_RULES: dict[str, list[tuple[str, str]]] = {}

def _role_for(last_heading: str, chain: str = "", profile: str = "generic") -> str | None:
    rules = _ROLE_RULES + _PROFILE_ROLE_RULES.get(profile, [])
    last = re.sub(r"^[\d.\s|]+", "", last_heading.lower()).strip()   # "2 | Data and Methods"
    for role, pat in rules:
        if re.search(pat, last):
            return role
    if chain:
        loose = " ".join(re.sub(r"^[\d.\s|]+", "", h.lower()) for h in chain.split(" / "))
        for role, pat in rules:
            if re.search(pat.replace("^", "").replace("$", ""), loose):
                return role
    return None


def _assign_roles(result: ExtractionResult, profile: str = "generic") -> None:
    dist: Counter[str] = Counter()
    current = "front_matter"
    body_started = False
    for el in result.elements:
        if el.kind in ("title", "heading"):
            role = _role_for(el.text, " / ".join(el.section_path + [el.text]), profile)
            if role:
                current = role
                body_started = body_started or role not in ("front_matter", "abstract")
            elif body_started:
                current = current if (el.level or 1) > 1 else "body"
            elif _NUMBERED_HEADING.match(el.text or ""):
                current = "body"                     # numbered section: body began
                body_started = True
            else:
                current = "front_matter"
        elif not body_started and re.match(r"^\s*abstract\b", el.text or "", re.I):
            current = "abstract"
        el.metadata["role"] = current
        dist[current] += 1

    # safety net: "abstract" is a short contiguous block near the front. If it
    # runs away (heading detection failed for this corpus), downgrade the tail
    # to "body" so summary/claim builders can't absorb the whole paper.
    abs_els = [e for e in result.elements if e.metadata.get("role") == "abstract"]
    if abs_els:
        first_page = abs_els[0].page or 1
        cut = 0
        for e in abs_els:
            if (e.page or first_page) > first_page + 1 or cut >= 30:
                e.metadata["role"] = "body"
                dist["abstract"] -= 1
                dist["body"] += 1
            else:
                cut += 1
    result.stats["role_distribution"] = {k: v for k, v in dist.items() if v}


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+\b")
_DATE_RE = re.compile(r"(received|revised|accepted)[:\s]+(\d{1,2}\s+\w+\s+(?:19|20)\d{2})", re.I)


def _extract_paper_metadata(result: ExtractionResult) -> None:
    page1 = [e for e in result.elements if (e.page or 1) <= 2 and e.text]
    all_p1 = " ".join(e.text for e in page1)
    paper: dict = {}

    if result.stats.get("source_doi"):
        paper["doi"] = result.stats["source_doi"]
    elif (m := _DOI_RE.search(all_p1)):
        paper["doi"] = m.group(0).rstrip(".,;")
    title_el = next((e for e in result.elements
                     if e.kind in ("title", "heading") and (e.level or 9) <= 1
                     and (e.page or 1) == 1 and len(e.text) > 15), None)
    if result.stats.get("source_title"):
        paper["title"] = result.stats["source_title"]
    elif title_el:
        paper["title"] = title_el.text
    if title_el:
        # authors: the next short capitalized-name line(s) after the title
        idx = result.elements.index(title_el)
        for e in result.elements[idx + 1: idx + 5]:
            t = e.text.strip()
            if 3 < len(t) < 220 and re.match(r"^[A-Z]", t) and not t.endswith(".") \
                    and re.search(r"[A-Z][a-zà-ÿ]+\s+[A-Z]", t) \
                    and not re.search(r"\b(university|institute|department|abstract)\b", t, re.I):
                paper.setdefault("authors_raw", t)
                break
    for m in _DATE_RE.finditer(all_p1):
        paper[m.group(1).lower()] = m.group(2)
    if (m := re.search(r"(19|20)\d{2}", paper.get("accepted", "") or paper.get("received", ""))):
        paper["year"] = int(m.group(0))
    elif (m := re.search(r"©?\s*((?:19|20)\d{2})", all_p1)):
        paper["year"] = int(m.group(1))
    if (m := re.search(r"\b((?:International |Intl )?Journal of [A-Z][a-z]+(?:\s+[A-Z]?[a-z]+){0,3})", all_p1)):
        j = re.split(r"\s+(?=[A-Z]{3,})", m.group(1))[0]        # cut at RESEARCH/OPEN/etc.
        paper["journal"] = re.sub(r"\s+", " ", j).strip()[:80]

    for i, e in enumerate(result.elements[:60]):
        m = re.search(r"\bkey\s*words?\s*[:.]?\s+(.{4,300}?)(?:\n|$|\s{3,}|(?=[A-Z]{4,}))", e.text or "", re.I)
        if not m and re.fullmatch(r"\s*key\s*words?\s*", e.text or "", re.I) and i + 1 < len(result.elements):
            m, src = None, result.elements[i + 1].text
        elif m:
            src = m.group(1)
        else:
            continue
        parts = [k.strip(" .") for k in re.split(r"[,;|]", src) if 1 < len(k.strip()) < 60]
        if len(parts) >= 2:
            paper["keywords"] = parts[:15]
            break
    fund = next((e.text for e in result.elements
                 if e.metadata.get("role") == "funding" and e.kind not in ("heading", "title")
                 and len(e.text) > 20), None)
    if fund:
        paper["funding"] = fund[:400]
    if paper:
        result.stats["paper"] = paper


_YEAR_RE = re.compile(r"\b(19|20)\d{2}[a-z]?\b")
_PAREN_CITE = re.compile(r"\(([^()]{0,240}?(?:19|20)\d{2}[a-z]?[^()]{0,80})\)")
_NARR_CITE = re.compile(r"\b([A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-']{2,})\s+(?:et\s+al\.?\s+)?\(((?:19|20)\d{2})[a-z]?\)")
_STOP_SURNAMES = {"The", "This", "These", "Table", "Figure", "Section", "Equation",
                  "January", "February", "March", "April", "May", "June", "July",
                  "August", "September", "October", "November", "December",
                  "Northern", "Europe", "Since", "During", "From", "Model", "Data"}


def _link_citations(result: ExtractionResult) -> None:
    # 1) reference entries: everything under the References role
    refs: list[Element] = []
    for el in result.elements:
        if el.metadata.get("role") == "references" and el.kind in ("narrative", "list_item"):
            el.kind = "reference"
            refs.append(el)
    if refs:
        result.stats["reference_entries"] = len(refs)

    def ref_lookup(surname: str, year: str) -> str | None:
        y = year[:4]
        word = re.compile(rf"(?<!{_LETTERS}){re.escape(surname.lower())}(?!{_LETTERS})")
        for r in refs:
            if y in r.text and word.search(r.text.lower()):
                return r.id
        return None

    graph: dict[str, dict] = {}
    for el in result.elements:
        if el.kind not in ("narrative", "list_item", "caption") or not el.text \
                or el.metadata.get("role") == "references":
            continue
        found: list[tuple[str, str]] = []
        for grp in _PAREN_CITE.findall(el.text):
            for part in re.split(r";", grp):
                y = _YEAR_RE.search(part)
                names = [w for w in re.findall(r"\b[A-Z][A-Za-zÀ-ÖØ-öø-ÿ\-']{2,}\b", part)
                         if w not in _STOP_SURNAMES]
                if y and names:
                    found.append((names[0], y.group(0)))
        for name, year in _NARR_CITE.findall(el.text):
            if name not in _STOP_SURNAMES:
                found.append((name, year))
        if not found:
            continue
        cites = []
        for surname, year in dict.fromkeys(found):          # de-dup, keep order
            key = f"{surname.lower()}{year[:4]}"
            ref_id = ref_lookup(surname, year)
            cites.append({"key": key, "ref_id": ref_id})
            g = graph.setdefault(key, {"surname": surname, "year": year[:4],
                                       "ref_id": ref_id, "mentions": []})
            g["mentions"].append(el.id)
            g["ref_id"] = g["ref_id"] or ref_id
        el.metadata["citations"] = cites
    if graph:
        linked = sum(1 for v in graph.values() if v["ref_id"])
        result.stats["citation_graph"] = graph
        result.stats["citations"] = {"unique_keys": len(graph), "linked_to_reference": linked}


_NUM = r"(?<![\w.,])[-\u2212]?\d+(?:[.,]\d+)?"
_RANGE_SEP = r"\s*(?:–|—|‒|-|\u2212|to|till|\.\.\.|…)\s*"
# Generic, domain-neutral units. Letter units must not run into a following
# letter ("5 mm" yes, "5 mmol" no); symbol units such as % never needed a
# word boundary after them, otherwise "35% compared" never matches.
_UNIT = (r"(?:kg/ha|t/ha|mg/kg|g/kg|kg/m²|kg/m2|g/m²|mg/l|µg/l|kWh|MWh|GWh|TWh|"
         r"km²|km2|m²|m2|m³|m3|°C|ppm|ppb|ha|kg|mg|µg|g|t|km|cm|mm|m|ml|l|"
         r"EUR|€|MJ|GJ|kW|MW|K|%|‰)")
_UNIT_END = r"(?![^\W\d_])"
_QTY_GENERIC = [
    re.compile(rf"{_NUM}\s*(?:{_UNIT})?{_RANGE_SEP}{_NUM}\s*{_UNIT}{_UNIT_END}"),
    re.compile(rf"{_NUM}\s*{_UNIT}{_UNIT_END}"),
    re.compile(rf"(?<!\w)\d+(?:[.,]\d+)?[- ](?:fold|times|kertainen|kertaiseksi|kertaisesti)\b",
               re.I),
]
_QTY_BY_PROFILE = {"generic": _QTY_GENERIC}
# older name kept for callers that imported it; it is the generic set
_QTY_RES = _QTY_GENERIC


def _quantity_patterns(profile: str) -> list[re.Pattern]:
    return _QTY_BY_PROFILE.get(profile, _QTY_GENERIC)


def find_quantities(text: str, profile: str = "generic") -> list[str]:
    """Return distinct quantity strings, longest match first, no nested duplicates."""
    spans: list[tuple[int, int]] = []
    out: list[str] = []
    for rx in _quantity_patterns(profile):
        for m in rx.finditer(text or ""):
            a, b = m.span()
            if any(a >= x and b <= y for x, y in spans):
                continue
            spans.append((a, b))
            out.append(m.group(0).strip())
    return list(dict.fromkeys(out))


_UNCERTAINTY = re.compile(r"±|\bconfidence\b|uncertaint|inter-?model|\bspread\b|\bensemble range\b"
                          r"|\bepävarm|\bluottamusväli|\bosäkerhet", re.I)
_CLAIM_VERB = re.compile(
    r"\b(show|shows|shown|found|find|increase[sd]?|decrease[sd]?|decline[sd]?|"
    r"reduce[sd]?|project(?:ed|s)?|indicate[sd]?|suggest(?:ed|s)?|rose|fell|"
    r"become[s]?|weaken(?:ed|s)?|warm(?:ing|ed|s)?|occur(?:red|s)?|account(?:s|ed)?|"
    r"amount(?:s|ed)?|reach(?:es|ed)?|varie[sd]|ranged?|"
    # Finnish reporting and change verbs
    r"kasvoi|kasvaa|kasvavat|kasvoivat|väheni|vähenee|vähenivät|lisääntyi|lisääntyy|"
    r"laski|laskee|nousi|nousee|pieneni|suureni|heikkeni|vahvistui|osoitti|osoittavat|"
    r"osoittivat|havaittiin|todettiin|arvioidaan|arvioitiin|ennustetaan|vaihteli|"
    r"oli|olivat|ovat|muodostaa|muodosti|sisältää|sisälsi|"
    # Swedish
    r"ökade|ökar|minskade|minskar|visade|visar|uppgick)\b", re.I)
_CLAIM_ROLES = {"abstract", "results", "discussion", "conclusion", "methods"}
_CHANGE_NEAR = re.compile(r"increas|decreas|reduc|declin|drop|grow|rise|risen|rose|fell|fall|"
                          r"weaken|strengthen|kasv|väh|lisä|laski|nous|pien|supist|heikk|"
                          r"vahvist|ökade|minskade", re.I)
_DEFINITION = re.compile(r"\bis defined as\b|\bwe define\b|\bdefined by\b|\bthreshold\b"
                         r"|\bcriterio?n\b|\bmääritel|\bkynnysarvo|\bdefinieras\b", re.I)


def _qty_type(q: str, sentence: str, profile: str = "generic") -> str:
    """Assign only subject-neutral physical quantity categories."""
    return _qty_type_generic(q, sentence)


_GENERIC_UNIT_TYPES = [
    (re.compile(r"kg/ha|t/ha", re.I), "mass_per_area"),
    (re.compile(r"mg/kg|g/kg|mg/l|µg/l|ppm|ppb", re.I), "concentration"),
    (re.compile(r"kwh|mwh|gwh|twh|\bmj\b|\bgj\b", re.I), "energy"),
    (re.compile(r"\bkw\b|\bmw\b", re.I), "power"),
    (re.compile(r"km²|km2|m²|m2|\bha\b", re.I), "area"),
    (re.compile(r"m³|m3|\bml\b|\bl\b", re.I), "volume"),
    (re.compile(r"\bkg\b|\bmg\b|µg|\bg\b|\bt\b", re.I), "mass"),
    (re.compile(r"\bkm\b|\bcm\b|\bmm\b|\bm\b", re.I), "length"),
    (re.compile(r"€|eur\b", re.I), "currency"),
]


def _qty_type_generic(q: str, sentence: str) -> str:
    ql = q.lower()
    if re.search(r"fold|times|kertai", ql):
        return "ratio"
    if "%" in q or "‰" in q:
        return "percentage_change" if _CHANGE_NEAR.search(sentence) else "percentage"
    if "°c" in ql or re.search(r"\dK\b|\sK\b", q):
        return "temperature_range" if re.search(r"–|—|\bto\b|-\s*\d", q) else "temperature"
    for rx, name in _GENERIC_UNIT_TYPES:
        if rx.search(q):
            return name
    return "quantity"



def _claim_type(sentence: str, role: str) -> str:
    if _UNCERTAINTY.search(sentence):
        return "uncertainty_claim"
    if role == "methods" or _DEFINITION.search(sentence):
        return "method_claim"
    if role == "conclusion":
        return "conclusion_claim"
    return "result_claim"


def _quantities_and_claims(result: ExtractionResult, profile: str = "generic",
                           max_claims: int = 400) -> None:
    claims: list[dict] = []
    roles_present = {e.metadata.get("role") for e in result.elements}
    claim_roles = set(_CLAIM_ROLES)
    if not roles_present & {"abstract", "results", "discussion", "conclusion"}:
        # reports without IMRaD headings (common in institutional reports):
        # every role except back matter can carry the findings
        claim_roles |= {"body", "introduction", "front_matter", None}
    for el in result.elements:
        if el.kind not in ("narrative", "list_item", "caption") or not el.text:
            continue
        qs = find_quantities(el.text, profile)
        if qs:
            el.metadata["quantities"] = qs[:20]
            el.metadata["quantities_labeled"] = [
                {"value": q, "type": _qty_type(q, el.text, profile)} for q in qs[:20]]
        if _UNCERTAINTY.search(el.text):
            el.metadata["uncertainty"] = True
        if el.kind == "caption":
            continue                       # quantified captions -> visual_evidence, below
        if qs and el.metadata.get("role") in claim_roles and len(claims) < max_claims:
            for sent in split_sentences(el.text):
                if len(sent) < 30 or len(sent) > 420:
                    continue
                if el.metadata.get("role") == "methods" and not _DEFINITION.search(sent):
                    continue
                sq = find_quantities(sent, profile)
                if _CLAIM_VERB.search(sent) and sq:
                    claims.append({"text": sent.strip(), "element_id": el.id,
                                   "page": el.page, "role": el.metadata.get("role"),
                                   "type": _claim_type(sent, el.metadata.get("role", "")),
                                   "quantities": [{"value": q, "type": _qty_type(q, sent, profile)}
                                                  for q in sq]})
                    if len(claims) >= max_claims:
                        break
    if claims:
        result.stats["claims"] = claims
    by_id = {e.id: e for e in result.elements}
    evidence = []
    for e in result.elements:
        if e.kind != "caption" or not e.metadata.get("quantities"):
            continue
        parent = by_id.get(e.parent_id)
        label = e.metadata.get("label", "")
        etype = parent.kind if parent is not None else \
            ("table" if label.lower().startswith("table") else
             "figure" if label else "visual")
        evidence.append({"evidence_type": etype, "label": label or None,
                         "text": e.text, "element_id": e.id, "page": e.page,
                         "ref_id": e.parent_id,
                         "quantities": e.metadata.get("quantities_labeled",
                                                      e.metadata.get("quantities", []))})
    if evidence:
        result.stats["visual_evidence"] = evidence


_LABEL_RE = re.compile(
    r"^\s*(fig(?:ure)?|figur|table|taulukko|tabell|kuvio|kuva|tab(?:elle)?|abb(?:ildung)?|"
    r"plate|scheme|chart|exhibit)"
    r"\.?\s*(\d+[A-Za-z]?)", re.IGNORECASE)
_LABEL_NORM = {"fig": "Figure", "figure": "Figure", "kuva": "Figure", "abb": "Figure",
               "kuvio": "Figure", "figur": "Figure", "taulukko": "Table", "tabell": "Table",
               "abbildung": "Figure", "table": "Table", "tab": "Table", "tabelle": "Table",
               "plate": "Plate", "scheme": "Scheme", "chart": "Chart", "exhibit": "Exhibit"}


def _label_visuals(result: ExtractionResult) -> None:
    by_id = {e.id: e for e in result.elements}
    for cap in result.elements:
        if cap.kind != "caption" or not cap.text:
            continue
        m = _LABEL_RE.match(cap.text)
        if not m:
            continue
        label = f"{_LABEL_NORM.get(m.group(1).lower(), m.group(1).title())} {m.group(2)}"
        cap.metadata["label"] = label
        parent = by_id.get(cap.parent_id)
        if parent is not None:
            parent.metadata["label"] = label
            if parent.kind == "table" and isinstance(parent.structured, dict):
                parent.structured.setdefault("table_number", label)


def annotate(result: ExtractionResult, cfg: ExtractionConfig) -> ExtractionResult:
    profile = "generic"
    _assign_roles(result, profile)
    _label_visuals(result)
    _extract_paper_metadata(result)
    _link_citations(result)
    _quantities_and_claims(result, profile, getattr(cfg, "max_claims_per_document", 400))
    return result
