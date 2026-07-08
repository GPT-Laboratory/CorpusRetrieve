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

_TEXT_KINDS = {"title", "heading", "narrative", "list_item", "caption", "reference"}

# --------------------------------------------------------------------------- clean

from functools import lru_cache


@lru_cache(maxsize=8)
def _license_re(patterns: tuple[str, ...]) -> re.Pattern:
    return re.compile("|".join(patterns), re.IGNORECASE)

_SOFT_CHARS = {"\u00ad": "", "\u2010": "-", "\u2011": "-", "\u202f": " ", "\u00a0": " "}
_HYPH_BREAK = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]+)-\s+([A-Za-zÀ-ÖØ-öø-ÿ]+)")

_MATH_SYMBOLS = set("≤≥±×÷∑∏∫√ΔδσμλνΦφθτ≈≠≡→←∂∞∝∈")
_EQ_NUM = re.compile(r"^\(\d{1,2}\)\s*|\s*\(\d{1,2}\)\s*$")


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
            vocab.update(w.lower() for w in re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ]{3,}", clean_text))
    return vocab


def _dehyphenate(text: str, vocab: set[str]) -> tuple[str, int]:
    n = 0

    def fix(m: re.Match) -> str:
        nonlocal n
        left, right = m.group(1), m.group(2)
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
    return (numbered and (symbols >= 1 or words <= 12)) or (symbols >= 2 and words <= 30)


def clean(result: ExtractionResult, cfg: ExtractionConfig) -> ExtractionResult:
    # 1) publisher license / download boilerplate — anywhere on the page, any orientation
    kept: list[Element] = []
    removed = 0
    lic_re = _license_re(tuple(cfg.noise_patterns))
    for el in result.elements:
        if el.kind in _TEXT_KINDS and el.text and lic_re.search(el.text) \
                and (len(el.text) < 600 or len(lic_re.findall(el.text)) >= 2):
            removed += 1
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
    nospace = re.compile(r"([A-Za-zÀ-ÖØ-öø-ÿ]{2,})-([a-zà-ÿ]{2,})")

    def fix_nospace(m: re.Match) -> str:
        l, r = m.group(1), m.group(2)
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
        if el.kind in ("narrative", "list_item") and el.text and _looks_like_equation(el.text, cfg):
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
    ("abstract", r"abstract|summary of the (paper|study)"),
    ("references", r"^references?$|^bibliography$|^literature cited$"),
    ("acknowledgements", r"acknowledg"),
    ("appendix", r"^appendi|supplementary|supporting information"),
    ("funding", r"funding"),
    ("methods", r"method|data and|materials|experimental setup|model description|indices"),
    ("results", r"result|finding"),
    ("discussion", r"discussion"),
    ("conclusion", r"conclusion|concluding|summary$"),
    ("introduction", r"introduction|background"),
]


def _role_for(last_heading: str, chain: str = "") -> str | None:
    last = re.sub(r"^[\d.\s|]+", "", last_heading.lower()).strip()   # "2 | Data and Methods"
    for role, pat in _ROLE_RULES:
        if re.search(pat, last):
            return role
    if chain:
        loose = " ".join(re.sub(r"^[\d.\s|]+", "", h.lower()) for h in chain.split(" / "))
        for role, pat in _ROLE_RULES:
            if re.search(pat.replace("^", "").replace("$", ""), loose):
                return role
    return None


def _assign_roles(result: ExtractionResult) -> None:
    dist: Counter[str] = Counter()
    current = "front_matter"
    body_started = False
    for el in result.elements:
        if el.kind in ("title", "heading"):
            role = _role_for(el.text, " / ".join(el.section_path + [el.text]))
            if role:
                current = role
                body_started = body_started or role not in ("front_matter", "abstract")
            elif body_started:
                current = current if (el.level or 1) > 1 else "body"
            else:
                current = "front_matter"
        elif not body_started and re.match(r"^\s*abstract\b", el.text or "", re.I):
            current = "abstract"
        el.metadata["role"] = current
        dist[current] += 1
    result.stats["role_distribution"] = dict(dist)


_DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:a-zA-Z0-9]+\b")
_DATE_RE = re.compile(r"(received|revised|accepted)[:\s]+(\d{1,2}\s+\w+\s+(?:19|20)\d{2})", re.I)


def _extract_paper_metadata(result: ExtractionResult) -> None:
    page1 = [e for e in result.elements if (e.page or 1) <= 2 and e.text]
    all_p1 = " ".join(e.text for e in page1)
    paper: dict = {}

    if (m := _DOI_RE.search(all_p1)):
        paper["doi"] = m.group(0).rstrip(".,;")
    title_el = next((e for e in result.elements
                     if e.kind in ("title", "heading") and (e.level or 9) <= 1
                     and (e.page or 1) == 1 and len(e.text) > 15), None)
    if title_el:
        paper["title"] = title_el.text
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
        s, y = surname.lower(), year[:4]
        for r in refs:
            t = r.text.lower()
            if s in t and y in r.text:
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


_QTY_RES = [
    re.compile(r"\d+(?:\.\d+)?\s*(?:–|—|‒|-|to)\s*\d+(?:\.\d+)?\s*(?:%|°C|K\b|days?\b|years?\b)"),
    re.compile(r"\d+(?:\.\d+)?\s*(?:%|°C|hPa|mm)\b"),
    re.compile(r"once in (?:about |approximately )?\d+(?:\s*(?:–|-|to)\s*\d+)?\s*years?", re.I),
    re.compile(r"\b\d+(?:\.\d+)?[- ](?:fold|times)\b", re.I),
]
_UNCERTAINTY = re.compile(r"±|\bconfidence\b|uncertaint|inter-?model|\bspread\b|\bensemble range\b", re.I)
_CLAIM_VERB = re.compile(r"\b(show|shows|shown|found|find|increase[sd]?|decrease[sd]?|decline[sd]?|"
                         r"reduce[sd]?|project(?:ed|s)?|indicate[sd]?|suggest(?:ed|s)?|"
                         r"become[s]?|weaken(?:ed|s)?|warm(?:ing|ed|s)?|occur(?:red|s)?)\b", re.I)
_CLAIM_ROLES = {"abstract", "results", "discussion", "conclusion"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


def _quantities_and_claims(result: ExtractionResult) -> None:
    claims: list[dict] = []
    for el in result.elements:
        if el.kind not in ("narrative", "list_item", "caption") or not el.text:
            continue
        qs: list[str] = []
        for rx in _QTY_RES:
            qs += rx.findall(el.text)
        qs = list(dict.fromkeys(q.strip() for q in qs))
        if qs:
            el.metadata["quantities"] = qs[:20]
        if _UNCERTAINTY.search(el.text):
            el.metadata["uncertainty"] = True
        if el.kind == "caption":
            continue                       # quantified captions -> visual_evidence, below
        if qs and el.metadata.get("role") in _CLAIM_ROLES and len(claims) < 40:
            for sent in _SENT_SPLIT.split(el.text):
                if len(sent) < 30 or len(sent) > 420:
                    continue
                if _CLAIM_VERB.search(sent) and any(rx.search(sent) for rx in _QTY_RES):
                    claims.append({"text": sent.strip(), "element_id": el.id,
                                   "page": el.page, "role": el.metadata.get("role")})
                    if len(claims) >= 40:
                        break
    if claims:
        result.stats["claims"] = claims
    evidence = [{"text": e.text, "element_id": e.id, "page": e.page,
                 "figure_id": e.parent_id, "quantities": e.metadata.get("quantities", [])}
                for e in result.elements
                if e.kind == "caption" and e.metadata.get("quantities")]
    if evidence:
        result.stats["visual_evidence"] = evidence


def annotate(result: ExtractionResult, cfg: ExtractionConfig) -> ExtractionResult:
    _assign_roles(result)
    _extract_paper_metadata(result)
    _link_citations(result)
    _quantities_and_claims(result)
    return result
