"""Chunking tests: every builder, dedup links, table row-grouping, graceful
degradation without VLM enrichment, and late-chunking offset integrity."""
from docparse.chunking import ChunkingConfig, chunk_result
from docparse.models import Element, ExtractionResult


def _el(kind, text, page=1, role=None, section=(), **kw):
    e = Element(kind=kind, text=text, page=page, section_path=list(section), **kw)
    if role:
        e.metadata["role"] = role
    return e


def _paper() -> ExtractionResult:
    r = ExtractionResult(source="paper.pdf", doc_type="pdf", page_count=6)
    title = _el("title", "Cold Extremes Under Warming Levels")
    abstract = _el("narrative",
                   "A transition from the 0.5°C to the 2.0°C warming level reduces the "
                   "multi-model mean annual number of cold days by 60%–80%. "
                   "This is shown in Figure 4 and Table 1. Cold spells weaken markedly.",
                   role="abstract")
    h_m = _el("heading", "2.3 Cold Extreme Indices", role="methods", level=2,
              section=["2 Data and Methods"])
    h_m.metadata["role"] = "methods"
    body = _el("narrative",
               "A cold day is defined as a day when the mean temperature falls below "
               "the 10th percentile. " + "The threshold follows earlier work. " * 30,
               role="methods", section=["2 Data and Methods", "2.3 Cold Extreme Indices"])
    formula = Element(kind="formula", text="EX = Σ max(Tp10 − Tday(t), 0) (2)",
                      page=2, section_path=["2 Data and Methods"],
                      structured={"number": 2,
                                  "expression": "EX = Σ max(Tp10 − Tday(t), 0)"})
    formula.metadata.update({"role": "methods", "context_before": body.id,
                             "context_after": None})
    table = Element(kind="table", text="| Model | Country | 2.0°C |", page=3,
                    section_path=["2 Data and Methods"],
                    structured={"header": ["Model", "Country", "2.0°C"],
                                "rows": [[f"Model{i}", "X", str(2020 + i)]
                                         for i in range(14)]},
                    visual_ref={"page": 3, "bbox": [0, 0, 1, 1]})
    table.metadata.update({"role": "methods", "label": "Table 1"})
    tcap = Element(kind="caption", text="Table 1. CMIP6 GCMs and warming-level years.",
                   page=3, parent_id=table.id)
    tcap.metadata["label"] = "Table 1"
    figure = Element(kind="figure",
                     text="Four-panel map of winter temperature change.",
                     page=4, section_path=["3 Results"],
                     structured={"figure_type": "map",
                                 "panels": [{"id": "A", "variable": "DJF mean T",
                                             "unit": "°C",
                                             "takeaway": "Strong north-south gradient."}]},
                     visual_ref={"page": 4, "bbox": [0, 0, 1, 1],
                                 "image_path": "assets/p4.png"})
    figure.metadata["label"] = "Figure 4"
    fcap = Element(kind="caption", text="Figure 4. Changes in cold day frequency.",
                   page=4, parent_id=figure.id)
    fcap.metadata["label"] = "Figure 4"
    results = _el("narrative",
                  "Cold days decrease by 60%–80% between warming levels, as Figure 4 "
                  "shows. The regional pattern matches Table 1. More text follows here.",
                  page=4, role="results", section=["3 Results"])
    results.metadata["quantities_labeled"] = [{"value": "60%–80%",
                                               "type": "percentage_change"}]
    results.metadata["citations"] = [{"key": "ipcc2021", "ref_id": None}]
    concl = _el("narrative", "We conclude that cold extremes weaken substantially "
                "under continued warming across all model ensembles considered.",
                page=5, role="conclusion", section=["4 Conclusions"])
    ref = _el("reference", "IPCC (2021). Climate Change 2021: The Physical Science "
              "Basis.", page=6, role="references")
    r.elements = [title, abstract, h_m, body, formula, table, tcap,
                  figure, fcap, results, concl, ref]
    r.stats["paper"] = {"title": title.text}
    r.stats["claims"] = [{
        "text": "Cold days decrease by 60%–80% between warming levels, as Figure 4 shows.",
        "element_id": results.id, "page": 4, "role": "results", "type": "result_claim",
        "quantities": [{"value": "60%–80%", "type": "percentage_change"}]}]
    r.stats["citation_graph"] = {"ipcc2021": {"surname": "IPCC", "year": "2021",
                                              "ref_id": ref.id,
                                              "mentions": [results.id]}}
    # char offsets like postprocess.finalize would assign
    pos = 0
    for e in r.elements:
        if e.text:
            e.char_start = pos
            pos += len(e.text)
            e.char_end = pos
            pos += 2
    return r


def test_all_builders_fire():
    cr = chunk_result(_paper(), ChunkingConfig(target_tokens=120, max_tokens=180))
    types = cr.stats["by_type"]
    for t in ("section_text", "claim", "table_summary", "table_row_group",
              "figure", "figure_panel", "formula", "summary", "reference"):
        assert types.get(t, 0) >= 1, f"missing chunk type {t}: {types}"
    assert types.get("table_fact", 0) == 0            # micro-facts off by default


def test_table_row_grouping_and_facts_in_metadata():
    cr = chunk_result(_paper(), ChunkingConfig(table_rows_per_group=6))
    groups = [c for c in cr.chunks if c.chunk_type == "table_row_group"]
    assert len(groups) == 3                            # 14 rows / 6 per group
    assert groups[0].metadata["facts"][0]["column"] == "Country"
    assert "Model: Model0" in groups[0].text           # kv serialization
    assert groups[0].parent_chunk_id                   # linked to summary


def test_claim_links_to_parent_section_chunk():
    cr = chunk_result(_paper())
    claim = next(c for c in cr.chunks if c.chunk_type == "claim")
    section_ids = {c.chunk_id for c in cr.chunks if c.chunk_type == "section_text"}
    assert claim.parent_chunk_id in section_ids        # retrieval can dedupe
    assert claim.quantities and claim.linked_figures == ["Figure 4"]


def test_breadcrumb_and_offsets():
    cr = chunk_result(_paper())
    sec = next(c for c in cr.chunks if c.chunk_type == "section_text"
               and c.role == "methods")
    assert sec.embed_text.startswith("[Cold Extremes Under Warming Levels >")
    assert sec.char_start is not None and sec.char_end > sec.char_start


def test_degrades_without_vlm_panels():
    r = _paper()
    fig = next(e for e in r.elements if e.kind == "figure")
    fig.structured = None
    fig.text = ""                                      # no VLM enrichment ran
    cr = chunk_result(r)
    assert cr.stats["by_type"].get("figure_panel", 0) == 0
    fc = next(c for c in cr.chunks if c.chunk_type == "figure")
    assert "Caption:" in fc.text                       # still retrievable via caption


def test_references_flagged_excluded():
    cr = chunk_result(_paper())
    ref = next(c for c in cr.chunks if c.chunk_type == "reference")
    assert ref.metadata["retrieval_default"] == "exclude"
    assert ref.metadata["citation_key"] == "ipcc2021"


def test_long_text_splits_with_sentence_overlap():
    r = ExtractionResult(source="long.txt", doc_type="text")
    sents = " ".join(f"Sentence number {i} carries some retrieval content here." for i in range(80))
    r.elements = [Element(kind="narrative", text=sents)]
    cr = chunk_result(r, ChunkingConfig(target_tokens=100, max_tokens=140,
                                        overlap_sentences=1))
    secs = [c for c in cr.chunks if c.chunk_type == "section_text"]
    assert len(secs) >= 4
    # sentence-level overlap: last sentence of chunk i starts chunk i+1
    first_sent = secs[1].text.split(".")[0] + "."
    assert first_sent in secs[0].text


# ---- reviewer-feedback regressions (boilerplate, paths, sizes, qty labels) ----

def test_boilerplate_never_reaches_chunks():
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    r.elements = [Element(kind="narrative", text=(
        "Cold extremes weaken under warming, which matters for infrastructure. "
        "This is an open access article under the terms of the Creative Commons License. "
        "© 2025 The Author(s). International Journal of Climatology. "
        "https://doi.org/10.1002/joc.1234. "
        "The signal is robust across the model ensemble considered here."))]
    cr = chunk_result(r)
    text = " ".join(c.text for c in cr.chunks)
    for bad in ("open access", "©", "doi.org", "Journal of Climatology"):
        assert bad not in text, f"boilerplate leaked: {bad}"
    assert "Cold extremes weaken" in text and "robust across" in text


def test_table_header_never_becomes_section_path():
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    good = Element(kind="heading", text="2.3 Cold Extreme Indices", level=2)
    fake1 = Element(kind="heading", text="Model Country Npar 1.0°C 1.5°C 2.0°C", level=2)
    fake2 = Element(kind="heading", text="횫Tglob 80°C days 160°C days", level=2)
    para = Element(kind="narrative", text="A cold day is defined by the 10th percentile "
                   "threshold and this paragraph is long enough to become a chunk easily.")
    para.section_path = [good.text, fake1.text, fake2.text]   # worst case
    r.elements = [good, fake1, fake2, para]
    cr = chunk_result(r)
    sec = next(c for c in cr.chunks if c.chunk_type == "section_text")
    assert sec.section_path == ["2.3 Cold Extreme Indices"]
    assert "1.5°C" not in sec.embed_text.split("\n")[0]        # breadcrumb clean


def test_fake_heading_demoted_at_source():
    from docparse.scientific import _fake_heading
    assert _fake_heading("Model Country Npar 1.0°C 1.5°C 2.0°C")
    assert _fake_heading("횫Tglob 80°C days 160°C days")
    assert not _fake_heading("2.3 Cold Extreme Indices")
    assert not _fake_heading("EX = Σ max(Tp10 − Tday, 0) (2)")  # formula, not fake


def test_small_section_chunks_merge_within_top_section():
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    els = []
    for i in range(4):                      # 4 tiny paragraphs in 4 subsections
        e = Element(kind="narrative",
                    text=f"Subsection {i} makes one short point about cold spells here.")
        e.section_path = ["3 Results", f"3.{i} Sub"]
        e.metadata["role"] = "results"
        els.append(e)
    r.elements = els
    cr = chunk_result(r)
    secs = [c for c in cr.chunks if c.chunk_type == "section_text"]
    assert len(secs) == 1                   # merged: shared top-level section + role
    assert secs[0].token_estimate > 40


def test_giant_sentence_hard_split():
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    r.elements = [Element(kind="narrative", text="cell " * 1200)]   # no sentence ends
    from docparse.chunking import ChunkingConfig as CC
    cfg = CC(target_tokens=200, max_tokens=280)
    cr = chunk_result(r, cfg)
    secs = [c for c in cr.chunks if c.chunk_type == "section_text"]
    assert len(secs) >= 4
    assert all(c.token_estimate <= cfg.max_tokens for c in secs)


def test_quantity_types_degree_days_vs_warming_level():
    from docparse.scientific import _qty_type
    ex = "The cold spell extremity index EX reaches 160°C days at the 1.0°C warming level."
    assert _qty_type("160°C days", ex) == "extremity_index"
    assert _qty_type("1.0°C", ex) == "warming_level"
    assert _qty_type("40°C", "values above 40°C at the global warming level") != "warming_level"
    assert _qty_type("80–160°C days", "degree-days accumulate") == "degree_days"


def test_per_type_token_stats_present():
    cr = chunk_result(_paper())
    bt = cr.stats["tokens_by_type"]
    assert "section_text" in bt and {"min", "median", "max"} <= set(bt["section_text"])


# ---- round-2 reviewer regressions (hierarchy collapse, roles, summary, panels) ----

def test_real_headings_with_warming_levels_survive():
    from docparse.scientific import _fake_heading, _FRONT_MATTER_HEADING
    real = ["3.1 | Changes in Cold Days at 1.0°C, 1.5°C and 2.0°C Warming Levels",
            "2 | Data and Methods", "6 | Conclusions",
            "Changes in the Frequency of Cold Days at 1.0°C, 1.5°C and 2.0°C"]
    for h in real:
        assert not _fake_heading(h) and not _FRONT_MATTER_HEADING.match(h), h
    banners = ["RESEARCH ARTICLE OPEN ACCESS", "OPEN ACCESS", "Research Article",
               "International Journal of Climatology", "Author Contributions",
               "Conflicts of Interest", "Data Availability Statement"]
    for b in banners:
        assert _FRONT_MATTER_HEADING.match(b), b


def test_role_and_path_recover_after_front_matter_banner():
    from docparse.models import ExtractionConfig
    from docparse.scientific import clean
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    banner = Element(kind="heading", text="RESEARCH ARTICLE OPEN ACCESS", level=1, page=1)
    abstract = Element(kind="narrative", text="Abstract. Cold extremes weaken with "
                       "warming across ensembles.", page=1)
    h1 = Element(kind="heading",
                 text="3.1 | Changes in Cold Days at 1.0°C, 1.5°C and 2.0°C Warming Levels",
                 level=2, page=4)
    body = Element(kind="narrative", text="Cold day frequency decreases strongly "
                   "between the successive warming levels considered.", page=4)
    parent = Element(kind="heading", text="3 | Results", level=1, page=4)
    r.elements = [banner, abstract, parent, h1, body]
    cfg = ExtractionConfig()
    clean(r, cfg)
    from docparse.postprocess import finalize
    from docparse.scientific import annotate
    finalize(r, cfg)
    annotate(r, cfg)
    assert banner.kind == "other" and banner.metadata.get("front_matter_banner")
    assert h1.kind == "heading"                       # numbered heading survives
    assert body.metadata["role"] == "results"         # NOT stuck on abstract
    assert "RESEARCH ARTICLE" not in " ".join(body.section_path)


def test_abstract_role_cannot_run_away():
    from docparse.models import ExtractionConfig
    from docparse.scientific import clean
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    els = [Element(kind="narrative", text="Abstract. The study begins here.", page=1)]
    for p in range(1, 8):                             # no headings detected at all
        for i in range(8):
            els.append(Element(kind="narrative", page=p,
                               text=f"Body paragraph {p}-{i} about methods and results."))
    r.elements = els
    cfg = ExtractionConfig()
    clean(r, cfg)
    from docparse.postprocess import finalize
    from docparse.scientific import annotate
    finalize(r, cfg)
    annotate(r, cfg)
    roles = [e.metadata["role"] for e in r.elements]
    assert roles.count("abstract") <= 30
    assert r.elements[-1].metadata["role"] == "body"  # tail downgraded


def test_summary_chunk_is_capped_and_contiguous():
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    els = []
    for i in range(60):                               # leaked-abstract worst case
        e = Element(kind="narrative",
                    text=f"Abstract-role sentence {i} with enough words to add up "
                         "to a very large total across the whole document body.")
        e.metadata["role"] = "abstract"
        els.append(e)
    r.elements = els
    from docparse.chunking import ChunkingConfig as CC
    cr = chunk_result(r, CC(summary_max_tokens=600))
    summ = [c for c in cr.chunks if c.chunk_type == "summary"]
    assert summ and all(c.token_estimate <= 600 for c in summ)


def test_figure_panels_from_prose_description():
    from docparse.chunking import _panels_from_text
    desc = ("Panel A: DJF mean temperature with a north-south gradient. "
            "Panel B: change at 1.5°C warming. Panel C shows variability decrease. "
            "Panel D: number of cold days.")
    ids = [p["id"] for p in _panels_from_text(desc)]
    assert ids == ["A", "B", "C", "D"]
    assert _panels_from_text("A single map of Finland.") == []   # no false positives


def test_figure_chunker_uses_prose_fallback():
    r = _paper()
    fig = next(e for e in r.elements if e.kind == "figure")
    fig.structured = {"figure_type": "map"}           # VLM gave prose, no JSON panels
    fig.text = ("Panel A: mean temperature. Panel B: projected change. "
                "Panel C: variability. Panel D: cold day counts.")
    cr = chunk_result(r)
    assert cr.stats["by_type"].get("figure_panel", 0) == 4


# ---- round-3 regressions (VLM truncation, Wiley captions, back matter) --------

def test_parse_facts_salvages_truncated_json():
    from docparse.enrich import _parse_facts
    trunc = ('```json\n{"type": "map", "description": "Four winter panels", '
             '"panels": [{"id": "A", "takeaway": "north-south gradient"}, '
             '{"id": "B", "takeaway": "warm')          # cut mid-string
    f = _parse_facts(trunc)
    assert f and f["description"] == "Four winter panels" and f.get("truncated")
    assert [p["id"] for p in f["panels"]] == ["A"]     # only the complete object


def test_wiley_caption_narrative_promoted_and_linked():
    from docparse.models import ExtractionConfig
    from docparse.scientific import clean
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    fig = Element(kind="figure", text="", page=3, visual_ref={"page": 3, "bbox": [0,0,1,1]})
    cap = Element(kind="narrative", page=3,
                  text="FIGURE 1   |   Winter mean temperature and its projected "
                       "change between the warming levels considered in this study.")
    r.elements = [fig, cap]
    clean(r, ExtractionConfig())
    assert cap.kind == "caption" and cap.metadata["label"] == "Figure 1"
    assert cap.parent_id == fig.id
    assert fig.metadata.get("label") == "Figure 1"
    assert r.stats.get("captions_promoted") == 1


def test_figure_chunk_gets_label_from_promoted_caption():
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    fig = Element(kind="figure", text="Panel A: cold days. Panel B: cold spells.",
                  page=3, visual_ref={"page": 3, "bbox": [0,0,1,1]})
    cap = Element(kind="caption", page=3, parent_id=fig.id,
                  text="FIGURE 5 | Changes in cold day counts.")
    r.elements = [fig, cap]
    cr = chunk_result(r)
    fc = next(c for c in cr.chunks if c["chunk_type"] == "figure") if isinstance(cr.chunks[0], dict) \
        else next(c for c in cr.chunks if c.chunk_type == "figure")
    assert fc.metadata["label"] == "Figure 5"
    pl = [c for c in cr.chunks if c.chunk_type == "figure_panel"]
    assert len(pl) == 2 and all(p.metadata["label"] == "Figure 5" for p in pl)


def test_back_matter_heading_does_not_nest_under_conclusions():
    from docparse.models import ExtractionConfig
    from docparse.postprocess import finalize
    from docparse.scientific import annotate, clean
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    h6 = Element(kind="heading", text="6 | Conclusions", level=1, page=15)
    body = Element(kind="narrative", text="We conclude cold extremes weaken.", page=15)
    ack = Element(kind="heading", text="Acknowledgements", level=2, page=16)
    thanks = Element(kind="narrative", text="The authors thank the modelling groups.", page=16)
    r.elements = [h6, body, ack, thanks]
    cfg = ExtractionConfig()
    clean(r, cfg); finalize(r, cfg); annotate(r, cfg)
    assert ack.level == 1
    assert "Conclusions" not in " ".join(thanks.section_path)
    assert thanks.metadata["role"] == "acknowledgements"


def test_section_path_whitespace_normalized():
    r = ExtractionResult(source="x.pdf", doc_type="pdf")
    e = Element(kind="narrative", text="Enough words to make a viable little chunk "
                "about methods and thresholds for cold day definitions in the study.")
    e.section_path = ["2.3   |   Cold Extreme Indices"]
    r.elements = [e]
    cr = chunk_result(r)
    sec = next(c for c in cr.chunks if c.chunk_type == "section_text")
    assert sec.section_path == ["2.3 | Cold Extreme Indices"]
