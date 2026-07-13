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
