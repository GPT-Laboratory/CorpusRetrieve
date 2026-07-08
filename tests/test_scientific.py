"""Tests for the scientific enrichment layers."""
import os

import pytest

from docparse import ExtractionConfig, extract
from docparse.models import Element, ExtractionResult
from docparse.scientific import annotate, clean

PAPER = "mydocs/climate_paper.pdf"


def _res(*els):
    r = ExtractionResult(source="x", doc_type="pdf")
    r.elements = list(els)
    return r


def test_dehyphenation_vocab_rule():
    r = _res(
        Element(kind="narrative", text="The bias correction handles corrected output."),
        Element(kind="narrative", text="The se- verity of a bias- corrected tempera- ture spell."),
    )
    clean(r, ExtractionConfig())
    fixed = r.elements[1].text
    assert "severity" in fixed                 # broken word joined
    assert "bias-corrected" in fixed           # true compound keeps its hyphen
    assert "temperature" in fixed
    assert r.stats["hyphenation_fixes"] == 3


def test_license_noise_removed():
    r = _res(
        Element(kind="narrative", text="Real content about cold spells."),
        Element(kind="narrative", text="Downloaded from Wiley Online Library. See the Terms and Conditions."),
    )
    clean(r, ExtractionConfig())
    assert len(r.elements) == 1 and r.stats["license_noise_removed"] == 1


def test_equation_promotion():
    r = _res(Element(kind="narrative", text="Tday ≤ Tp10,DJF (1)"))
    clean(r, ExtractionConfig())
    el = r.elements[0]
    assert el.kind == "formula" and el.structured["number"] == 1


def test_roles_citations_claims():
    els = [
        Element(kind="heading", text="1 | Introduction", level=1),
        Element(kind="narrative", text="Cold spells matter (IPCC 2021; Masselot et al. 2023)."),
        Element(kind="heading", text="3 | Results", level=1),
        Element(kind="narrative",
                text="Cold days decrease by 60–80% under warming, as shown by the model ensemble."),
        Element(kind="heading", text="References", level=1),
        Element(kind="narrative", text="Masselot, P., et al. (2023). Heat and cold mortality. Lancet."),
    ]
    r = _res(*els)
    for e in r.elements:
        e.section_path = []
    annotate(r, ExtractionConfig())
    assert r.elements[1].metadata["role"] == "introduction"
    assert r.elements[3].metadata["role"] == "results"
    assert r.elements[5].kind == "reference"
    keys = [c["key"] for c in r.elements[1].metadata["citations"]]
    assert "masselot2023" in keys
    assert r.stats["citation_graph"]["masselot2023"]["ref_id"] == r.elements[5].id
    assert r.elements[3].metadata["quantities"] == ["60–80%"]
    assert any("60–80%" in c["text"] for c in r.stats["claims"])


@pytest.mark.skipif(not os.path.exists(PAPER), reason="climate paper not present")
def test_climate_paper_integration():
    r = extract(PAPER, ExtractionConfig(table_serializations=("markdown", "kv")))
    assert r.stats["paper"]["doi"] == "10.1002/joc.8875"
    assert r.stats["paper"]["year"] == 2025
    tables = [e for e in r.elements if e.kind == "table"]
    assert len(tables) >= 2
    t1 = tables[0]
    assert len(t1.structured["rows"]) == 25                      # the 25 CMIP6 models
    assert any("MIROC6" in " ".join(row) for row in t1.structured["rows"])
    assert r.counts().get("formula", 0) >= 4
    assert r.stats["reference_entries"] >= 40
    assert r.stats["citations"]["linked_to_reference"] >= 40
    assert r.stats["role_distribution"].get("methods", 0) > 10
    assert len(r.stats.get("claims", [])) >= 5
    assert not any("Wiley Online" in e.text for e in r.elements)
