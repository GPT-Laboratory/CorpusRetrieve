"""Unit tests: multilingual text rules, quantities, claims, roles, labels.

Each test names the behaviour it covers and, where relevant, the failure it
guards against.
"""
from __future__ import annotations

import pytest

from docparse.models import Element, ExtractionConfig, ExtractionResult
from docparse.scientific import (
    _dehyphenate,
    _qty_type,
    _role_for,
    annotate,
    clean,
    find_quantities,
)
from docparse.text import (
    estimate_tokens,
    garbled_ratio,
    is_suspended_hyphen,
    split_sentences,
)
from docparse.vectorization.canonical import canonical_text, normalize_quantity

# ---------------------------------------------------------- suspended hyphens

@pytest.mark.parametrize("text", [
    "maa- ja metsätalous",          # Finnish: agriculture and forestry
    "kasvi- tai eläintuotanto",     # Finnish "or"
    "pre- and post-processing",     # English
    "jord- och skogsbruk",          # Swedish
])
def test_suspended_hyphen_is_preserved_by_line_break_repair(text):
    vocab = {"maa", "ja", "metsätalous", "pre", "and", "post", "jord", "och", "kasvi", "tai"}
    assert _dehyphenate(text, vocab)[0] == text


def test_real_line_break_hyphen_is_still_repaired():
    assert _dehyphenate("metsä- talous on tärkeää", {"metsätalous"})[0].startswith("metsätalous")


def test_canonical_text_no_longer_rejoins_hyphens():
    chunk = {"chunk_type": "section_text", "text": "Maa- ja metsätalous kasvoi."}
    assert "Maa- ja metsätalous" in canonical_text(chunk)


def test_connective_lookup_is_case_insensitive():
    assert is_suspended_hyphen("JA") and not is_suspended_hyphen("talous")


# ------------------------------------------------------- sentence splitting

def test_sentences_split_before_finnish_and_swedish_capitals():
    text = "Tulos oli hyvä. Ääriarvot kasvoivat. Öljyn osuus laski. Åkermarken minskade."
    assert len(split_sentences(text)) == 4


@pytest.mark.parametrize("text, n", [
    ("Arvo kasvoi mm. kokeessa. Seuraava lause.", 2),
    ("Esim. ohra ja vehnä. Toinen lause.", 2),
    ("As shown by Smith et al. 2012 the yield rose. Next.", 2),
    ("See Fig. 3 for details. Next sentence.", 2),
])
def test_sentence_splitter_respects_abbreviations(text, n):
    assert len(split_sentences(text)) == n


# --------------------------------------------------------- token estimates

def test_token_estimate_does_not_undercount_finnish():
    finnish = "maatalousyrittäjien ympäristökorvausjärjestelmän toimeenpanosuunnitelma"
    assert estimate_tokens(finnish) > round(3 * 4 / 3)
    assert estimate_tokens("the cat sat on the mat") == 8        # English unchanged


# ------------------------------------------------------------ garbled text

def test_garbled_ratio_detects_undecodable_text_layers():
    assert garbled_ratio("\ufffd\ufffd\ue001\ue002 ab") > 0.5
    assert garbled_ratio("Normal Finnish text: määrä") == 0.0


# ------------------------------------------------------------- quantities

@pytest.mark.parametrize("text, expected", [
    ("yield rose 35% compared to 2010", "35%"),
    ("from 60% to 85%.", "60% to 85%"),
    ("osuus oli 35 %.", "35 %"),                            # Finnish spacing
    ("kasvoi 3,5 % vuodessa", "3,5 %"),                     # decimal comma
    ("kuorma 4200 kg/ha", "4200 kg/ha"),
    ("lämpötila −5 °C", "−5 °C"),                           # Unicode minus
    ("hinta 12 €", "12 €"),
])
def test_quantity_parser_finds_common_forms(text, expected):
    assert expected in find_quantities(text)


@pytest.mark.parametrize("text", ["vuonna 2015", "5 mmol", "table 3", "Section 2.1"])
def test_quantity_parser_ignores_non_quantities(text):
    assert find_quantities(text) == []


def test_generic_quantity_types_are_domain_neutral():
    assert _qty_type("4200 kg/ha", "kuorma") == "mass_per_area"
    assert _qty_type("35%", "share increased") == "percentage_change"
    assert _qty_type("1.5°C", "global warming level of 1.5°C") == "temperature"


def test_quantity_type_is_subject_neutral():
    assert _qty_type("1.5°C", "temperature in test chamber") == "temperature"
    assert _qty_type("4200 kg/ha", "measured surface loading") == "mass_per_area"


def test_normalize_quantity_handles_decimal_comma_and_ranges():
    assert normalize_quantity({"value": "3,5 %"})["value_norm"] == 3.5
    q = normalize_quantity({"value": "60% to 85%"})
    assert (q["value_norm_min"], q["value_norm_max"], q["unit"]) == (60.0, 85.0, "percent")
    assert normalize_quantity({"value": "4200 kg/ha"})["unit"] == "kg_per_ha"


# ---------------------------------------------------------------- claims

def _result(*specs) -> ExtractionResult:
    r = ExtractionResult(source="x.pdf")
    for i, (kind, text, level) in enumerate(specs):
        r.elements.append(Element(kind=kind, text=text, level=level, page=1, order=i))
    from docparse.postprocess import finalize
    r.stats["source_sha256"] = "0" * 64
    finalize(r, ExtractionConfig(detect_language=False))
    return r


def test_english_percent_claims_are_extracted():
    r = _result(("heading", "Results", 1),
                ("narrative", "Yield rose 35% compared to 2010. The share increased "
                              "from 60% to 85%.", None))
    annotate(r, ExtractionConfig())
    assert len(r.stats["claims"]) == 2


def test_finnish_claims_in_reports_without_imrad_headings():
    r = _result(("heading", "Laitteen mittaukset", 1),
                ("narrative", "Laitteen tehokkuus kasvoi 12 % vuosina 2010–2015. "
                              "Energiankulutus väheni 8 kWh kokeessa.", None))
    annotate(r, ExtractionConfig())
    assert len(r.stats.get("claims", [])) == 2


def test_claim_cap_is_configurable():
    body = " ".join(f"Laitteen tehokkuus kasvoi {i} % kokeessa vuonna 2015." for i in range(1, 60))
    r = _result(("heading", "Tulokset", 1), ("narrative", body, None))
    annotate(r, ExtractionConfig(max_claims_per_document=10))
    assert len(r.stats["claims"]) == 10
    r2 = _result(("heading", "Tulokset", 1), ("narrative", body, None))
    annotate(r2, ExtractionConfig())
    assert len(r2.stats["claims"]) == 59


# ----------------------------------------------------------------- roles

@pytest.mark.parametrize("heading, role", [
    ("2 Aineisto ja menetelmät", "methods"),
    ("Tulokset ja tarkastelu", "results"),
    ("Tiivistelmä", "abstract"),
    ("Lähdeluettelo", "references"),
    ("Kirjallisuus", "references"),
    ("Sammandrag", "abstract"),
    ("Slutsatser", "conclusion"),
    ("Johtopäätökset", "conclusion"),
])
def test_finnish_and_swedish_section_roles(heading, role):
    assert _role_for(heading) == role


def test_ambiguous_heading_does_not_get_domain_specific_role():
    assert _role_for("Indices") is None


# --------------------------------------------------- Finnish caption labels

@pytest.mark.parametrize("caption, label, kind", [
    ("Taulukko 3. Sadot", "Table 3", "table"),
    ("Kuvio 2. Kehitys", "Figure 2", "figure"),
    ("Kuva 4 Kartta", "Figure 4", "figure"),
    ("Tabell 1. Skördar", "Table 1", "table"),
])
def test_finnish_and_swedish_captions_are_labelled_and_linked(caption, label, kind):
    r = ExtractionResult(source="x.pdf")
    target = Element(kind=kind, page=1, order=0, bbox=[72, 100, 300, 200])
    cap = Element(kind="narrative", text=caption, page=1, order=1, bbox=[72, 205, 300, 215])
    r.elements = [target, cap]
    clean(r, ExtractionConfig())
    assert cap.kind == "caption" and cap.metadata["label"] == label
    assert cap.parent_id == target.id and target.metadata["label"] == label


def test_table_caption_regex_matches_taulukko():
    import re
    assert re.match(ExtractionConfig().table_caption_regex, "Taulukko 1. Sadot", re.I)
