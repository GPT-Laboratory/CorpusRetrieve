"""Unit tests: VLM figure description retries, caching, budget."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from docparse import ExtractionConfig, chunk_result
from docparse.enrich import VLMBudget, describe_figures
from docparse.models import Element, ExtractionResult
from docparse.postprocess import finalize

GOOD = '{"type": "chart", "description": "Bar chart of measured values."}'


class RateLimit(Exception):
    status_code = 429


def _figures(tmp_path: Path, contents: list[bytes], repeated: tuple[int, ...] = ()):
    r = ExtractionResult(source="x.pdf")
    for i, data in enumerate(contents):
        crop = tmp_path / f"fig{i}.png"
        crop.write_bytes(data)
        meta = {"repeated_image": True} if i in repeated else {}
        r.elements.append(Element(kind="figure", page=1, order=i, bbox=[0, 0, 50, 50],
                                  visual_ref={"image_path": str(crop)}, metadata=meta))
    r.stats["source_sha256"] = "3" * 64
    finalize(r, ExtractionConfig(detect_language=False))
    return r


def test_rate_limits_are_retried_with_backoff(tmp_path):
    r = _figures(tmp_path, [b"a"])
    calls, sleeps = [], []

    def describe(path, ctx):
        calls.append(path)
        if len(calls) < 3:
            raise RateLimit("429")
        return GOOD

    assert describe_figures(r, describe, sleep=sleeps.append, backoff_seconds=1) == 1
    assert len(calls) == 3 and sleeps == [1, 2]          # exponential backoff
    assert r.stats["figures_failed"] == 0


def test_permanent_errors_are_counted_not_raised(tmp_path):
    r = _figures(tmp_path, [b"a", b"b"])

    def describe(path, ctx):
        raise ValueError("bad image")

    assert describe_figures(r, describe, sleep=lambda s: None) == 0
    assert r.stats["figures_failed"] == 2


def test_identical_images_are_described_once(tmp_path):
    r = _figures(tmp_path, [b"same", b"same", b"other"])
    calls = []
    describe_figures(r, lambda p, c: calls.append(p) or GOOD, sleep=lambda s: None)
    assert len(calls) == 2 and r.stats["figures_cache_hits"] == 1
    assert all(e.text for e in r.elements)


def test_exhausted_budget_is_counted(tmp_path):
    r = _figures(tmp_path, [b"1", b"2", b"3", b"4"])
    budget = VLMBudget(max_calls=2)
    describe_figures(r, lambda p, c: GOOD, budget=budget, sleep=lambda s: None)
    assert r.stats["figures_described"] == 2
    assert r.stats["figures_skipped_budget"] == 2 and budget.exhausted


def test_repeated_images_are_never_sent(tmp_path):
    r = _figures(tmp_path, [b"logo", b"chart"], repeated=(0,))
    calls = []
    describe_figures(r, lambda p, c: calls.append(p) or GOOD, sleep=lambda s: None)
    assert len(calls) == 1 and r.stats["figures_skipped_repeated"] == 1


@pytest.mark.parametrize("vlm_type", ["logo", "text_block"])
def test_non_figures_reported_by_the_model_are_not_units(tmp_path, vlm_type):
    r = _figures(tmp_path, [b"x"])
    reply = json.dumps({"type": vlm_type, "description": "Publisher logo."})
    describe_figures(r, lambda p, c: reply, sleep=lambda s: None)
    assert r.elements[0].metadata["not_a_figure"] is True
    assert not [c for c in chunk_result(r, doc_id="d").chunks if c.chunk_type == "figure"]


def test_described_text_is_flagged_as_derived(tmp_path):
    r = _figures(tmp_path, [b"x"])
    describe_figures(r, lambda p, c: GOOD, sleep=lambda s: None)
    assert r.elements[0].metadata["derived_text"] is True
