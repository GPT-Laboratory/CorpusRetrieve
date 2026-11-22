"""Unit tests: collection-manifest conversion.

Files holding several publications keep every record, and basename keys are
checked for collisions.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "manifest_to_document_config.py"
_spec = importlib.util.spec_from_file_location("m2dc", TOOL)
m2dc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(m2dc)


def _manifest(*docs) -> dict:
    return {"scenarios": [{"id": "s1", "themes": [{"theme_id": "t1", "documents": list(docs)}]}]}


ISSUE_A = {"doc_id": "a", "file_path": "pdf/issue.pdf", "title": "Article A", "year": 2014,
           "handle_url": "http://h/a", "page_start": 1, "page_end": 8}
ISSUE_B = {"doc_id": "b", "file_path": "pdf/issue.pdf", "title": "Article B", "year": 2014,
           "handle_url": "http://h/b", "page_start": 9, "page_end": 20}


def test_every_record_of_a_multi_publication_file_is_kept():
    config, stats = m2dc.build(_manifest(ISSUE_A, ISSUE_B))
    records = config["issue.pdf"]["metadata"]["records"]
    assert [r["title"] for r in records] == ["Article A", "Article B"]
    assert records[1]["page_start"] == 9
    assert "title" not in config["issue.pdf"]["metadata"]   # file title would be wrong
    assert stats["multi_record_files"] == 1 and stats["records"] == 2


def test_single_record_file_keeps_flat_fields_too():
    config, _ = m2dc.build(_manifest({**ISSUE_A, "file_path": "pdf/one.pdf"}))
    meta = config["one.pdf"]["metadata"]
    assert meta["title"] == "Article A" and len(meta["records"]) == 1


def test_case_and_basename_collisions_are_reported():
    _, stats = m2dc.build(_manifest(
        {"doc_id": "x", "file_path": "a/Report.pdf", "title": "X"},
        {"doc_id": "y", "file_path": "b/report.pdf", "title": "Y"},
        {"doc_id": "z", "file_path": "c/report.pdf", "title": "Z"}))
    assert stats["case_collisions"] == [["Report.pdf", "report.pdf"]]
    assert stats["key_collisions"] == ["report.pdf"]


def test_cli_refuses_to_write_an_ambiguous_config(tmp_path):
    src = tmp_path / "manifest.json"
    src.write_text(json.dumps(_manifest(
        {"doc_id": "x", "file_path": "Report.pdf"}, {"doc_id": "y", "file_path": "report.pdf"})))
    out = tmp_path / "config.json"
    run = subprocess.run([sys.executable, str(TOOL), str(src), "-o", str(out)],
                         capture_output=True, text=True)
    assert run.returncode == 2 and not out.exists()
    ok = subprocess.run([sys.executable, str(TOOL), str(src), "-o", str(out),
                         "--allow-collisions"], capture_output=True, text=True)
    assert ok.returncode == 0 and out.exists()
