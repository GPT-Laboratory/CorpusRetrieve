"""Optional multimodal enrichment: figure understanding via a pluggable VLM.

Panel-by-panel description of maps/plots cannot be done deterministically — it
needs a vision model. This module provides the seam:

    from docparse import extract, ExtractionConfig
    from docparse.enrich import describe_figures, make_anthropic_describer

    cfg = ExtractionConfig(assets_dir="assets")        # crops must exist on disk
    result = extract("paper.pdf", cfg)
    describe_figures(result, make_anthropic_describer())   # needs ANTHROPIC_API_KEY

Any callable ``(image_path, context) -> str`` works — a local VLM, a cloud API,
or a human-in-the-loop tool. The description becomes the figure's ``text`` (so it
embeds and retrieves like any other element) and is kept in metadata too.
"""
from __future__ import annotations

import base64
import mimetypes
import os
import re

from docparse.models import ExtractionResult

_PROMPT = (
    "You are indexing a figure from a document for a retrieval system. Reply with a "
    "single JSON object, nothing else:\n"
    "{{\n"
    '  "type": "chart|map|diagram|photo|table_image|text_block|logo|other",\n'
    '  "description": "dense panel-by-panel prose: what each panel shows, axes/legends/'
    'units, regions if a map, notable values, extremes, trends, main takeaway",\n'
    '  "panels": [{{"id": "A", "label": "...", "variable": "...", "unit": "...",\n'
    '              "notable_values": [{{"item": "region/series/category", "value": "..."}}],\n'
    '              "takeaway": "..."}}]\n'
    "}}\n"
    "The panels array is REQUIRED whenever the figure has multiple panels/subplots "
    "(labeled A/B/C, a/b/c, or arranged in a grid) — one entry per panel. "
"If it is not a real figure (a text block, logo, decoration), say so in type and keep "
    "description one sentence. Be factual; never invent values you cannot read.\n"
    "Caption for context:\n{caption}"
)


def _parse_facts(raw: str) -> dict | None:
    """Pull the JSON object out of the model reply; tolerate code fences, chatter,
    and nested braces (a lazy regex truncates at the first inner '}')."""
    import json as _json
    import re

    def balanced(text: str):
        start = text.find("{")
        while start != -1:
            depth = 0
            for j in range(start, len(text)):
                if text[j] == "{":
                    depth += 1
                elif text[j] == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start:j + 1]
                        break
            start = text.find("{", start + 1)

    for cand in balanced(raw):
        try:
            obj = _json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, dict) and "description" in obj:
            return obj

    # salvage: the reply was truncated mid-JSON (hit max_tokens). Recover what
    # is recoverable — the description string and any COMPLETE panel objects —
    # rather than dumping raw fenced JSON into the element text.
    out: dict = {}
    if (m := re.search(r'"type"\s*:\s*"([^"]*)"', raw)):
        out["type"] = m.group(1)
    if (m := re.search(r'"description"\s*:\s*"((?:[^"\\]|\\.)*)', raw)):
        out["description"] = m.group(1).replace('\\"', '"').rstrip("\\")
    panels = []
    if (pm := re.search(r'"panels"\s*:\s*\[', raw)):
        for cand in balanced(raw[pm.end():]):
            try:
                p = _json.loads(cand)
            except Exception:
                break                              # first broken object = truncation point
            if isinstance(p, dict) and p.get("id"):
                panels.append(p)
    if panels:
        out["panels"] = panels
    if out.get("description"):
        out["truncated"] = True
        return out
    return None


def describe_figures(result: ExtractionResult, describe, kinds=("figure",), overwrite=False) -> int:
    """Run ``describe(image_path, context)`` over figure crops; returns count described."""
    captions = {e.parent_id: e.text for e in result.elements if e.kind == "caption" and e.parent_id}
    n = 0
    for el in result.elements:
        if el.kind not in kinds or (el.text and not overwrite):
            continue
        path = (el.visual_ref or {}).get("image_path")
        if not path or not os.path.exists(path):
            continue
        try:
            desc = describe(path, captions.get(el.id, ""))
        except Exception as exc:                       # enrichment must never break parsing
            result.warnings.append(f"figure description failed for {path}: {exc}")
            continue
        if desc:
            facts = _parse_facts(desc)
            if facts:                                       # structured, queryable
                prose = facts.get("description", "").strip()
                for panel in facts.get("panels") or []:
                    if isinstance(panel, dict) and panel.get("takeaway"):
                        prose += f"\nPanel {panel.get('id', '?')}: {panel['takeaway']}"
                el.text = prose or desc.strip()
                el.structured = {
                    "figure_number": el.metadata.get("label"),      # "Figure 1", from caption
                    "figure_type": facts.get("type"),
                    "panels": facts.get("panels") or [],
                    "figure_facts": facts,
                }
                el.metadata["vlm_type"] = facts.get("type")
            else:                                           # fall back to prose
                el.text = re.sub(r"```(?:json)?|```", "", desc).strip()
            el.metadata["description"] = "vlm"
            n += 1
    if n:
        result.stats["figures_described"] = n
    return n


def _resolve_api_key(api_key: str | None) -> str:
    """Explicit arg > environment > .env file (searched upward from CWD).

    The .env load is a no-op if python-dotenv is missing or no file exists,
    so nothing here makes the VLM stage mandatory — it stays opt-in.
    """
    if api_key:
        return api_key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        try:
            from dotenv import find_dotenv, load_dotenv
            load_dotenv(find_dotenv(usecwd=True))
        except ImportError:
            pass
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Put ANTHROPIC_API_KEY=sk-ant-... in a .env "
            "file at the project root (see .env.example) or export it in your shell.")
    return key


def make_anthropic_describer(model: str = "claude-sonnet-4-6", api_key: str | None = None,
                             max_tokens: int = 1600):
    """Reference implementation using the Anthropic API (pip install anthropic)."""
    import anthropic  # optional dependency
    client = anthropic.Anthropic(api_key=_resolve_api_key(api_key))

    def describe(image_path: str, context: str = "") -> str:
        media = mimetypes.guess_type(image_path)[0] or "image/png"
        with open(image_path, "rb") as fh:
            data = base64.standard_b64encode(fh.read()).decode()
        msg = client.messages.create(
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
                {"type": "text", "text": _PROMPT.format(caption=context or "(none)")},
            ]}])
        return "".join(b.text for b in msg.content if b.type == "text")

    return describe
