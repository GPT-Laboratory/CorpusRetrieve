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
    "If it is not a real figure (a text block, logo, decoration), say so in type and keep "
    "description one sentence. Be factual; never invent values you cannot read.\n"
    "Caption for context:\n{caption}"
)


def _parse_facts(raw: str) -> dict | None:
    """Pull the JSON object out of the model reply; tolerate code fences and chatter."""
    import json as _json
    import re as _re
    m = _re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, _re.S) or _re.search(r"\{.*\}", raw, _re.S)
    if not m:
        return None
    try:
        obj = _json.loads(m.group(1) if m.lastindex else m.group(0))
        return obj if isinstance(obj, dict) and "description" in obj else None
    except Exception:
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
                el.text = facts.get("description", "").strip() or desc.strip()
                el.structured = {"figure_facts": facts}
                el.metadata["vlm_type"] = facts.get("type")
            else:                                           # fall back to prose
                el.text = desc.strip()
            el.metadata["description"] = "vlm"
            n += 1
    if n:
        result.stats["figures_described"] = n
    return n


def make_anthropic_describer(model: str = "claude-sonnet-4-6", api_key: str | None = None,
                             max_tokens: int = 700):
    """Reference implementation using the Anthropic API (pip install anthropic)."""
    import anthropic                                    # optional dependency
    client = anthropic.Anthropic(api_key=api_key) if api_key else anthropic.Anthropic()

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
