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
import hashlib
import mimetypes
import os
import re
import time
from collections import Counter

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


class VLMBudget:
    """Shared cap on VLM calls across a whole corpus build.

    Without a cap, a provider usage limit makes every later figure fail one at
    a time. A budget makes exhaustion an explicit, countable event.
    """

    def __init__(self, max_calls: int | None = None) -> None:
        self.max_calls = max_calls
        self.used = 0
        self.exhausted = False

    def take(self) -> bool:
        if self.max_calls is not None and self.used >= self.max_calls:
            self.exhausted = True
            return False
        self.used += 1
        return True


class VLMCircuit:
    """Stops figure descriptions for the rest of a build once they cannot work.

    A wrong key, a missing model or an exhausted credit balance fails every
    later call in exactly the same way, so the circuit opens on the first such
    error. Any other error opens it after ``max_consecutive`` failures in a
    row. While open, figures are counted as ``figures_skipped_vlm_unavailable``
    instead of being sent, so a build is not slowed down by thousands of
    doomed calls and their retry back-off.
    """

    def __init__(self, max_consecutive: int = 25) -> None:
        self.max_consecutive = max(1, max_consecutive)
        self.consecutive = 0
        self.open = False
        self.reason: str | None = None

    def success(self) -> None:
        self.consecutive = 0

    def failure(self, exc: Exception) -> None:
        self.consecutive += 1
        if _is_fatal(exc):
            self._trip(f"{type(exc).__name__}: {_short(exc)}")
        elif self.consecutive >= self.max_consecutive:
            self._trip(f"{self.consecutive} consecutive failures, last "
                       f"{type(exc).__name__}: {_short(exc)}")

    def _trip(self, reason: str) -> None:
        if not self.open:
            self.open = True
            self.reason = reason


def _status(exc: Exception) -> int | None:
    return getattr(exc, "status_code", None) or getattr(exc, "status", None)


def _short(exc: Exception, limit: int = 300) -> str:
    text = re.sub(r"req_[A-Za-z0-9]+", "req_…", str(exc))
    return text[:limit]


_FATAL_WORDS = ("credit balance", "billing", "quota", "invalid x-api-key",
                "permission", "not_found_error", "model:")


def _is_fatal(exc: Exception) -> bool:
    """An error that every later call will repeat: stop instead of retrying."""
    status = _status(exc)
    if status in (401, 403, 404):
        return True
    name = type(exc).__name__.lower()
    if any(k in name for k in ("authentication", "permissiondenied", "notfound")):
        return True
    text = str(exc).lower()
    return status == 400 and any(word in text for word in _FATAL_WORDS)


def _is_retryable(exc: Exception) -> bool:
    if _is_fatal(exc):
        return False
    status = _status(exc)
    if status in (408, 409, 429, 500, 502, 503, 504, 529):
        return True
    name = type(exc).__name__.lower()
    return any(k in name for k in ("ratelimit", "overload", "timeout", "connection"))


def _image_digest(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_NON_FIGURE_TYPES = {"logo", "text_block"}


def describe_figures(result: ExtractionResult, describe, kinds=("figure",), overwrite=False,
                     cache: dict | None = None, budget: VLMBudget | None = None,
                     max_retries: int = 3, backoff_seconds: float = 2.0,
                     sleep=time.sleep, circuit: VLMCircuit | None = None) -> int:
    """Run ``describe(image_path, context)`` over figure crops; returns count described.

    * identical images (same bytes) are described once via ``cache``
    * images repeated across many pages (logos) are skipped
    * transient provider errors are retried with exponential backoff
    * errors that every later call would repeat open ``circuit``, after which
      the remaining figures are counted rather than sent
    * every outcome is counted in ``result.stats`` so the manifest can report it
    """
    captions = {e.parent_id: e.text for e in result.elements if e.kind == "caption" and e.parent_id}
    stats = result.stats
    for key in ("figures_described", "figures_failed", "figures_skipped_budget",
                "figures_skipped_vlm_unavailable", "figures_cache_hits",
                "figures_skipped_repeated"):
        stats.setdefault(key, 0)
    cache = cache if cache is not None else {}
    failures: Counter[str] = Counter()
    n = 0
    for el in result.elements:
        if el.kind not in kinds or (el.text and not overwrite):
            continue
        if el.metadata.get("repeated_image"):
            stats["figures_skipped_repeated"] += 1
            continue
        path = (el.visual_ref or {}).get("image_path")
        if not path or not os.path.exists(path):
            continue
        context = captions.get(el.id, "")
        key = f"{_image_digest(path)}:{context}"
        desc = cache.get(key)
        if desc is not None:
            stats["figures_cache_hits"] += 1
        else:
            if circuit is not None and circuit.open:
                stats["figures_skipped_vlm_unavailable"] += 1
                continue
            if budget is not None and not budget.take():
                stats["figures_skipped_budget"] += 1
                continue
            desc, attempt = None, 0
            while True:
                try:
                    desc = describe(path, context)
                    if circuit is not None:
                        circuit.success()
                    break
                except Exception as exc:          # enrichment must never break parsing
                    if attempt < max_retries and _is_retryable(exc):
                        sleep(backoff_seconds * (2 ** attempt))
                        attempt += 1
                        continue
                    failures[f"{type(exc).__name__}: {_short(exc)}"] += 1
                    stats["figures_failed"] += 1
                    if circuit is not None:
                        circuit.failure(exc)
                    break
            if desc is None:
                continue
            cache[key] = desc
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
                if facts.get("type") in _NON_FIGURE_TYPES:
                    el.metadata["not_a_figure"] = True
            else:                                           # fall back to prose
                el.text = re.sub(r"```(?:json)?|```", "", desc).strip()
            el.metadata["description"] = "vlm"
            # model-written text: it has no character span in the source
            el.metadata["derived_text"] = True
            n += 1
    # one warning per distinct error, with its count, instead of one per figure
    for message, count in failures.most_common():
        result.warnings.append(f"figure description failed ({count} figure(s)): {message}")
    if circuit is not None and circuit.open and stats["figures_skipped_vlm_unavailable"]:
        result.warnings.append(
            f"figure descriptions stopped for this build ({circuit.reason}); "
            f"{stats['figures_skipped_vlm_unavailable']} figure(s) in this document not sent")
    stats["figures_described"] += n
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
                             max_tokens: int = 1600, timeout_seconds: float = 120.0):
    """Reference implementation using the Anthropic API (pip install anthropic).

    The SDK's own retries are turned off: ``describe_figures`` retries, so
    leaving both on multiplies the attempts and the waiting per figure.
    """
    import anthropic  # optional dependency
    client = anthropic.Anthropic(api_key=_resolve_api_key(api_key), max_retries=0,
                                 timeout=timeout_seconds)

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
