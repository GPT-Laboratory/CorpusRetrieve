"""Extractor interface + registry. Register your own backend and it's picked up."""
from __future__ import annotations

from typing import Callable, Protocol, runtime_checkable

from docparse.models import ExtractionConfig, ExtractionResult


@runtime_checkable
class Extractor(Protocol):
    doc_types: tuple[str, ...]

    def extract(self, path: str, config: ExtractionConfig) -> ExtractionResult: ...


_REGISTRY: dict[str, Extractor] = {}


def register(extractor: Extractor) -> Extractor:
    for dt in extractor.doc_types:
        _REGISTRY[dt] = extractor
    return extractor


def get_extractor(doc_type: str) -> Extractor | None:
    return _REGISTRY.get(doc_type)


def registered_types() -> list[str]:
    return sorted(_REGISTRY)


# convenience for functional extractors
def register_fn(doc_types: tuple[str, ...]) -> Callable:
    def deco(fn):
        class _Wrap:
            def __init__(self):
                self.doc_types = doc_types
            def extract(self, path, config):
                return fn(path, config)
        register(_Wrap())
        return fn
    return deco
