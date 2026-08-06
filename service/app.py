"""ASGI entry point: ``uvicorn service.app:app``."""
from corpusretrieve.api.app import app

__all__ = ["app"]
