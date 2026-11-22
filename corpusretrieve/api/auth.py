"""Optional API-key authentication.

Without keys, anyone who can reach the service can upload, read, or delete
could reach the service could upload, read, or DELETE any job. With
``API_KEYS`` set, every ``/v1`` and ``/console`` route requires a key, except
the endpoints a load balancer or a client needs before it has one.
"""
from __future__ import annotations

import hmac
import logging

from fastapi import FastAPI, Request

from corpusretrieve.api.errors import _problem
from corpusretrieve.config import Settings

logger = logging.getLogger(__name__)

PUBLIC_PATHS = frozenset({"/v1", "/v1/", "/v1/health", "/v1/capabilities"})
PROTECTED_PREFIXES = ("/v1", "/console")


def _presented_key(request: Request) -> str | None:
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip() or None
    return request.headers.get("X-API-Key") or None


def key_is_valid(presented: str | None, keys: tuple[str, ...]) -> bool:
    if not presented:
        return False
    # compare against every key so timing does not reveal which one matched
    results = [hmac.compare_digest(presented.encode(), k.encode()) for k in keys]
    return any(results)


def install_authentication(app: FastAPI, settings: Settings) -> None:
    if not settings.api_keys:
        logger.warning("API_KEYS is empty: the API accepts unauthenticated requests. "
                       "Set API_KEYS for any deployment reachable by others.")
        return

    @app.middleware("http")
    async def require_api_key(request: Request, call_next):
        path = request.url.path
        if (request.method != "OPTIONS" and path.startswith(PROTECTED_PREFIXES)
                and path not in PUBLIC_PATHS
                and not key_is_valid(_presented_key(request), settings.api_keys)):
            return _problem(request, 401, "authentication-required",
                            "Send a valid key as 'Authorization: Bearer <key>' or "
                            "'X-API-Key: <key>'.",
                            headers={"WWW-Authenticate": "Bearer"})
        return await call_next(request)
