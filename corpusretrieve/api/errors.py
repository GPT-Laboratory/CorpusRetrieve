from __future__ import annotations

import logging
import re
import uuid
from http import HTTPStatus

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        detail: str,
        title: str | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.title = title or HTTPStatus(status_code).phrase
        self.headers = headers or {}


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", uuid.uuid4().hex)


def _problem(
    request: Request,
    status: int,
    code: str,
    detail: str,
    title: str | None = None,
    errors: list[dict] | None = None,
    headers: dict[str, str] | None = None,
) -> JSONResponse:
    content = {
        "type": f"urn:corpusretrieve:problem:{code}",
        "title": title or HTTPStatus(status).phrase,
        "status": status,
        "detail": detail,
        "instance": request.url.path,
        "code": code,
        "request_id": _request_id(request),
    }
    if errors:
        content["errors"] = errors
    response_headers = {"X-Request-ID": _request_id(request), **(headers or {})}
    return JSONResponse(
        content,
        status_code=status,
        media_type="application/problem+json",
        headers=response_headers,
    )


def install_error_handling(app: FastAPI) -> None:
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied
            if re.fullmatch(r"[A-Za-z0-9._-]{1,100}", supplied)
            else uuid.uuid4().hex
        )
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return _problem(
            request,
            exc.status_code,
            exc.code,
            exc.detail,
            exc.title,
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        errors = []
        for item in exc.errors():
            location = [str(part) for part in item.get("loc", ()) if part != "body"]
            errors.append({
                "field": ".".join(location) or "request",
                "message": item.get("msg", "Invalid value"),
                "code": item.get("type", "validation_error"),
            })
        return _problem(
            request,
            422,
            "validation-error",
            "The request contains missing or invalid fields.",
            "Request validation failed",
            errors,
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(request: Request, exc: StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else "The request failed."
        return _problem(
            request,
            exc.status_code,
            f"http-{exc.status_code}",
            detail,
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.exception("Unhandled API error; request_id=%s", _request_id(request))
        return _problem(
            request,
            500,
            "internal-error",
            "An unexpected error occurred. Use request_id when contacting support.",
        )
