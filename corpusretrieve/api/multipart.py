from __future__ import annotations

from collections.abc import Callable, Coroutine
from typing import Any

from fastapi import Request, Response
from fastapi.routing import APIRoute

from corpusretrieve.config import get_settings


class LargeUploadRequest(Request):
    """A request whose multipart parser honours the configured file limit.

    FastAPI calls ``Request.form()`` without arguments, so Starlette's default
    ceiling of 1000 parts applies before any application validation runs. That
    ceiling is independent of ``MAX_FILES_PER_JOB`` and rejects the submission
    with a generic framework error rather than the service's own limit message.

    One part of headroom is allowed above the configured limit so a submission
    at exactly the limit is parsed, and a submission one over it reaches the
    upload handler and receives the documented ``too-many-files`` problem
    document instead of a parser error.

    The parser also caps each individual part at 1 MB by default, which rejects
    a ``document_config`` describing a large corpus. That ceiling is raised to
    ``MAX_FORM_FIELD_BYTES``. File parts are streamed to disk and are bounded
    separately by ``MAX_UPLOAD_BYTES``.
    """

    def form(self, **kwargs: Any):
        settings = get_settings()
        kwargs.setdefault("max_files", settings.max_files_per_job + 1)
        kwargs.setdefault("max_part_size", settings.max_form_field_bytes)
        return super().form(**kwargs)


class LargeUploadRoute(APIRoute):
    """Applies :class:`LargeUploadRequest` to the routes it is mounted on."""

    def get_route_handler(
        self,
    ) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def route_handler(request: Request) -> Response:
            return await handler(
                LargeUploadRequest(request.scope, request.receive)
            )

        return route_handler
