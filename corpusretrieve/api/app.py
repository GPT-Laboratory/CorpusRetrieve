from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from corpusretrieve.api.errors import install_error_handling
from corpusretrieve.api.outputs import router as outputs_router
from corpusretrieve.api.routes import router as service_router
from corpusretrieve.api.schemas import ProblemDetail
from corpusretrieve.api.uploads import router as uploads_router
from corpusretrieve.config import Settings, get_settings
from corpusretrieve.web.routes import router as console_router

PROBLEM_MEDIA_TYPE = "application/problem+json"
PROBLEM_REF = {"$ref": "#/components/schemas/ProblemDetail"}
HTTP_METHODS = {"get", "post", "put", "patch", "delete"}

OPENAPI_TAGS = [
    {
        "name": "Upload",
        "description": "Upload one document, multiple documents, or one ZIP.",
    },
    {"name": "Jobs", "description": "Monitor progress, stream events, and delete jobs."},
    {"name": "Documents", "description": "Per-document extraction, chunks, and assets."},
    {"name": "Corpus", "description": "Resources that combine every document in the job."},
    {
        "name": "Index",
        "description": "Portable dense and sparse index files; no database is required.",
    },
    {
        "name": "Knowledge Graph",
        "description": "The same graph in JSON, GraphML, and Cypher formats.",
    },
    {
        "name": "Artifacts",
        "description": "Manifest, generic file discovery, individual downloads, and ZIP bundle.",
    },
    {"name": "Service", "description": "Start guide, capabilities, and dependency health."},
]


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="CorpusRetrieve Integration API",
        version=settings.version,
        openapi_tags=OPENAPI_TAGS,
        docs_url=None,
        redoc_url="/redoc",
    )
    install_error_handling(app)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type", "X-Anthropic-API-Key", "X-Request-ID"],
        expose_headers=["Location", "Retry-After", "X-Request-ID"],
    )

    app.include_router(uploads_router)
    app.include_router(service_router)
    app.include_router(outputs_router)
    if settings.console_enabled:
        app.include_router(console_router)

    app.openapi = lambda: _openapi_schema(app)
    _mount_interface(app, settings)
    return app


def _mount_interface(app: FastAPI, settings: Settings) -> None:
    static_dir = settings.static_dir
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    if settings.console_enabled:

        @app.get("/", include_in_schema=False)
        def console_index():
            return FileResponse(static_dir / "index.html")

    @app.get("/docs", include_in_schema=False)
    def swagger_docs():
        page = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title="CorpusRetrieve API",
            swagger_ui_parameters={
                "deepLinking": True,
                "displayRequestDuration": True,
                "docExpansion": "list",
                "filter": True,
                "defaultModelsExpandDepth": 1,
                "showExtensions": True,
                "syntaxHighlight.theme": "obsidian",
            },
        )
        body = page.body.decode("utf-8").replace(
            "</head>", '<link rel="stylesheet" href="/static/api-docs.css"></head>'
        )
        return HTMLResponse(body)


def _openapi_schema(app: FastAPI) -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title, version=app.version, routes=app.routes, tags=OPENAPI_TAGS
    )
    _mark_binary_media(schema)
    schema.setdefault("components", {}).setdefault("schemas", {})["ProblemDetail"] = (
        ProblemDetail.model_json_schema(ref_template="#/components/schemas/{model}")
    )
    for path_item in schema.get("paths", {}).values():
        for method, operation in path_item.items():
            if method in HTTP_METHODS:
                _document_errors(operation)
    _describe_upload_encoding(schema)
    app.openapi_schema = schema
    return schema


def _mark_binary_media(node: Any) -> None:
    if isinstance(node, dict):
        if node.get("contentMediaType") == "application/octet-stream":
            node["format"] = "binary"
        for value in node.values():
            _mark_binary_media(value)
    elif isinstance(node, list):
        for value in node:
            _mark_binary_media(value)


def _document_errors(operation: dict[str, Any]) -> None:
    responses = operation.setdefault("responses", {})
    responses.setdefault(
        "500",
        {
            "description": "Unexpected service error. Use the request ID for support.",
            "content": {PROBLEM_MEDIA_TYPE: {"schema": PROBLEM_REF}},
        },
    )
    if operation.get("parameters") or operation.get("requestBody"):
        responses.setdefault(
            "422",
            {
                "description": "Request validation failed.",
                "content": {PROBLEM_MEDIA_TYPE: {"schema": PROBLEM_REF}},
            },
        )
    for status, response in responses.items():
        if not str(status).isdigit() or int(status) < 400:
            continue
        content = response.setdefault("content", {})
        documented = content.get("application/json")
        reference = documented.get("schema", {}).get("$ref", "") if documented else ""
        if str(status) == "422" or reference.endswith("/ProblemDetail"):
            content.pop("application/json", None)
            content[PROBLEM_MEDIA_TYPE] = documented or {"schema": PROBLEM_REF}
            response.setdefault("headers", {})["X-Request-ID"] = {
                "description": "Request correlation identifier.",
                "schema": {"type": "string"},
            }


def _describe_upload_encoding(schema: dict[str, Any]) -> None:
    operation = schema.get("paths", {}).get("/v1/jobs", {}).get("post")
    if not operation:
        return
    media = operation["requestBody"]["content"]["multipart/form-data"]
    media.setdefault("encoding", {})["files"] = {
        "contentType": "application/octet-stream"
    }


app = create_app()
