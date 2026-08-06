"""Exercise a running CorpusRetrieve deployment end to end.

Usage:
    python tools/api_smoke_test.py [document ...] [--base-url URL] [--delete-job]

With no documents the client generates two temporary text files, so the smoke
test is runnable against a fresh deployment without any sample data.
"""
from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import httpx

POLL_SECONDS = 1.0
SAMPLE_DOCUMENTS = {
    "overview.txt": (
        "Overview\n\nThis document records a measurement procedure.\n\n"
        "Method\n\nSamples were processed in three batches of 40 units.\n\n"
        "Results\n\nThroughput increased by 18 percent over the baseline.\n"
    ),
    "appendix.txt": (
        "Appendix\n\nCalibration was repeated before every batch.\n\n"
        "References\n\nInternal procedure note 12.\n"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--corpus-name", default="smoke-test")
    parser.add_argument("--output-dir", type=Path, default=Path("api-test-output"))
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--enable-vlm", action="store_true")
    parser.add_argument("--api-key", default=None, help="X-Anthropic-API-Key override")
    parser.add_argument("--delete-job", action="store_true")
    return parser.parse_args()


class SmokeFailure(RuntimeError):
    pass


def checked(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        try:
            body = response.json()
            detail = body.get("detail") or body
        except (json.JSONDecodeError, AttributeError):
            detail = response.text
        raise SmokeFailure(
            f"{response.request.method} {response.request.url} -> "
            f"HTTP {response.status_code}: {detail}"
        )
    return response


def get_json(client: httpx.Client, path: str):
    return checked(client.get(path)).json()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def download(client: httpx.Client, path: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", path) as response:
        checked(response)
        with destination.open("wb") as handle:
            for chunk in response.iter_bytes():
                handle.write(chunk)


def sample_files(stack: ExitStack) -> list[Path]:
    directory = Path(stack.enter_context(tempfile.TemporaryDirectory()))
    paths = []
    for name, body in SAMPLE_DOCUMENTS.items():
        path = directory / name
        path.write_text(body, encoding="utf-8")
        paths.append(path)
    return paths


def submit(client: httpx.Client, args: argparse.Namespace, paths: list[Path]) -> str:
    with ExitStack() as stack:
        uploads = [
            (
                "files",
                (
                    path.name,
                    stack.enter_context(path.open("rb")),
                    mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                ),
            )
            for path in paths
        ]
        headers = {"X-Anthropic-API-Key": args.api_key} if args.api_key else {}
        response = checked(
            client.post(
                "/v1/jobs",
                files=uploads,
                data={
                    "corpus_name": args.corpus_name,
                    "metadata": json.dumps({"origin": "api_smoke_test"}),
                    "enable_vlm": "true" if args.enable_vlm else "false",
                },
                headers=headers,
            )
        )
    payload = response.json()
    print(f"  accepted  job={payload['job_id']} mode={payload['submission_mode']}")
    for warning in payload.get("warnings", []):
        print(f"  warning   {warning}")
    return payload["job_id"]


def wait_for_completion(client: httpx.Client, job_id: str, timeout: int) -> dict:
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        state = get_json(client, f"/v1/jobs/{job_id}")
        marker = (state["stage"], state["progress"])
        if marker != last:
            print(f"  {state['progress']:3d}%  {state['stage']:<10} {state['message']}")
            last = marker
        if state["state"] == "completed":
            return state
        if state["state"] == "failed":
            raise SmokeFailure(f"Job failed: {state.get('error')}")
        time.sleep(POLL_SECONDS)
    raise SmokeFailure(f"Job did not finish within {timeout} seconds")


def collect(client: httpx.Client, job_id: str, output_dir: Path) -> None:
    root = output_dir / job_id
    manifest = get_json(client, f"/v1/jobs/{job_id}/manifest")
    save_json(root / "manifest.json", manifest)

    documents = get_json(client, f"/v1/jobs/{job_id}/documents")
    save_json(root / "documents.json", documents)
    save_json(
        root / "corpus-chunks.json", get_json(client, f"/v1/jobs/{job_id}/corpus/chunks")
    )

    for document in documents["documents"]:
        doc_id = document["doc_id"]
        base = f"/v1/jobs/{job_id}/documents/{doc_id}"
        save_json(root / doc_id / "extraction.json", get_json(client, f"{base}/extraction"))
        save_json(root / doc_id / "chunks.json", get_json(client, f"{base}/chunks"))
        assets = get_json(client, f"{base}/assets")
        save_json(root / doc_id / "assets.json", assets)
        for asset in assets["assets"][:5]:
            download(client, asset["download"], root / doc_id / "assets" / asset["path"])

    save_json(root / "index.json", get_json(client, f"/v1/jobs/{job_id}/index"))
    save_json(
        root / "sparse-encoder.json",
        get_json(client, f"/v1/jobs/{job_id}/index/sparse-encoder"),
    )
    download(client, f"/v1/jobs/{job_id}/index/vectors", root / "vectors.jsonl")

    save_json(root / "graph.json", get_json(client, f"/v1/jobs/{job_id}/graph"))
    download(client, f"/v1/jobs/{job_id}/graph/graphml", root / "graph.graphml")
    download(client, f"/v1/jobs/{job_id}/graph/cypher", root / "graph.cypher")

    save_json(root / "artifacts.json", get_json(client, f"/v1/jobs/{job_id}/artifacts"))
    download(client, f"/v1/jobs/{job_id}/bundle", root / "bundle.zip")
    print(f"  outputs   {root.resolve()}")


def check_event_stream(client: httpx.Client, job_id: str) -> None:
    with client.stream("GET", f"/v1/jobs/{job_id}/events") as response:
        checked(response)
        for line in response.iter_lines():
            if line.startswith("data:"):
                print("  events    stream returned a job event")
                return
    raise SmokeFailure("The event stream produced no job event")


def run(args: argparse.Namespace) -> None:
    with ExitStack() as stack:
        paths = args.files or sample_files(stack)
        for path in paths:
            if not path.is_file():
                raise SmokeFailure(f"Not a file: {path}")

        client = stack.enter_context(
            httpx.Client(base_url=args.base_url.rstrip("/"), timeout=120.0)
        )

        print("health")
        health = get_json(client, "/v1/health")
        print(
            f"  {health['status']} storage={health['storage']} "
            f"job_mode={health['job_mode']} embedders={','.join(health['embedders'])} "
            f"vlm_key={health['vlm_api_key_configured']}"
        )
        capabilities = get_json(client, "/v1/capabilities")
        print(f"  stages    {' -> '.join(capabilities['stages'])}")

        if args.enable_vlm and not (
            args.api_key or health["vlm_api_key_configured"]
        ):
            raise SmokeFailure(
                "--enable-vlm needs --api-key or ANTHROPIC_API_KEY on the service"
            )

        print("upload")
        job_id = submit(client, args, paths)
        print("processing")
        wait_for_completion(client, job_id, args.timeout)
        print("verification")
        check_event_stream(client, job_id)
        collect(client, job_id, args.output_dir)

        if args.delete_job:
            checked(client.delete(f"/v1/jobs/{job_id}"))
            print("  deleted   remote job removed")
        else:
            print(f"  retained  {args.base_url}/v1/jobs/{job_id}")
    print("\nSmoke test passed.")


def main() -> int:
    try:
        run(parse_args())
    except SmokeFailure as exc:
        print(f"\nSmoke test failed: {exc}", file=sys.stderr)
        return 1
    except httpx.HTTPError as exc:
        print(f"\nCould not reach the service: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
