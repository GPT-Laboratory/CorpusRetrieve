# Production acceptance checklist — CorpusRetrieve 2.0.0

This is a **checklist**, not a certification of production readiness.

- [ ] Pin an immutable repository commit and Docker image digest.
- [ ] Deploy with `DEPLOYMENT_MODE=production`, `STORAGE_BACKEND=s3`, `JOB_MODE=rq`.
- [ ] Configure non-default S3 access credentials, Redis connectivity, a valid
      Fernet key, TLS termination, API keys and access control for all users/jobs.
- [ ] Restrict exposed ports to a reverse proxy; no public MinIO administrative port.
- [ ] Verify the proxy provides compatible authenticated access to the web UI;
      the bundled browser UI does **not** currently prompt for a service API key.
      Alternatively disable the console and use authorized API clients.
- [ ] Set per-job upload, expanded archive, file-count, worker concurrency,
      CPU, memory, disk and VLM request/cost limits.
- [ ] Install OCR language packs actually required by the input corpus, then
      set `OCR_LANGS` accordingly. Default image includes English only.
- [ ] Use a trained multilingual embedder for relevance-sensitive production
      retrieval; `hash` is an offline testing baseline.
- [ ] Check `manifest.release_ready`, `manifest.quality.reasons`, bibliographic
      resolution and content-loss warnings; require release readiness at import.
- [ ] Manually inspect reference-heavy pages, multi-column scientific text,
      multi-page tables, figures, author lists and multi-publication PDFs.
- [ ] Run representative corpus regression tests and test citation accuracy
      within the *actual consumer*, including multilingual queries and refusals.
- [ ] Test interrupt/retry/rebuild, artifact retention, object-store recovery,
      authentication, rate limits and cross-tenant job access.
- [ ] Monitor API errors, worker failures, storage space, queue length, VLM
      costs, job duration, schema compatibility and checksum verification.

CLI quick checks:

```bash
python -m pytest -q
corpusretrieve verify /path/to/output/bundle.zip --require-release-ready
```

Production mode rejects missing API keys, insecure storage/job backends,
default S3 secrets, and wildcard CORS. It does **not** replace authentication
and authorization infrastructure, a security audit, or load testing.
