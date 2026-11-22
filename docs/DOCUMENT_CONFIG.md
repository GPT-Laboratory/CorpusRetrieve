# Document metadata and bibliographic provenance

CorpusRetrieve is domain-neutral. Its **per-document configuration** carries the
identity and bibliographic metadata of the input documents; `metadata` in the upload
form carries *whole-corpus* context. A metadata JSON file is not the source PDF.

## JSON format

Direct upload of files named `study.pdf` and `report.pdf`:

```json
{
  "study.pdf": {
    "document_id": "repository-record-001",
    "metadata": {
      "source": "Example institutional repository",
      "language": "en",
      "records": [{
        "title": "Comparative Study of a Measurement Method",
        "authors": ["A. Researcher", "B. Analyst"],
        "year": 2025,
        "doi": "10.1234/example.0001",
        "online_url": "https://example.org/record/001",
        "origin": "manifest"
      }]
    }
  },
  "report.pdf": {
    "document_id": "repository-record-002",
    "metadata": {
      "records": [{
        "title": "Technical Report on System Measurements",
        "year": 2024,
        "handle_url": "https://example.org/record/002",
        "origin": "manifest"
      }]
    }
  }
}
```

Replace values with real metadata. The `doi` in this example is illustrative,
not a verified identifier. Unknown or absent fields must be omitted rather than invented.
The `records` list allows multiple publications in one file. Use optional
`page_start` and `page_end` when an archive combines several distinct papers.
When records cannot be resolved uniquely, strict mode rejects release readiness.

For an uploaded ZIP, key each entry by its **ZIP member path** (e.g.
`folder/study.pdf`); for direct PDF uploads, key by exact **filename**.
Filename matching is exact and case-sensitive. Preview available for direct
files; ZIP member matching is validated on the server.

## Provenance priority

1. Authoritative supplied metadata (`metadata.records`), including stable
   identifiers, titles, authors, year, DOI when known, source links and ranges.
2. With no supplied record, a *PDF-extracted* DOI/title candidate, marked
   `origin: "extracted"`. This is not registry-verified and may be incomplete.
3. If neither resolves, the `bibliographic` field is null and strict release
   readiness fails.

The generated payload also preserves independent location information:
`doc_id`, pages, bounding boxes, source element IDs, spans and derived-text flags.
The underlying reference list *inside* the document is extracted content, not
`metadata.records`.

## Strict mode

- Web upload: check **Require resolvable publication metadata**.
- API: send `require_bibliographic=true` in multipart data.
- CLI: `corpusretrieve build ... --require-bibliographic`.
- Environment default: `REQUIRE_BIBLIOGRAPHIC=true`.

Strictness validates existence/resolvability, not the correctness of titles,
DOIs, PDFs, or underlying scientific claims. Review sample records and source
regions before trusting high-stakes answers. Use `corpusretrieve verify` and
inspect `manifest.json` for the gate result.

## Corpus-level metadata

`metadata` (upload form) applies to the whole run, such as
`{"project": "example_project", "group": "batch_A"}`. The optional
`__collection__` entry in `document_config` carries input-source run metadata
and exclusion records into `manifest.collection`. Neither substitutes for
per-document bibliographic records.
