# Generic manifest and group-based processing

The code has no built-in groups, sector keywords, collections, or project IDs.
The optional converter accepts an **external** nested manifest shaped as:

```json
{
  "project_name": "example_collection",
  "run_id": "collection-run-001",
  "scenarios": [{
    "id": "catalogue",
    "themes": [{
      "theme_id": "group_a",
      "theme_name": "Group A",
      "documents": [{
        "doc_id": "source-001",
        "title": "A General Research Study",
        "authors": ["A. Author"],
        "year": 2025,
        "file_path": "files/group_a/doc001.pdf",
        "online_url": "https://example.org/publication/001"
      }]
    }]
  }]
}
```

The nested schema is a supported *import format*, not an application data
model or a requirement that clients use themes. User-supplied classification
stays in external JSON and output metadata; no theme names are hardcoded.
The manifest does **not** contain PDF bytes: files must exist under a
corresponding local root before extraction.

## Create metadata and upload batches

```bash
python tools/prepare_theme_jobs.py input_manifest.json --out group_jobs --max-files 150
python tools/prepare_theme_jobs.py input_manifest.json --out group_jobs \
  --files-root /path/to/source-root --make-zips --max-files 150 --max-zip-mb 80
```

For each group ID, the tool writes `full_theme.path_config.json`,
`part-NNN.path_config.json`, `part-NNN.direct_config.json` (when filenames
are unique), `part-NNN.files.csv`, and a report. ZIP batches require matching
`part-NNN.path_config.json`. Direct browser uploads require the corresponding
`*.direct_config.json`. An absent source document must not silently vanish;
check the missing-file report.

## One unified index per group

```bash
corpusretrieve build /path/to/source-root \
  --config group_jobs/group_a/full_theme.path_config.json \
  --key-by path --only-config-files --require-bibliographic \
  --corpus-name group_a --out outputs/group_a
```

This produces **one** index for the entire group and reuses unchanged document
extractions when rerun in that output directory. HTTP upload jobs, including
batch ZIPs, each produce their **own independent bundle**. The API does not
implicitly merge those indexes, so a downstream client must either load
multiple bundles or use the single-group CLI build.

Documents in multiple groups keep source identities and group-membership
metadata, but independently built output directories do not share an extraction
cache. The chosen source-record IDs are not derived from group names.

For exact UI upload structure and strict bibliography fields see
[Document configuration](DOCUMENT_CONFIG.md).
