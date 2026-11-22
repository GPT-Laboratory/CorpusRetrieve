# Embeddings

The dense model decides which passages a question finds. Choose it per job in
the web interface or with the `embedder` upload field, set the default with
`DEFAULT_EMBEDDER`, or swap it later with `corpusretrieve reindex`.

| Choice | Kind | Dimension | Needs | Notes |
|---|---|---|---|---|
| `hash` | lexical | 512 | nothing | Default. ASCII word features, kept so existing consumers keep working. |
| `hash-v2` | lexical | 512 | nothing | Unicode word features; better for non-English text. |
| `st:intfloat/multilingual-e5-small` | semantic | 384 | sentence-transformers | Fastest semantic choice on a CPU. |
| `st:intfloat/multilingual-e5-base` | semantic | 768 | sentence-transformers | Better quality, roughly twice the cost of small. |
| `bge-m3` | semantic | 1024 | FlagEmbedding | Long inputs, strongest; practical mainly with a GPU. |
| `st:<model id>` | semantic | model's | sentence-transformers | Any sentence-transformers model. |

Lexical embedders match passages that share words or word fragments with the
question. They cannot match a question to a passage that says the same thing
in other words or another language. Semantic models can.

## Installing a semantic model

```bash
pip install -r requirements-embeddings.txt
```

The model is downloaded from Hugging Face the first time it is used, so the
machine needs internet access once (or a pre-populated Hugging Face cache).
A GPU is used automatically when PyTorch can see one.

## Adding a semantic model to a finished build

Extraction is the slow part of a build; embedding is not. To give an existing
build semantic vectors without processing the documents again:

```bash
corpusretrieve reindex corpusretrieve-<job>.zip --embedder st:intfloat/multilingual-e5-small --out reindexed
corpusretrieve verify reindexed
```

`reindex` extracts the bundle into `--out` (or works on an output directory in
place), re-embeds every unit, rebuilds the knowledge graph (vector point IDs
include the model name), rewrites the catalog and `bundle.zip`, and records the
run in `manifest.repairs`. Time a sample first on CPU-only machines: semantic
models are much slower than the hashing embedders, and BGE-M3 is the slowest.

## Querying

Queries must be embedded with the same model and its query prefix. E5 models
use `passage: ` for indexed units and `query: ` for questions; both are
recorded in `index/metadata.json` (`model_id`, `passage_prefix`,
`query_prefix`). The reference consumer applies them automatically:

```python
from corpusretrieve.consumer import PortableIndex
index = PortableIndex.load("reindexed")
hits = index.search("how did the treatment change durability?", k=10)
```

A consumer that embeds queries itself must load `model_id`, prepend
`query_prefix`, and normalise the vector before the dot product with
`index/dense.npy`.
