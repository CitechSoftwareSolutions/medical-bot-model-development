# Changelog

## This delivery: Upload API + preprocessing + extraction + embedding + FAISS API

### ⚠️ Changes to the document LOADING part (as requested, flagged explicitly)

**File: `src/document_loader/loader.py`**

One addition, no removals. Existing behavior for `load_all()` and `load_one()`
is unchanged — all 5 original tests in `tests/test_loader.py` still pass
unmodified against the new code.

- **Added** `DocumentLoader.load_from_dict(data, doc_id, source_path)` — validates
  and wraps an already-parsed dict into a `Document`, without touching disk.
- **Refactored** `load_one()` and `load_from_dict()` to share one internal
  `_build_document()` method, instead of `load_one()` duplicating the
  validation/wrapping logic inline.
- **Why**: the upload API needs to validate an uploaded file's contents
  *before* deciding whether to write it to `data/raw/`. Without this, the
  only way to validate would have been to write the file first and delete it
  on failure — meaning a bad upload would briefly touch disk. Now it never
  does.
- **Renamed internal parameter**: `_validate_schema(data, path)` →
  `_validate_schema(data, context_name)`, and `_parse_json(raw_text, path)` →
  `_parse_json(raw_text, context_name)` — both now take a plain string for
  logging instead of requiring a `Path` object, since `load_from_dict()`
  doesn't have a real file path to give it. Purely internal; not part of
  the public interface.

**No changes** to `src/document_loader/schema.py`, `src/config.py`'s existing
settings, or `validate_new_document.py` (still works exactly as before).

### New: preprocessing stage — `src/preprocessing/`

Normalizes the two different `management_plan` shapes found across your
files (`initial_conservative/specific_advanced/follow_up_monitoring` vs.
`pharmacological/non_pharmacological`) into one consistent structure, plus
text cleanup (whitespace collapsing). An unrecognized future shape doesn't
crash — it's flattened into the "advanced" bucket with a logged warning.

### New: extraction stage — `src/extraction/`

Builds one `Chunk` per condition (43 chunks across your 6 files) — this is
the retrieval unit: everything needed to answer "how do I manage X" in one
piece. Each chunk carries a stable `chunk_id`, the assembled text that gets
embedded, and metadata (source doc, red flags, investigations, etc.) kept
alongside for display.

### New: embedding stage — `src/embedding/`

Interface (`BaseEmbedder`) + two backends:
- `SentenceTransformerEmbedder` — real semantic embeddings (needs internet
  on first run to download the model).
- `HashingEmbedder` — deterministic, fully offline fallback, used
  automatically if the real model can't be loaded, and used by the test
  suite so tests never depend on network access.

`get_embedder()` factory tries the real model first (`EMBEDDING_BACKEND =
"auto"` in `src/config.py`) and falls back automatically, logging a clear
warning when it does.

### New: vector store — `src/vector_store/faiss_store.py`

FAISS `IndexFlatIP` (cosine similarity via normalized vectors) with a JSON
metadata sidecar, save/load persistence, and a `rebuild`/`clear` path for
testing.

### New: pipeline orchestrator — `src/pipeline.py`

Ties every stage together. `rebuild_index()` rebuilds from scratch;
`ingest_document()` adds one document incrementally without a full rebuild
— this is what the upload endpoint calls.

### New: API — `api/`

- `POST /documents/upload` — upload a new guideline JSON file. Validates
  before writing to disk; rejects wrong extension, oversized files, invalid
  JSON, schema violations, and duplicate filenames. On any ingestion
  failure after the file is written, the file write is rolled back.
- `GET /documents` — list everything currently in `data/raw/`.
- `POST /search` — embed a query and search the FAISS store.
- `POST /vector-store/rebuild` — wipe and rebuild the whole index from
  `data/raw/` (for testing, or after manually editing files on disk).
- `GET /health` — reports which embedding backend is actually active,
  vector/document counts.

Run with: `uvicorn api.main:app --reload --port 8000`, then see `/docs`.

### Bug found and fixed during this delivery

`src/pipeline.py`: `rebuild_index()` originally used
`store = store or get_vector_store()`. `FAISSVectorStore` defines `__len__`,
so a freshly created but still-empty store is *falsy* in Python — meaning
`or` incorrectly treated a valid empty store as "missing" and called
`get_vector_store()` again, which called back into `rebuild_index()`,
infinitely. Fixed with an explicit `if store is None:` check. Caught by
actually running the full pipeline end-to-end before shipping, not by
inspection.

### Test coverage added

`tests/test_preprocessor.py`, `test_extractor.py`, `test_embedding.py`,
`test_vector_store.py`, `test_pipeline.py`, `test_api.py` — 35 tests total,
all passing. `test_loader.py` (the 5 original tests) unmodified and still
green.
