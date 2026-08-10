# Medical Guiding System — Sprint 1: Document Handling

Pipeline: **Loading → Preprocessing → Extraction → Embedding → Vector Store (FAISS)**,
plus an API to upload documents and search the store.

See `CHANGELOG.md` for exactly what changed in this delivery, including the
one (backward-compatible) change to the document loading code.

## Quick start

```bash
pip install -r requirements.txt

# Run the API
uvicorn api.main:app --reload --port 8000
# then open http://localhost:8000/docs for interactive Swagger docs

# Or run the pipeline directly, no API
python main.py                    # just loading, prints a summary
python -m pytest tests/ -v        # full test suite (35 tests)
```

On first run, the API builds the FAISS index from whatever is in
`data/raw/` (currently your 6 guideline files, 43 conditions total) and
saves it to `data/vector_store/`. Later runs load the saved index instead
of rebuilding, unless you call `POST /vector-store/rebuild`.

## Architecture

```
src/
  config.py                 Central paths & settings for every stage
  document_loader/          STAGE 1: read raw JSON, validate, wrap as Document
  preprocessing/             STAGE 2: normalize management_plan shapes, clean text
  extraction/                STAGE 3: build one Chunk per condition (43 total)
  embedding/                 STAGE 4: text -> vector (real model or offline fallback)
  vector_store/              STAGE 5: FAISS index + metadata, save/load
  pipeline.py                Orchestrates all 5 stages

api/
  main.py                    FastAPI app
  routers/documents.py       POST /documents/upload, GET /documents
  routers/search.py          POST /search, POST /vector-store/rebuild, GET /health
  schemas.py                  Request/response models
```

## API endpoints

| Method | Path                     | What it does |
|--------|--------------------------|---------------|
| POST   | `/documents/upload`      | Upload a new guideline `.json`. Validates before writing to disk; embeds and indexes it immediately. |
| GET    | `/documents`              | List everything currently in `data/raw/`. |
| POST   | `/search`                 | `{"query": "...", "top_k": 5}` → ranked chunks from the FAISS store. |
| POST   | `/vector-store/rebuild`  | Wipe and rebuild the whole index from `data/raw/`. |
| GET    | `/health`                 | Which embedding backend is active, document/vector counts. |

Example:
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@clinical_guidelines_new_condition.json"

curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query": "throbbing headache with nausea", "top_k": 3}'
```

## Adding a new guideline document

**Via the API (recommended — validates, indexes, and searchable immediately):**
```bash
curl -X POST http://localhost:8000/documents/upload -F "file=@your_file.json"
```

**Manually (drop into `data/raw/`, then rebuild):**
```bash
python validate_new_document.py path/to/new_file.json   # check it first
cp path/to/new_file.json data/raw/
curl -X POST http://localhost:8000/vector-store/rebuild
```

Either way, the only requirement is the schema confirmed earlier:
`document_metadata` + a non-empty `conditions_registry`. Everything else
(including which `management_plan` shape it uses) is handled automatically.

## Embedding backend — read this before deploying

`src/config.py` has `EMBEDDING_BACKEND = "auto"`, which tries to load a real
`sentence-transformers` model (`all-MiniLM-L6-v2` by default) and **falls
back automatically** to a deterministic offline `HashingEmbedder` if the
model can't be downloaded (no internet, firewall, etc.).

**This fallback is real and was actually triggered while building this**,
because the sandbox this was built in has no access to `huggingface.co`.
The hashing embedder is fine for verifying the pipeline's plumbing (upload
→ chunk → embed → store → search all work end-to-end), but it has **no
real semantic understanding** — it's keyword-overlap-ish, not meaning-based.
For actual retrieval quality, you need the real model:

1. Make sure your machine has normal internet access.
2. First run of the API (or `rebuild_index()`) will download the model
   automatically and cache it — no code change needed.
3. Check which backend is actually active any time via `GET /health` →
   `"embedding_backend"` will say `"sentence-transformers"` once it's
   working, `"hashing"` if it's still falling back.
4. For better clinical-domain results specifically, consider swapping
   `EMBEDDING_MODEL_NAME` in `src/config.py` to a biomedical model (e.g.
   `pritamdeka/S-PubMedBert-MS-MARCO`) — test it in your environment first.

## Chunking design

One chunk per `condition_name` (43 total across your 6 files) — not per
document, not per sentence. Rationale: a medical student asking "how do I
manage migraines" should get back one complete, coherent answer, not
fragments split mid-management-plan. Each chunk's embedded text is:
condition name → key features → investigations → management (conservative,
then advanced) → follow-up → differential red flags, in that order, so the
most search-relevant content (name, symptoms) leads.

## Next steps (not in this delivery)

- Frontend / chat interface consuming `/search`.
- Reranking or hybrid (keyword + vector) search if pure vector search proves
  too coarse once real content grows.
- Auth on the upload endpoint before this is exposed beyond local testing.
- If corpus size grows into the hundreds of thousands of chunks, revisit
  `IndexFlatIP` (exact search) for an approximate index (e.g. HNSW).
