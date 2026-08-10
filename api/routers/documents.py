"""
Document endpoints: upload a new guideline file, list what's currently loaded.
"""

import json
import logging

from fastapi import APIRouter, HTTPException, UploadFile

from src import config
from src.document_loader.loader import DocumentLoader, DocumentLoadError
from src.pipeline import ingest_document
from api.schemas import DocumentSummary, UploadResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile):
    """
    Upload a new guideline JSON file.

    Flow: validate in memory -> reject before touching disk if invalid ->
    write to data/raw/ -> extract chunks -> embed -> add to the FAISS store
    -> persist the store. All-or-nothing: if validation fails, nothing is
    written and nothing is indexed.
    """
    filename = file.filename or "upload.json"
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if suffix not in config.ALLOWED_UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(config.ALLOWED_UPLOAD_EXTENSIONS)}",
        )

    raw_bytes = await file.read()
    size_mb = len(raw_bytes) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({size_mb:.1f} MB). Max is {config.MAX_UPLOAD_SIZE_MB} MB.",
        )

    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail=f"File is not valid UTF-8: {e}")

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON at line {e.lineno}: {e.msg}")

    doc_id = filename.rsplit(".", 1)[0]
    dest_path = config.RAW_DATA_DIR / f"{doc_id}.json"

    loader = DocumentLoader()
    try:
        # Validated purely from the in-memory dict — nothing touches disk yet.
        document = loader.load_from_dict(data, doc_id=doc_id, source_path=str(dest_path))
    except DocumentLoadError as e:
        raise HTTPException(status_code=400, detail=f"Document failed validation: {e}")

    if dest_path.exists():
        raise HTTPException(
            status_code=409,
            detail=f"A document named {dest_path.name!r} already exists. Rename the file or delete the existing one first.",
        )

    config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest_path.write_text(text, encoding="utf-8")
    logger.info("Wrote uploaded document to %s", dest_path)

    try:
        chunks_added = ingest_document(document)
    except Exception:
        # Roll back the file write so a failed ingest doesn't leave an
        # unindexed file sitting in data/raw/ that a future load_all() would
        # pick up inconsistently with the vector store's contents.
        dest_path.unlink(missing_ok=True)
        logger.exception("Ingestion failed for %s — rolled back file write", dest_path)
        raise HTTPException(status_code=500, detail="Ingestion failed; upload was rolled back.")

    from src.pipeline import get_vector_store
    store = get_vector_store()

    return UploadResponse(
        doc_id=document.doc_id,
        title=document.title,
        conditions_found=document.n_conditions,
        chunks_added=chunks_added,
        total_vectors_in_store=len(store),
    )


@router.get("", response_model=list[DocumentSummary])
def list_documents():
    """List every document currently in data/raw/, with basic stats."""
    documents = DocumentLoader().load_all()
    return [
        DocumentSummary(
            doc_id=doc.doc_id,
            title=doc.title,
            category=doc.category,
            conditions_count=doc.n_conditions,
            global_red_flags_count=len(doc.global_red_flags),
        )
        for doc in documents
    ]
