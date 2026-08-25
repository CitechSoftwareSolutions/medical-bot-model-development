"""
These tests exercise the pipeline against the REAL files in data/raw/
(DocumentLoader's default directory is bound at import time, so it always
points at the real corpus regardless of monkeypatching config here) — but
use a temp directory for the FAISS index/metadata files and force the
hashing embedder, so tests are fast, deterministic, and don't touch the
project's real data/vector_store/ files.
"""

import pytest

import src.embedding as embedding_module
import src.pipeline as pipeline_module
from src import config
from src.pipeline import rebuild_index, ingest_document
from src.document_loader.loader import DocumentLoader


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(config, "FAISS_METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")

    # Reset singletons so each test gets a fresh embedder/store bound to
    # the tmp_path config above, instead of reusing another test's instance.
    embedding_module._embedder_instance = None
    pipeline_module._store_instance = None
    yield
    embedding_module._embedder_instance = None
    pipeline_module._store_instance = None


def test_rebuild_index_matches_real_corpus():
    result = rebuild_index()
    assert result.documents_loaded == 6
    assert result.chunks_added == 43
    assert result.total_vectors_in_store == 43


def test_rebuild_index_is_idempotent():
    first = rebuild_index()
    second = rebuild_index()
    assert first.total_vectors_in_store == second.total_vectors_in_store


def test_ingest_document_adds_to_existing_store():
    rebuild_index()  # start from the real 6-document baseline
    store = pipeline_module.get_vector_store()
    before = len(store)

    extra_doc = DocumentLoader().load_from_dict(
        {
            "document_metadata": {"title": "Extra test doc"},
            "conditions_registry": [
                {"condition_name": "Test condition", "clinical_presentation": {"key_features": ["Feature A"]}}
            ],
        },
        doc_id="test_extra_doc",
        source_path="test_extra_doc.json",
    )
    added = ingest_document(extra_doc)

    assert added == 1
    assert len(store) == before + 1
