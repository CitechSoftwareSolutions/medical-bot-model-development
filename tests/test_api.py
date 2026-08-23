"""
API integration tests.

Same caveat as test_pipeline.py: DocumentLoader's default data_dir is bound
at import time to the real data/raw/, so list/upload endpoints operate on
the real 6-document corpus. FAISS index/metadata paths are redirected to a
temp directory so these tests never touch the project's real vector store.
Any file this test uploads into the real data/raw/ is removed in teardown.
"""

import json

import pytest
from fastapi.testclient import TestClient

import src.embedding as embedding_module
import src.pipeline as pipeline_module
import api.routers.search as search_router
from src import config


@pytest.fixture(autouse=True)
def isolated_store(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "FAISS_INDEX_PATH", tmp_path / "index.faiss")
    monkeypatch.setattr(config, "FAISS_METADATA_PATH", tmp_path / "metadata.json")
    monkeypatch.setattr(config, "EMBEDDING_BACKEND", "hashing")
    embedding_module._embedder_instance = None
    pipeline_module._store_instance = None
    yield
    embedding_module._embedder_instance = None
    pipeline_module._store_instance = None


@pytest.fixture
def client():
    from api.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["documents_indexed"] == 6
    assert body["vectors_in_store"] == 43


def test_list_documents(client):
    r = client.get("/documents")
    assert r.status_code == 200
    docs = r.json()
    assert len(docs) == 6
    assert any(d["doc_id"] == "clinical_guidelines_headache" for d in docs)


def test_search_returns_results(client):
    r = client.post("/search", json={"query": "migraine with nausea", "top_k": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["query"] == "migraine with nausea"
    assert len(body["results"]) == 3
    assert all("score" in hit for hit in body["results"])


def test_ask_returns_grounded_answer_and_sources(client, monkeypatch):
    class FakeGenerator:
        def generate(self, question, context):
            assert question == "What helps migraine?"
            assert context
            return "Use the retrieved guideline context."

    monkeypatch.setattr(search_router, "get_generator", lambda: FakeGenerator())
    response = client.post("/ask", json={"question": "What helps migraine?", "top_k": 2})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Use the retrieved guideline context."
    assert len(body["sources"]) == 2
    assert all("condition_name" in source for source in body["sources"])


def test_search_top_k_out_of_range_rejected(client):
    r = client.post("/search", json={"query": "x", "top_k": 999})
    assert r.status_code == 422  # pydantic validation: top_k <= 20


def test_upload_reject_wrong_extension(client):
    files = {"file": ("notes.txt", "not json", "text/plain")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 400


def test_upload_reject_invalid_schema(client):
    bad_doc = {"document_metadata": {"title": "Broken"}}
    files = {"file": ("clinical_guidelines_zzz_test_broken.json", json.dumps(bad_doc), "application/json")}
    r = client.post("/documents/upload", files=files)
    assert r.status_code == 400
    assert not (config.RAW_DATA_DIR / "clinical_guidelines_zzz_test_broken.json").exists()


def test_upload_valid_document_then_duplicate_then_search(client):
    filename = "clinical_guidelines_zzz_test_upload.json"
    dest_path = config.RAW_DATA_DIR / filename
    assert not dest_path.exists(), "test artifact from a previous run was not cleaned up"

    try:
        new_doc = {
            "document_metadata": {"title": "Zzz Test Condition Guide"},
            "conditions_registry": [
                {
                    "condition_name": "Zzz Test Condition",
                    "clinical_presentation": {"key_features": ["Unique zzzalpha marker feature"]},
                    "management_plan": {
                        "initial_conservative": ["Zzz conservative step"],
                        "specific_advanced": ["Zzz advanced step"],
                        "follow_up_monitoring": ["Zzz follow up"],
                    },
                }
            ],
            "global_red_flags": ["Zzz red flag"],
        }

        files = {"file": (filename, json.dumps(new_doc), "application/json")}
        r = client.post("/documents/upload", files=files)
        assert r.status_code == 200
        body = r.json()
        assert body["doc_id"] == "clinical_guidelines_zzz_test_upload"
        assert body["chunks_added"] == 1
        assert dest_path.exists()

        # duplicate filename should be rejected
        r2 = client.post("/documents/upload", files={"file": (filename, json.dumps(new_doc), "application/json")})
        assert r2.status_code == 409

        # newly uploaded content should be immediately searchable
        r3 = client.post("/search", json={"query": "zzzalpha marker feature", "top_k": 1})
        assert r3.status_code == 200
        assert r3.json()["results"][0]["condition_name"] == "Zzz Test Condition"

    finally:
        dest_path.unlink(missing_ok=True)
