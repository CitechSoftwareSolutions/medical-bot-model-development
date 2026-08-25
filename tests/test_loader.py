"""
Basic tests for the document loading stage.

Run:
    python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document_loader.loader import DocumentLoader, DocumentLoadError
from src.document_loader.schema import Document


def test_loads_all_real_guideline_files():
    loader = DocumentLoader()
    documents = loader.load_all()

    assert len(documents) == 6
    assert all(isinstance(d, Document) for d in documents)
    assert all(d.n_conditions > 0 for d in documents)


def test_document_has_expected_fields():
    loader = DocumentLoader()
    documents = loader.load_all()
    doc = next(d for d in documents if d.doc_id == "clinical_guidelines_headache")

    assert doc.title == "Evaluation of Headache in Adults: A Clinical Flow Chart Guide"
    assert doc.category == "Clinical Guidelines"
    assert len(doc.conditions_registry) == 9
    assert "Migraines" in [c["condition_name"] for c in doc.conditions_registry]


def test_rejects_malformed_json(tmp_path):
    bad_file = tmp_path / "bad.json"
    bad_file.write_text('{ "document_metadata": {} ')  # missing closing brace

    loader = DocumentLoader(data_dir=tmp_path)
    try:
        loader.load_one(bad_file)
        assert False, "Expected DocumentLoadError"
    except DocumentLoadError:
        pass


def test_rejects_missing_required_keys(tmp_path):
    bad_file = tmp_path / "missing_keys.json"
    bad_file.write_text(json.dumps({"document_metadata": {"title": "x"}}))

    loader = DocumentLoader(data_dir=tmp_path)
    try:
        loader.load_one(bad_file)
        assert False, "Expected DocumentLoadError"
    except DocumentLoadError:
        pass


def test_load_all_skips_bad_files_but_keeps_good_ones(tmp_path):
    good = {
        "document_metadata": {"title": "Good Doc"},
        "conditions_registry": [{"condition_name": "X"}],
    }
    (tmp_path / "good.json").write_text(json.dumps(good))
    (tmp_path / "bad.json").write_text("{not valid json")

    loader = DocumentLoader(data_dir=tmp_path)
    documents = loader.load_all()

    assert len(documents) == 1
    assert documents[0].title == "Good Doc"
