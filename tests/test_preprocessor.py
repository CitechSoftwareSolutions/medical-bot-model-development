from src.preprocessing.preprocessor import (
    clean_text,
    normalize_management_plan,
    preprocess_document,
)
from src.document_loader.loader import DocumentLoader


def test_clean_text_collapses_whitespace():
    assert clean_text("  a   b\n c  ") == "a b c"


def test_normalize_shape_a_initial_conservative():
    condition = {
        "management_plan": {
            "initial_conservative": ["Rest"],
            "specific_advanced": ["Surgery"],
            "follow_up_monitoring": ["Review in 2 weeks"],
        }
    }
    result = normalize_management_plan(condition, context_name="test")
    assert result == {
        "conservative": ["Rest"],
        "advanced": ["Surgery"],
        "follow_up": ["Review in 2 weeks"],
    }


def test_normalize_shape_b_pharmacological():
    condition = {
        "management_plan": {
            "pharmacological": {
                "acute_abortive": ["Ibuprofen"],
                "preventive": ["Propranolol"],
            },
            "non_pharmacological": ["Hydration"],
        }
    }
    result = normalize_management_plan(condition, context_name="test")
    assert result["conservative"] == ["Hydration"]
    assert set(result["advanced"]) == {"Ibuprofen", "Propranolol"}
    assert result["follow_up"] == []


def test_normalize_unknown_shape_flattens_instead_of_crashing():
    condition = {
        "condition_name": "Something new",
        "management_plan": {"some_new_bucket": {"nested": ["Do X", "Do Y"]}},
    }
    result = normalize_management_plan(condition, context_name="test")
    assert result["conservative"] == []
    assert set(result["advanced"]) == {"Do X", "Do Y"}


def test_preprocess_document_on_real_headache_file():
    loader = DocumentLoader()
    headache = next(d for d in loader.load_all() if d.doc_id == "clinical_guidelines_headache")
    conditions = preprocess_document(headache)

    migraine = next(c for c in conditions if c.condition_name == "Migraines")
    assert "Trigger avoidance" in migraine.management_conservative
    assert any("Sumatriptan" in item for item in migraine.management_advanced)
