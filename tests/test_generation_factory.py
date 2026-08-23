import pytest

import src.generation as generation_module
from src import config


@pytest.fixture(autouse=True)
def reset_generator_singleton():
    generation_module._generator_instance = None
    yield
    generation_module._generator_instance = None


def test_get_generator_uses_gemini_backend(monkeypatch):
    class FakeGeminiGenerator:
        name = "gemini"

        def __init__(self, model_name, api_key, max_output_tokens):
            self.model_name = model_name
            self.api_key = api_key
            self.max_output_tokens = max_output_tokens

    monkeypatch.setattr(config, "LLM_BACKEND", "gemini")
    monkeypatch.setattr(config, "LLM_MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setattr(config, "LLM_MAX_NEW_TOKENS", 256)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "test-key")

    import src.generation.gemini_generator as gemini_module

    monkeypatch.setattr(gemini_module, "GeminiGenerator", FakeGeminiGenerator)

    generator = generation_module.get_generator()

    assert generator.name == "gemini"
    assert generator.model_name == "gemini-2.5-flash"
    assert generator.api_key == "test-key"
    assert generator.max_output_tokens == 256


def test_get_generator_gemini_missing_key_raises(monkeypatch):
    monkeypatch.setattr(config, "LLM_BACKEND", "gemini")
    monkeypatch.setattr(config, "LLM_MODEL_NAME", "gemini-2.5-flash")
    monkeypatch.setattr(config, "LLM_MAX_NEW_TOKENS", 256)
    monkeypatch.setattr(config, "GEMINI_API_KEY", None)

    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        generation_module.get_generator()
