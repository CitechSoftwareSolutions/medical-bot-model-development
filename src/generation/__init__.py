"""Lazy generation backend factory for retrieval-augmented answers."""

from typing import Any

_generator_instance: Any | None = None


def get_generator() -> Any:
    """Return the process-wide generator, loading the backend only when needed."""
    global _generator_instance
    if _generator_instance is None:
        from src import config

        if config.LLM_BACKEND == "gemini":
            from src.generation.gemini_generator import GeminiGenerator

            _generator_instance = GeminiGenerator(
                model_name=config.LLM_MODEL_NAME,
                api_key=config.GEMINI_API_KEY or "",
                max_output_tokens=config.LLM_MAX_NEW_TOKENS,
            )
        elif config.LLM_BACKEND == "huggingface":
            from src.generation.huggingface_generator import HuggingFaceGenerator

            _generator_instance = HuggingFaceGenerator(
                model_name=config.LLM_MODEL_NAME,
                max_new_tokens=config.LLM_MAX_NEW_TOKENS,
                token=config.HF_TOKEN,
            )
        else:
            raise ValueError(f"Unknown LLM_BACKEND: {config.LLM_BACKEND!r}")
    return _generator_instance
