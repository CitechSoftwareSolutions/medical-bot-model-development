"""Text generation backed by the Gemini API."""

from typing import Any


class GeminiGenerator:
    name = "gemini"

    def __init__(self, model_name: str, api_key: str, max_output_tokens: int = 512):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini backend.")
        self.model_name = model_name
        self.api_key = api_key
        self.max_output_tokens = max_output_tokens
        self._client: Any | None = None

    def _load(self) -> Any:
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "You are a medical information assistant. Answer using only the "
            "provided clinical guideline context. If the context does not "
            "support an answer, say so. Do not invent diagnoses, doses, or "
            "treatment recommendations. Advise urgent professional care when "
            "red flags are present.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        response = self._load().models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={"max_output_tokens": self.max_output_tokens},
        )

        text = getattr(response, "text", None)
        if text:
            return text.strip()

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            joined = "".join(getattr(part, "text", "") for part in parts).strip()
            if joined:
                return joined

        raise RuntimeError("Gemini returned an empty response.")
