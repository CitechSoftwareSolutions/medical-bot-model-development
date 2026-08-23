"""Text generation backed by a Hugging Face causal language model."""

from typing import Any


class HuggingFaceGenerator:
    name = "huggingface"

    def __init__(self, model_name: str, max_new_tokens: int = 512, token: str | None = None):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.token = token
        self._pipeline: Any | None = None

    def _load(self) -> Any:
        if self._pipeline is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                token=self.token,
                trust_remote_code=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                torch_dtype="auto",
                token=self.token,
                trust_remote_code=True,
            )
            self._pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
            )
        return self._pipeline

    def generate(self, question: str, context: str) -> str:
        prompt = (
            "You are a medical information assistant. Answer using only the "
            "provided clinical guideline context. If the context does not "
            "support an answer, say so. Do not invent diagnoses, doses, or "
            "treatment recommendations. Advise urgent professional care when "
            "red flags are present.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        result = self._load()(prompt, max_new_tokens=self.max_new_tokens, do_sample=False, return_full_text=False)
        return result[0]["generated_text"].strip()
