"""Text generation backed by the Hugging Face Inference API."""

import requests
import time

class HuggingFaceGenerator:
    name = "huggingface"

    def __init__(self, model_name: str, max_new_tokens: int | None = 512, token: str | None = None, temperature: float = 0.0):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens or 512
        self.token = token
        self.temperature = temperature

    def generate(self, question: str, context: str) -> str:
        if not self.token:
            raise ValueError("HF_TOKEN is required to use the Hugging Face Inference API.")
            
        prompt = (
            "You are a medical information assistant. Answer using only the "
            "provided clinical guideline context. If the context does not "
            "support an answer, say so. Do not invent diagnoses, doses, or "
            "treatment recommendations. Advise urgent professional care when "
            "red flags are present.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        
        api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        headers = {"Authorization": f"Bearer {self.token}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": self.max_new_tokens,
                "temperature": self.temperature if self.temperature > 0 else 0.01,
                "return_full_text": False
            }
        }
        
        # Simple retry loop for loading models
        for attempt in range(3):
            response = requests.post(api_url, headers=headers, json=payload)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and "generated_text" in result[0]:
                    return result[0]["generated_text"].strip()
                return str(result)
            elif response.status_code == 503:
                # Model is currently loading
                time.sleep(10)
                continue
            else:
                raise RuntimeError(f"HF API Error {response.status_code}: {response.text}")
                
        raise RuntimeError("Hugging Face API timed out waiting for the model to load.")

