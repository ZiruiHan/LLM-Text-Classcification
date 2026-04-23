from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List, Sequence


class OpenAICompatibleChatClassifier:
    """
    Minimal optional adapter for OpenAI-compatible chat completion APIs.

    The benchmark keeps this backend optional. It requires:
    - `OPENAI_API_KEY`
    - optionally `OPENAI_BASE_URL`
    """

    def __init__(self, model_id: str, label_names: Sequence[str]) -> None:
        self.model_id = model_id
        self.label_names = list(label_names)
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the optional zero_shot_llm backend.")

    def predict(self, texts: List[str]) -> List[str]:
        predictions: List[str] = []
        for text in texts:
            predictions.append(self._predict_one(text))
        return predictions

    def _predict_one(self, text: str) -> str:
        prompt = (
            "Classify the text into exactly one label.\n"
            f"Allowed labels: {', '.join(self.label_names)}\n"
            "Reply with only the label.\n"
            f"Text: {text}"
        )
        payload = json.dumps(
            {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = body["choices"][0]["message"]["content"].strip()
        for label in self.label_names:
            if content == label:
                return label
        raise ValueError(f"LLM output did not match a known label: {content}")
