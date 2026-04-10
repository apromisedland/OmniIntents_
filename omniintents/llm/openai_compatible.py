from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests

from .base import LLMResult


@dataclass
class OpenAICompatibleClient:
    """An *optional* adapter for an OpenAI-compatible Chat Completions API.

    This file is provided as a convenience scaffold. You may need to adapt it
    to your exact provider/model.

    Environment variables supported:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (default: https://api.openai.com)
    - OPENAI_MODEL (default: gpt-4o-mini or any you use)
    - OPENAI_VISION_MODEL (optional; default uses OPENAI_MODEL)

    Note: This project does **not** hard-depend on this client.
    """

    api_key: Optional[str] = None
    base_url: str = "https://api.openai.com"
    model: str = "gpt-4o-mini"
    vision_model: Optional[str] = None
    timeout_s: int = 60

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("OPENAI_BASE_URL", self.base_url).rstrip("/")
        self.model = os.getenv("OPENAI_MODEL", self.model)
        self.vision_model = os.getenv("OPENAI_VISION_MODEL", self.vision_model) or self.model

        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 800) -> LLMResult:
        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
        }
        r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResult(text=text, raw=data)

    def vision_describe(self, image_path: str, prompt: str, *, max_tokens: int = 300) -> LLMResult:
        # Encode image as data URL (common multimodal format for OpenAI-compatible APIs).
        with open(image_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        data_url = f"data:image/png;base64,{b64}"

        url = f"{self.base_url}/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0.2,
            "max_tokens": int(max_tokens),
        }
        r = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        data = r.json()
        text = data["choices"][0]["message"]["content"]
        return LLMResult(text=text, raw=data)
