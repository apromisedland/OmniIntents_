"""Chat Completions adapter with explicit models and bounded HTTP retries."""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from ..errors import ProviderError, ValidationError
from ..types import finite_number
from .base import LLMRequest, LLMResult


class OpenAICompatibleClient:
    backend = "openai-compatible"

    def __init__(
        self, *, model: str | None = None, api_key: str | None = None,
        base_url: str | None = None, vision_model: str | None = None,
        timeout_s: float = 60, retries: int = 2, token_parameter: str = "max_tokens",
        send_seed: bool = False, session: Any = None,
    ):
        self.model = model or os.getenv("OPENAI_MODEL", "")
        self.vision_model = vision_model or os.getenv("OPENAI_VISION_MODEL") or self.model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        if not isinstance(self.model, str) or not self.model.strip() or not isinstance(self.api_key, str) or not self.api_key.strip():
            raise ValidationError("Explicit model and API credential are required for the live backend")
        if not self.base_url.startswith(("https://", "http://")):
            raise ValidationError("base_url must be an HTTP(S) URL")
        if self.base_url.endswith("/chat/completions"):
            raise ValidationError("base_url must end at the API root, not chat/completions")
        finite_number(timeout_s, "timeout_s", 0)
        if timeout_s <= 0 or type(retries) is not int or not 0 <= retries <= 5:
            raise ValidationError("Invalid timeout or retry limit")
        if token_parameter not in {"max_tokens", "max_completion_tokens"}:
            raise ValidationError("Unsupported completion token parameter")
        self.timeout_s = timeout_s
        self.retries = retries
        self.token_parameter = token_parameter
        self.send_seed = send_seed
        self.session = session or requests.Session()

    def _image_content(self, filename: str) -> dict[str, Any]:
        path = Path(filename)
        try:
            with Image.open(path) as image:
                mime = Image.MIME.get(image.format)
                image.verify()
            if mime not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
                raise ValidationError("Unsupported image MIME type")
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        except (OSError, ValueError) as error:
            raise ValidationError("Cannot read image for the vision request") from error
        return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}}

    def generate(self, request: LLMRequest) -> LLMResult:
        content: list[dict[str, Any]] = [{
            "type": "text", "text": json.dumps(request.payload, ensure_ascii=False, allow_nan=False),
        }]
        content.extend(self._image_content(path) for path in request.image_paths)
        model = self.vision_model if request.image_paths else self.model
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": request.system},
                         {"role": "user", "content": content}],
            "temperature": request.temperature,
            self.token_parameter: request.max_tokens,
            "response_format": {"type": "json_object"},
        }
        if self.send_seed:
            payload["seed"] = request.seed
        endpoint = self.base_url
        if endpoint == "https://api.openai.com":
            endpoint += "/v1"
        endpoint += "/chat/completions"
        for attempt in range(self.retries + 1):
            retryable = False
            try:
                response = self.session.post(
                    endpoint, headers={"Authorization": f"Bearer {self.api_key}"},
                    json=payload, timeout=(min(10, self.timeout_s), self.timeout_s),
                )
                status = response.status_code
                if status == 429 or status >= 500:
                    retryable = True
                    code = "rate_limit" if status == 429 else "service_error"
                elif status >= 400:
                    code = "authentication" if status in {401, 403} else "request_rejected"
                    raise ProviderError(code, f"Provider rejected request (HTTP {status})")
                else:
                    try:
                        data = response.json()
                        choice = data["choices"][0]
                        text = choice["message"]["content"]
                        if not isinstance(text, str) or not text.strip():
                            raise ValueError("empty response")
                        if choice.get("finish_reason") == "length":
                            raise ProviderError("truncated_response", "Provider exhausted the output token limit")
                        usage = data.get("usage", {})
                        if not isinstance(usage, dict) or not isinstance(data.get("model", model), str):
                            raise ValueError("invalid metadata")
                        metadata = {
                            "backend": self.backend, "model": data.get("model", model),
                            "requested_model": model, "attempts": attempt + 1,
                            "usage": {key: usage[key] for key in (
                                "prompt_tokens", "completion_tokens", "total_tokens",
                            ) if key in usage},
                            "system_fingerprint": data.get("system_fingerprint"),
                            "seed_sent": self.send_seed,
                        }
                        return LLMResult(text, metadata)
                    except (KeyError, IndexError, TypeError, ValueError) as error:
                        raise ProviderError("invalid_response", "Provider returned an invalid completion") from error
            except requests.Timeout:
                retryable, code = True, "timeout"
            except requests.ConnectionError:
                retryable, code = True, "connection"
            except requests.RequestException as error:
                raise ProviderError("transport", "Provider transport failed") from error
            if retryable and attempt < self.retries:
                time.sleep(min(2.0, 0.25 * 2 ** attempt))
                continue
            raise ProviderError(code, f"Provider request failed after {attempt + 1} attempt(s)")
        raise ProviderError("service_error", "Provider request failed")
