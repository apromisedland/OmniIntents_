"""Stage-aware model interface; request routing never parses prompt markers."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..config import OmniIntentsConfig
from ..errors import OmniIntentsError
from ..types import PipelineTrace
from ..utils.json_utils import require_json_object


@dataclass
class LLMRequest:
    stage: str
    system: str
    payload: dict[str, Any]
    image_paths: list[str] = field(default_factory=list)
    temperature: float = 0.0
    max_tokens: int = 1600
    seed: int = 0


@dataclass
class LLMResult:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMClient(Protocol):
    backend: str

    def generate(self, request: LLMRequest) -> LLMResult:
        ...


class RecordingClient:
    def __init__(self, client: LLMClient):
        self.client = client
        self.backend = client.backend
        self.calls: list[dict[str, Any]] = []

    def generate(self, request: LLMRequest) -> LLMResult:
        start = time.perf_counter()
        payload_hash = hashlib.sha256(json.dumps(
            {"system": request.system, "payload": request.payload},
            sort_keys=True, ensure_ascii=False, allow_nan=False,
        ).encode()).hexdigest()
        record = {
            "stage": request.stage, "backend": self.backend, "request_sha256": payload_hash,
            "temperature": request.temperature, "max_tokens": request.max_tokens, "seed": request.seed,
            "image_count": len(request.image_paths),
        }
        try:
            try:
                record["image_sha256"] = [hashlib.sha256(Path(path).read_bytes()).hexdigest()
                                         for path in request.image_paths]
            except OSError as error:
                from ..errors import ValidationError

                raise ValidationError("Unreadable model image input") from error
            result = self.client.generate(request)
            record.update(result.metadata)
            record["status"] = "ok"
            return result
        except OmniIntentsError as error:
            record.update(status="error", error_code=error.code)
            raise
        finally:
            record["duration_s"] = time.perf_counter() - start
            self.calls.append(record)


def generate_json(
    client: LLMClient, request: LLMRequest, config: OmniIntentsConfig, trace: PipelineTrace | None,
) -> dict[str, Any]:
    result = client.generate(request)
    data = require_json_object(result.text)
    if trace is not None:
        trace.add("model_call", {"stage": request.stage, "metadata": result.metadata})
    return data
