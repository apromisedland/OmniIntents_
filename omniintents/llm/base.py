from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol


@dataclass
class LLMResult:
    text: str
    raw: Optional[Dict[str, Any]] = None


class LLMClient(Protocol):
    """LLM interface needed by this project.

    - `complete`: text completion/chat
    - `vision_describe`: (optional) image -> description
    """

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 800) -> LLMResult:
        ...

    def vision_describe(self, image_path: str, prompt: str, *, max_tokens: int = 300) -> LLMResult:
        ...
