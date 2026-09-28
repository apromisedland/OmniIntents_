"""Five-keyframe brightness and structured visual extraction."""

from __future__ import annotations

from typing import Any

from ..config import OmniIntentsConfig
from ..errors import ValidationError
from ..llm.base import LLMClient, generate_json
from ..prompts import request_for
from ..types import MultimodalInput, PipelineTrace
from ..utils.image_utils import choose_keyframes, smoothed_brightness
from ..utils.validate import require_fields


class VisualProcessor:
    def __init__(self, llm: LLMClient, config: OmniIntentsConfig | None = None):
        self.llm = llm
        self.config = config or OmniIntentsConfig()

    def process(self, data: MultimodalInput, trace: PipelineTrace | None = None) -> dict[str, Any]:
        paths = choose_keyframes(data.image_paths)
        extracted: dict[str, Any] = {}
        brightness = data.brightness
        if paths:
            if brightness is None:
                brightness = smoothed_brightness(paths)
            request = request_for("vision", {"purpose": "structured multimodal input"}, self.config, paths)
            extracted = generate_json(self.llm, request, self.config, trace)
            fields = {"activity", "location", "objects", "eye_state", "eye_target",
                      "hand_state", "hand_target", "scene_description"}
            require_fields(extracted, fields)
            if not isinstance(extracted["objects"], list) or not all(isinstance(item, str) for item in extracted["objects"]):
                raise ValidationError("Vision objects must be a list of strings")
            if any(extracted[field] is not None and not isinstance(extracted[field], str) for field in fields - {"objects"}):
                raise ValidationError("Vision scalar fields must be string or null")
        def choose(name: str, source_name: str | None = None) -> Any:
            supplied = getattr(data, name)
            return supplied if supplied is not None else extracted.get(source_name or name)
        result = {
            "status": "present" if paths or any(value is not None for value in (
                data.objects, data.brightness, data.context_location, data.context_activity,
                data.hand_state, data.eye_state, data.hand_target, data.eye_target,
            )) else "missing",
            "brightness": brightness,
            "brightness_method": "paper_linear_RGB_equation",
            "keyframe_count": len(paths),
            "context": {"location": choose("context_location", "location"), "activity": choose("context_activity", "activity")},
            "objects": choose("objects") or [],
            "hand_state": choose("hand_state"), "hand_target": choose("hand_target"),
            "eye_state": choose("eye_state"), "eye_target": choose("eye_target"),
            "scene_description": choose("scene_description"),
        }
        return result
