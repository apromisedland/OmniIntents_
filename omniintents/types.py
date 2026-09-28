"""Versioned runtime objects. Ground-truth labels never belong to model inputs."""

from __future__ import annotations

import enum
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .errors import ValidationError

SCHEMA_VERSION = "0.3"


def json_value(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


class Serializable:
    def to_json_dict(self) -> dict[str, Any]:
        return json_value(asdict(self))


class Capability(str, enum.Enum):
    SPEECH_IO = "speech_io"
    TEXT_IO = "text_io"
    VISION_PERCEPTION = "vision_perception"
    VISUAL_EXPRESSION = "visual_expression"
    NAVIGATION_GUIDANCE = "navigation_guidance"
    PHYSICAL_MOVEMENT = "physical_movement"
    PHYSICAL_INTERACTION = "physical_interaction"
    CREATIVITY = "creativity"


class AgentType(str, enum.Enum):
    VOICE_ASSISTANT = "voice_assistant"
    AR_AGENT = "ar_agent"
    GENERATIVE_AI_AGENT = "generative_ai_agent"
    PHYSICAL_AGENT = "physical_agent"


def finite_number(value: Any, name: str, lower: float | None = None, upper: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValidationError(f"{name} must be a finite number")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValidationError(f"{name} is outside its permitted range")
    return float(value)


@dataclass
class MultimodalInput(Serializable):
    utterance: str = ""
    sample_id: str = "input"
    session_id: str = "default"
    sequence_index: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    image_paths: list[str] = field(default_factory=list)
    audio_path: str | None = None
    speech_transcript: str | None = None
    speech_confidence: float | None = None
    brightness: float | None = None
    volume_db: float | None = None
    sound_events: list[dict[str, Any]] | None = None
    objects: list[str] | None = None
    context_location: str | None = None
    context_activity: str | None = None
    hand_state: str | None = None
    hand_target: str | None = None
    eye_state: str | None = None
    eye_target: str | None = None
    voice_target: str | None = None
    scene_description: str | None = None

    def __post_init__(self) -> None:
        for name in ("utterance", "sample_id", "session_id"):
            if not isinstance(getattr(self, name), str):
                raise ValidationError(f"{name} must be a string")
        if not self.sample_id.strip() or not self.session_id.strip():
            raise ValidationError("sample_id and session_id must be non-empty")
        if type(self.sequence_index) is not int or self.sequence_index < 0:
            raise ValidationError("sequence_index must be a non-negative integer")
        if not isinstance(self.timestamp, datetime) or self.timestamp.utcoffset() is None:
            raise ValidationError("timestamp must include a timezone")
        for name in ("speech_confidence", "brightness"):
            value = getattr(self, name)
            if value is not None:
                finite_number(value, name, 0, 1)
        if self.volume_db is not None:
            finite_number(self.volume_db, "volume_db")
        if not isinstance(self.image_paths, list) or not all(isinstance(path, str) and path for path in self.image_paths):
            raise ValidationError("image_paths must be a list of non-empty paths")
        if self.objects is not None and (
            not isinstance(self.objects, list) or not all(isinstance(item, str) for item in self.objects)
        ):
            raise ValidationError("objects must be a list of strings")
        for name in (
            "audio_path", "speech_transcript", "context_location", "context_activity",
            "hand_state", "hand_target", "eye_state", "eye_target", "voice_target", "scene_description",
        ):
            if getattr(self, name) is not None and not isinstance(getattr(self, name), str):
                raise ValidationError(f"{name} must be a string or null")
        if self.sound_events is not None:
            if not isinstance(self.sound_events, list):
                raise ValidationError("sound_events must be a list")
            for event in self.sound_events:
                if not isinstance(event, dict) or not isinstance(event.get("label"), str):
                    raise ValidationError("sound events require a label and confidence")
                finite_number(event.get("confidence"), "sound event confidence", 0, 1)

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> MultimodalInput:
        if not isinstance(data, dict):
            raise ValidationError("Input must be a JSON object")
        values = dict(data)
        if "timestamp" in values:
            try:
                values["timestamp"] = datetime.fromisoformat(values["timestamp"].replace("Z", "+00:00"))
            except (AttributeError, TypeError, ValueError) as error:
                raise ValidationError("timestamp must be an ISO 8601 string") from error
        try:
            return cls(**values)
        except TypeError as error:
            raise ValidationError(f"Unsupported input fields: {error}") from error


@dataclass
class StructuredText(Serializable):
    utterance: str = ""
    visual: dict[str, Any] = field(default_factory=dict)
    audio: dict[str, Any] = field(default_factory=dict)
    credibility_report: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)
    history_summary: str = ""
    source_mode: str = "preextracted"
    image_paths: list[str] = field(default_factory=list)
    errors: list[dict[str, str]] = field(default_factory=list)

    def prompt_dict(self) -> dict[str, Any]:
        data = self.to_json_dict()
        data.pop("image_paths")
        data.pop("errors")
        return data


@dataclass
class Intent(Serializable):
    specific_label: str
    description: str
    confidence: float
    general_label: str = ""

    def __post_init__(self) -> None:
        from .intentgpt.taxonomy import SPECIFIC_TO_GENERAL

        if not isinstance(self.specific_label, str) or self.specific_label not in SPECIFIC_TO_GENERAL:
            raise ValidationError(f"Unknown specific label: {self.specific_label!r}")
        expected = SPECIFIC_TO_GENERAL[self.specific_label]
        if self.general_label and self.general_label != expected:
            raise ValidationError("General and specific intent labels disagree")
        self.general_label = expected
        finite_number(self.confidence, "confidence", 0, 1)
        if not isinstance(self.description, str) or not self.description.strip():
            raise ValidationError("Intent description must be non-empty")

    @property
    def label(self) -> str:
        return self.specific_label


@dataclass
class IntentPrediction(Serializable):
    sample_id: str
    session_id: str
    sequence_index: int
    timestamp: datetime
    structured_text: StructuredText
    candidates: list[Intent]
    complementarity: str
    ambiguity: str
    schema_version: str = SCHEMA_VERSION
    backend: str = ""
    evidence_summary: str = ""


@dataclass
class TaskStep(Serializable):
    step_id: int
    instruction: str
    required_capabilities: list[Capability] = field(default_factory=list)

    def __post_init__(self) -> None:
        if type(self.step_id) is not int or self.step_id < 1:
            raise ValidationError("step_id must be a positive integer")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValidationError("Task instruction must be non-empty")
        if not isinstance(self.required_capabilities, list) or not self.required_capabilities:
            raise ValidationError("Task capabilities must be a non-empty list")
        try:
            self.required_capabilities = list(dict.fromkeys(Capability(value) for value in self.required_capabilities))
        except (ValueError, TypeError) as error:
            raise ValidationError("Unknown task capability") from error


@dataclass
class TaskPlan(Serializable):
    goal: str
    steps: list[TaskStep]

    def __post_init__(self) -> None:
        if not isinstance(self.goal, str) or not self.goal.strip():
            raise ValidationError("Task goal must be non-empty")
        if not isinstance(self.steps, list) or not self.steps or any(
            not isinstance(step, TaskStep) or step.step_id != index
            for index, step in enumerate(self.steps, 1)
        ):
            raise ValidationError("Task steps must be non-empty and consecutively numbered")

    def required_capabilities(self) -> list[Capability]:
        return list(dict.fromkeys(
            capability for step in self.steps for capability in step.required_capabilities
        ))


@dataclass
class AgentRecommendation(Serializable):
    agent_type: AgentType | None
    rationale: str
    estimated_cost: float | None
    satisfied_capabilities: list[Capability]
    missing_capabilities: list[Capability]
    status: str = "selected"
    strategy: str = "implicit"


@dataclass
class PipelineTrace(Serializable):
    enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    def add(self, name: str, payload: dict[str, Any] | None = None) -> None:
        if self.enabled:
            self.events.append({"name": name, "payload": json_value(payload or {})})


@dataclass
class OmniIntentsResult(Serializable):
    status: str
    prediction: IntentPrediction | None = None
    selected_index: int | None = None
    task_plan: TaskPlan | None = None
    agent: AgentRecommendation | None = None
    errors: list[dict[str, str]] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION

    @property
    def intent(self) -> Intent | None:
        if self.prediction is None or self.selected_index is None:
            return None
        return self.prediction.candidates[self.selected_index]

    @property
    def structured_text(self) -> StructuredText | None:
        return self.prediction.structured_text if self.prediction else None
