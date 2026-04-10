from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence


class Capability(str, enum.Enum):
    """Atomic capability flags used by TaskGPT & AgentGPT."""

    # Interaction / IO
    SPEECH_IO = "speech_io"
    TEXT_IO = "text_io"
    VISUAL_EXPRESSION = "visual_expression"

    # Perception
    VISION_PERCEPTION = "vision_perception"
    AUDIO_PERCEPTION = "audio_perception"

    # Physical
    PHYSICAL_INTERACTION = "physical_interaction"
    NAVIGATION = "navigation"

    # Optional (paper mentions creativity in explicit decision tree criteria)
    CREATIVITY = "creativity"


class AgentType(str, enum.Enum):
    VOICE_ASSISTANT = "voice_assistant"
    DIGITAL_ASSISTANT = "digital_assistant"
    MULTIMODAL_ASSISTANT = "multimodal_assistant"
    AR_COMPANION = "ar_companion"
    PHYSICAL_ROBOT = "physical_robot"


@dataclass
class MultimodalInput:
    """Input container for OmniIntents.

    This implementation supports:
    - utterance text
    - one or multiple image paths (keyframes)
    - audio file path

    It also supports injection of pre-extracted metadata so the pipeline can run
    without external APIs/models (e.g., injected speech transcript/confidence).
    """

    utterance: str = ""
    image_paths: Sequence[str] = field(default_factory=list)
    audio_path: Optional[str] = None

    # Optional: pre-extracted speech transcript/confidence
    speech_transcript: Optional[str] = None
    speech_confidence: Optional[float] = None

    # Optional: injected vision/context metadata
    objects: Optional[List[str]] = None
    context_location: Optional[str] = None
    context_activity: Optional[str] = None
    hand_state: Optional[str] = None
    eye_state: Optional[str] = None
    interaction_target: Optional[str] = None

    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class StructuredText:
    """Unified structured representation stored as JSON."""

    visual: Dict[str, Any] = field(default_factory=dict)
    audio: Dict[str, Any] = field(default_factory=dict)
    credibility_report: Dict[str, Any] = field(default_factory=dict)
    history_summary: str = ""
    history_retrieval: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "visual": self.visual,
            "audio": self.audio,
            "credibility_report": self.credibility_report,
            "history_summary": self.history_summary,
            "history_retrieval": self.history_retrieval,
            "raw": self.raw,
        }


@dataclass
class Intent:
    label: str
    description: str
    confidence: float = 0.5
    entities: Dict[str, Any] = field(default_factory=dict)

    requires_visual: bool = False
    requires_audio: bool = False
    requires_physical: bool = False

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "description": self.description,
            "confidence": self.confidence,
            "entities": self.entities,
            "requires_visual": self.requires_visual,
            "requires_audio": self.requires_audio,
            "requires_physical": self.requires_physical,
        }


@dataclass
class TaskStep:
    step_id: int
    instruction: str
    required_capabilities: List[Capability] = field(default_factory=list)

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "step_id": self.step_id,
            "instruction": self.instruction,
            "required_capabilities": [c.value for c in self.required_capabilities],
        }


@dataclass
class TaskPlan:
    goal: str
    steps: List[TaskStep] = field(default_factory=list)

    def required_capabilities(self) -> List[Capability]:
        caps: List[Capability] = []
        for s in self.steps:
            caps.extend(s.required_capabilities)
        # unique, preserve order
        seen = set()
        out: List[Capability] = []
        for c in caps:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    def to_json_dict(self) -> Dict[str, Any]:
        return {"goal": self.goal, "steps": [s.to_json_dict() for s in self.steps]}


@dataclass
class AgentRecommendation:
    agent_type: AgentType
    rationale: str
    estimated_cost: float
    satisfied_capabilities: List[Capability]
    missing_capabilities: List[Capability]

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "rationale": self.rationale,
            "estimated_cost": self.estimated_cost,
            "satisfied_capabilities": [c.value for c in self.satisfied_capabilities],
            "missing_capabilities": [c.value for c in self.missing_capabilities],
        }


@dataclass
class OmniIntentsResult:
    structured_text: StructuredText
    intent: Intent
    task_plan: TaskPlan
    agent: AgentRecommendation

    def to_json_dict(self) -> Dict[str, Any]:
        return {
            "structured_text": self.structured_text.to_json_dict(),
            "intent": self.intent.to_json_dict(),
            "task_plan": self.task_plan.to_json_dict(),
            "agent": self.agent.to_json_dict(),
        }


@dataclass
class TraceEvent:
    name: str
    timestamp: str
    payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineTrace:
    """Collects intermediate artifacts for debugging/analysis."""

    events: List[TraceEvent] = field(default_factory=list)

    def add(self, name: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            TraceEvent(name=name, timestamp=datetime.utcnow().isoformat() + "Z", payload=payload or {})
        )

    def to_json_dict(self) -> Dict[str, Any]:
        return {"events": [{"name": e.name, "timestamp": e.timestamp, "payload": e.payload} for e in self.events]}
