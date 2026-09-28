"""Explicit experimental defaults; numerical costs are relative, not currency."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ValidationError
from .types import AgentType, Capability, Serializable, finite_number


def default_capabilities() -> dict[str, list[str]]:
    communication = [Capability.SPEECH_IO.value, Capability.TEXT_IO.value]
    visual = communication + [Capability.VISION_PERCEPTION.value, Capability.VISUAL_EXPRESSION.value,
                              Capability.NAVIGATION_GUIDANCE.value]
    return {
        AgentType.VOICE_ASSISTANT.value: communication,
        AgentType.AR_AGENT.value: visual,
        AgentType.GENERATIVE_AI_AGENT.value: visual + [Capability.CREATIVITY.value],
        AgentType.PHYSICAL_AGENT.value: visual + [
            Capability.PHYSICAL_MOVEMENT.value, Capability.PHYSICAL_INTERACTION.value,
        ],
    }


@dataclass
class OmniIntentsConfig(Serializable):
    brightness_credible_threshold: float = 0.2
    speech_confidence_credible_threshold: float = 0.8
    sound_event_confidence_threshold: float = 0.70
    audio_frame_hz: int = 20
    voiced_rms_threshold: float = 0.01
    db_calibration: float = 0.0
    max_history_entries: int = 5
    history_rounds: int = 2
    cbas_enabled: bool = True
    ifas_enabled: bool = True
    enable_trace: bool = False
    agent_strategy: str = "implicit"
    failure_policy: str = "error"
    removed_modalities: tuple[str, ...] = ()
    structured_ablation_mode: bool = False
    temperature: float = 0.0
    max_tokens: int = 1600
    seed: int = 0
    agent_cost: dict[str, float] = field(default_factory=lambda: {
        "voice_assistant": 1.0, "ar_agent": 2.0, "generative_ai_agent": 3.0, "physical_agent": 4.0,
    })
    agent_capabilities: dict[str, list[str]] = field(default_factory=default_capabilities)

    def __post_init__(self) -> None:
        for name in ("brightness_credible_threshold", "speech_confidence_credible_threshold",
                     "sound_event_confidence_threshold", "voiced_rms_threshold"):
            finite_number(getattr(self, name), name, 0, 1)
        finite_number(self.temperature, "temperature", 0, 2)
        finite_number(self.db_calibration, "db_calibration")
        for name in ("audio_frame_hz", "max_tokens"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValidationError(f"{name} must be a positive integer")
        if type(self.max_history_entries) is not int or not 0 <= self.max_history_entries <= 5:
            raise ValidationError("max_history_entries must be in 0..5")
        if type(self.history_rounds) is not int or not 0 <= self.history_rounds <= self.max_history_entries:
            raise ValidationError("history_rounds must be in 0..max_history_entries")
        for name in ("cbas_enabled", "ifas_enabled", "enable_trace", "structured_ablation_mode"):
            if type(getattr(self, name)) is not bool:
                raise ValidationError(f"{name} must be boolean")
        if type(self.seed) is not int:
            raise ValidationError("seed must be an integer")
        if not isinstance(self.agent_strategy, str) or self.agent_strategy not in {"implicit", "explicit"}:
            raise ValidationError("agent_strategy must be implicit or explicit")
        if self.failure_policy != "error":
            raise ValidationError("Automatic fallbacks are disabled; explicitly choose the mock backend for demos")
        if not isinstance(self.removed_modalities, (list, tuple)) or not all(isinstance(value, str) for value in self.removed_modalities):
            raise ValidationError("removed_modalities must be a sequence")
        self.removed_modalities = tuple(self.removed_modalities)
        if not set(self.removed_modalities) <= {"context", "objects", "hand_eye", "speech", "targets", "audio_labels"}:
            raise ValidationError("Unknown modality ablation")
        expected = {agent.value for agent in AgentType}
        if not isinstance(self.agent_cost, dict) or not isinstance(self.agent_capabilities, dict):
            raise ValidationError("Agent cost and capabilities must be JSON objects")
        if set(self.agent_cost) != expected or set(self.agent_capabilities) != expected:
            raise ValidationError("Cost and capability tables must contain exactly the four paper agents")
        for agent, cost in self.agent_cost.items():
            finite_number(cost, f"cost for {agent}", 0)
        for capabilities in self.agent_capabilities.values():
            if not isinstance(capabilities, list):
                raise ValidationError("Agent capabilities must be lists")
            try:
                for capability in capabilities:
                    Capability(capability)
            except (ValueError, TypeError) as error:
                raise ValidationError("Unknown agent capability") from error

    @classmethod
    def from_json_dict(cls, data: dict[str, Any]) -> OmniIntentsConfig:
        if not isinstance(data, dict):
            raise ValidationError("Configuration must be a JSON object")
        try:
            return cls(**data)
        except TypeError as error:
            raise ValidationError(f"Invalid configuration: {error}") from error
