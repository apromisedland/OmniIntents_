from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional

from .types import AgentType, Capability


@dataclass(frozen=True)
class OmniIntentsConfig:
    """Central configuration.

    Defaults are aligned with the provided paper excerpt:
    - brightness threshold: 0.15
    - speech confidence threshold: 0.70
    - sound event confidence threshold: 0.70
    - audio frames: 20 fps
    - memory size: 5
    """

    # -------------------------
    # IntentGPT: Credibility
    # -------------------------
    brightness_credible_threshold: float = 0.15
    speech_confidence_credible_threshold: float = 0.70
    sound_event_confidence_threshold: float = 0.70

    # If modality input is missing (no image/audio), should it be considered unreliable?
    missing_visual_is_incredible: bool = True
    missing_audio_is_incredible: bool = True

    # -------------------------
    # Audio volume measurement
    # -------------------------
    audio_frame_hz: int = 20  # paper: Web Audio API samples at 20 fps
    voiced_rms_threshold: float = 0.01
    db_calibration: float = 0.0

    # -------------------------
    # Contextual Memory Tracker
    # -------------------------
    max_history_entries: int = 5
    memory_retrieval_top_k: int = 3

    # -------------------------
    # TaskGPT: IFAS
    # -------------------------
    ifas_enabled: bool = True

    # -------------------------
    # Trace & debugging
    # -------------------------
    enable_trace: bool = True
    max_prompt_chars_in_trace: int = 8000

    # -------------------------
    # AgentGPT: Cost optimization
    # -------------------------
    agent_cost: Optional[Dict[AgentType, float]] = None
    agent_capabilities: Optional[Dict[AgentType, FrozenSet[Capability]]] = None

    def __post_init__(self) -> None:
        if self.agent_cost is None:
            object.__setattr__(
                self,
                "agent_cost",
                {
                    AgentType.VOICE_ASSISTANT: 1.0,
                    AgentType.DIGITAL_ASSISTANT: 1.2,
                    AgentType.MULTIMODAL_ASSISTANT: 1.6,
                    AgentType.AR_COMPANION: 2.0,
                    AgentType.PHYSICAL_ROBOT: 5.0,
                },
            )
        if self.agent_capabilities is None:
            object.__setattr__(
                self,
                "agent_capabilities",
                {
                    AgentType.VOICE_ASSISTANT: frozenset({Capability.SPEECH_IO}),
                    AgentType.DIGITAL_ASSISTANT: frozenset({Capability.SPEECH_IO, Capability.TEXT_IO}),
                    AgentType.MULTIMODAL_ASSISTANT: frozenset(
                        {Capability.SPEECH_IO, Capability.TEXT_IO, Capability.VISION_PERCEPTION}
                    ),
                    AgentType.AR_COMPANION: frozenset(
                        {
                            Capability.SPEECH_IO,
                            Capability.TEXT_IO,
                            Capability.VISION_PERCEPTION,
                            Capability.VISUAL_EXPRESSION,
                        }
                    ),
                    AgentType.PHYSICAL_ROBOT: frozenset(
                        {
                            Capability.SPEECH_IO,
                            Capability.VISION_PERCEPTION,
                            Capability.PHYSICAL_INTERACTION,
                            Capability.NAVIGATION,
                        }
                    ),
                },
            )
