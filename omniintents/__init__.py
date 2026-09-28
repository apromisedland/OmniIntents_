"""OmniIntents 0.3: a paper-aligned reference with explicit reproduction limits."""

__version__ = "0.3.0"

from .config import OmniIntentsConfig
from .pipeline import OmniIntentsPipeline
from .types import (
    AgentRecommendation, AgentType, Capability, Intent, IntentPrediction,
    MultimodalInput, OmniIntentsResult, PipelineTrace, StructuredText, TaskPlan, TaskStep,
)

__all__ = [
    "OmniIntentsConfig", "OmniIntentsPipeline", "AgentRecommendation", "AgentType", "Capability",
    "Intent", "IntentPrediction", "MultimodalInput", "OmniIntentsResult", "PipelineTrace",
    "StructuredText", "TaskPlan", "TaskStep",
]
