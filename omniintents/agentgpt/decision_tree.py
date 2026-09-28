"""Four paper agent categories; navigation guidance is not robot movement."""

from ..types import AgentType


class ExplicitDecisionTree:
    def recommend(self, flags: dict[str, bool]) -> AgentType:
        if flags["physical_interaction"]:
            return AgentType.PHYSICAL_AGENT
        if flags["creativity"]:
            return AgentType.GENERATIVE_AI_AGENT
        if flags["visual_expression"]:
            return AgentType.AR_AGENT
        return AgentType.VOICE_ASSISTANT
