from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Tuple

from ..config import OmniIntentsConfig
from ..types import AgentType, Capability, TaskPlan


def _missing(required: List[Capability], available: FrozenSet[Capability]) -> List[Capability]:
    return [c for c in required if c not in available]


@dataclass
class ImplicitDecisionTree:
    """Implicit Decision Tree (paper 6.3.1).

    The excerpt describes a *dynamic* step-by-step analysis of functionalities
    (voice / visual / physical), narrowing down to the best agent type.

    We implement a deterministic version that:
    - first checks physical requirements
    - then checks vision perception vs. visual expression
    - then checks text processing needs
    """

    config: OmniIntentsConfig = OmniIntentsConfig()

    def recommend_candidates_with_path(self, task_plan: TaskPlan) -> Tuple[List[AgentType], List[str]]:
        required = task_plan.required_capabilities()
        path: List[str] = []
        # 1) Physical
        if Capability.PHYSICAL_INTERACTION in required or Capability.NAVIGATION in required:
            path.append("requires physical_interaction/navigation -> physical_robot")
            return [AgentType.PHYSICAL_ROBOT], path

        # 2) Vision perception
        if Capability.VISION_PERCEPTION in required:
            path.append("requires vision_perception -> multimodal/ar")
            return [AgentType.MULTIMODAL_ASSISTANT, AgentType.AR_COMPANION, AgentType.DIGITAL_ASSISTANT], path

        # 3) Visual expression only (e.g., AR overlays)
        if Capability.VISUAL_EXPRESSION in required:
            path.append("requires visual_expression (no vision perception) -> ar/digital")
            return [AgentType.AR_COMPANION, AgentType.DIGITAL_ASSISTANT, AgentType.VOICE_ASSISTANT], path

        # 4) Text processing
        if Capability.TEXT_IO in required:
            path.append("requires text_io -> digital/voice")
            return [AgentType.DIGITAL_ASSISTANT, AgentType.VOICE_ASSISTANT], path

        # 5) Speech-only
        path.append("default -> voice assistant (cheapest)")
        return [AgentType.VOICE_ASSISTANT, AgentType.DIGITAL_ASSISTANT], path

    def recommend_candidates(self, task_plan: TaskPlan) -> List[AgentType]:
        cands, _ = self.recommend_candidates_with_path(task_plan)
        return cands


@dataclass
class ExplicitDecisionTree:
    """Explicit Decision Tree (paper 6.3.1) for comparison.

    The paper states it categorizes tasks into 4 criteria:
    - Speech Expression
    - Visual Expression
    - Physical Interaction
    - Creativity

    We approximate mapping from TaskPlan.required_capabilities().
    """

    config: OmniIntentsConfig = OmniIntentsConfig()

    def categorize(self, task_plan: TaskPlan) -> Dict[str, bool]:
        required = set(task_plan.required_capabilities())
        return {
            "Speech Expression": Capability.SPEECH_IO in required,
            "Visual Expression": Capability.VISUAL_EXPRESSION in required,
            "Physical Interaction": (Capability.PHYSICAL_INTERACTION in required) or (Capability.NAVIGATION in required),
            "Creativity": Capability.CREATIVITY in required,
        }

    def recommend(self, task_plan: TaskPlan) -> AgentType:
        flags = self.categorize(task_plan)
        if flags["Physical Interaction"]:
            return AgentType.PHYSICAL_ROBOT
        if flags["Visual Expression"] and flags["Speech Expression"]:
            return AgentType.AR_COMPANION
        if flags["Visual Expression"]:
            return AgentType.DIGITAL_ASSISTANT
        if flags["Creativity"]:
            return AgentType.DIGITAL_ASSISTANT
        if flags["Speech Expression"]:
            return AgentType.VOICE_ASSISTANT
        return AgentType.VOICE_ASSISTANT


def cost_optimized_pick(
    *,
    candidates: List[AgentType],
    required: List[Capability],
    agent_capabilities: Dict[AgentType, FrozenSet[Capability]],
    agent_cost: Dict[AgentType, float],
) -> Tuple[AgentType, List[Capability], List[Capability], float]:
    """Pick lowest-cost agent that satisfies all required capabilities when possible."""

    satisfying: List[AgentType] = []
    for a in candidates:
        avail = agent_capabilities[a]
        miss = _missing(required, avail)
        if not miss:
            satisfying.append(a)

    if satisfying:
        best = min(satisfying, key=lambda a: agent_cost[a])
        avail = agent_capabilities[best]
        miss = _missing(required, avail)
        sat = [c for c in required if c in avail]
        return best, sat, miss, float(agent_cost[best])

    # Otherwise choose the candidate that covers most capabilities, then cheaper.
    def score(a: AgentType) -> Tuple[int, float]:
        avail = agent_capabilities[a]
        covered = sum(1 for c in required if c in avail)
        # higher covered better; lower cost better
        return (covered, -agent_cost[a])

    best = max(candidates, key=score)
    avail = agent_capabilities[best]
    miss = _missing(required, avail)
    sat = [c for c in required if c in avail]
    return best, sat, miss, float(agent_cost[best])
