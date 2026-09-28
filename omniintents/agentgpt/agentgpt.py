"""Implicit model selection and explicit capability-flag baseline."""

from __future__ import annotations

from ..config import OmniIntentsConfig
from ..llm.base import LLMClient, generate_json
from ..prompts import request_for
from ..types import AgentRecommendation, Capability, PipelineTrace, TaskPlan
from ..utils.validate import parse_agent, parse_flags
from .decision_tree import ExplicitDecisionTree


class AgentGPT:
    def __init__(self, llm: LLMClient, config: OmniIntentsConfig | None = None):
        self.llm = llm
        self.config = config or OmniIntentsConfig()

    def recommend(self, task_plan: TaskPlan, trace: PipelineTrace | None = None) -> AgentRecommendation:
        payload = {
            "task_plan": task_plan.to_json_dict(),
            "agent_capabilities": self.config.agent_capabilities,
            "relative_costs": self.config.agent_cost,
        }
        if self.config.agent_strategy == "explicit":
            flags = parse_flags(generate_json(
                self.llm, request_for("agent_flags", payload, self.config), self.config, trace,
            ))
            chosen = ExplicitDecisionTree().recommend(flags)
            rationale = f"Explicit four-criterion decision: {flags}"
        else:
            chosen, rationale = parse_agent(generate_json(
                self.llm, request_for("agent", payload, self.config), self.config, trace,
            ))
        required = task_plan.required_capabilities()
        available = {Capability(value) for value in self.config.agent_capabilities[chosen.value]}
        missing = [capability for capability in required if capability not in available]
        recommendation = AgentRecommendation(
            agent_type=None if missing else chosen,
            rationale=rationale if not missing else f"Rejected {chosen.value}: required capabilities are missing. {rationale}",
            estimated_cost=None if missing else self.config.agent_cost[chosen.value],
            satisfied_capabilities=[capability for capability in required if capability in available],
            missing_capabilities=missing, status="no_suitable_agent" if missing else "selected",
            strategy=self.config.agent_strategy,
        )
        if trace is not None:
            trace.add("agent_recommendation", recommendation.to_json_dict())
        return recommendation
