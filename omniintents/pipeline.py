from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .agentgpt.agentgpt import AgentGPT
from .config import OmniIntentsConfig
from .intentgpt.intentgpt import IntentGPT
from .llm.base import LLMClient
from .taskgpt.taskgpt import TaskGPT
from .types import MultimodalInput, OmniIntentsResult, PipelineTrace


@dataclass
class OmniIntentsPipeline:
    """End-to-end OmniIntents pipeline orchestrator."""

    llm: LLMClient
    config: OmniIntentsConfig = OmniIntentsConfig()
    intentgpt: Optional[IntentGPT] = None
    taskgpt: Optional[TaskGPT] = None
    agentgpt: Optional[AgentGPT] = None

    def __post_init__(self) -> None:
        self.intentgpt = self.intentgpt or IntentGPT(llm=self.llm, config=self.config)
        self.taskgpt = self.taskgpt or TaskGPT(llm=self.llm, config=self.config)
        self.agentgpt = self.agentgpt or AgentGPT(llm=self.llm, config=self.config)

    def run(self, mm: MultimodalInput) -> OmniIntentsResult:
        result, _ = self.run_with_trace(mm)  # trace discarded
        return result

    def run_with_trace(self, mm: MultimodalInput) -> Tuple[OmniIntentsResult, PipelineTrace]:
        trace = PipelineTrace()
        trace.add("pipeline.start", {"timestamp": mm.timestamp.isoformat(), "utterance": mm.utterance})

        structured, intent = self.intentgpt.predict_intent(mm, trace=trace)  # type: ignore[union-attr]
        task_plan = self.taskgpt.plan(structured=structured, intent=intent, trace=trace)  # type: ignore[union-attr]
        agent = self.agentgpt.recommend(task_plan, trace=trace)  # type: ignore[union-attr]

        result = OmniIntentsResult(
            structured_text=structured,
            intent=intent,
            task_plan=task_plan,
            agent=agent,
        )

        trace.add("pipeline.result", result.to_json_dict())
        trace.add("pipeline.end", {})
        return result, trace
