"""Two-phase prediction and selection. This package never executes physical tasks."""

from __future__ import annotations

from .agentgpt.agentgpt import AgentGPT
from .config import OmniIntentsConfig
from .errors import OmniIntentsError, ValidationError
from .intentgpt.audio import AudioProcessor
from .intentgpt.intentgpt import IntentGPT
from .llm.base import LLMClient
from .taskgpt.taskgpt import TaskGPT
from .types import IntentPrediction, MultimodalInput, OmniIntentsResult, PipelineTrace


class OmniIntentsPipeline:
    def __init__(
        self, llm: LLMClient, config: OmniIntentsConfig | None = None, *,
        audio_processor: AudioProcessor | None = None,
    ):
        self.llm = llm
        self.config = config or OmniIntentsConfig()
        self.intentgpt = IntentGPT(llm, self.config, audio_processor=audio_processor)
        self.taskgpt = TaskGPT(llm, self.config)
        self.agentgpt = AgentGPT(llm, self.config)

    def predict(self, data: MultimodalInput, trace: PipelineTrace | None = None) -> IntentPrediction:
        return self.intentgpt.predict(data, trace)

    def plan_selected(
        self, prediction: IntentPrediction, selected_index: int, *, trace: PipelineTrace | None = None,
    ) -> OmniIntentsResult:
        if type(selected_index) is not int or not 0 <= selected_index < len(prediction.candidates):
            raise ValidationError("selected_index must identify an available candidate")
        selected = prediction.candidates[selected_index]
        result = OmniIntentsResult("planning", prediction, selected_index)
        try:
            result.task_plan = self.taskgpt.plan(
                structured=prediction.structured_text, intent=selected, trace=trace,
            )
            result.status = "selecting_agent"
            result.agent = self.agentgpt.recommend(result.task_plan, trace)
            result.status = "completed" if result.agent.status == "selected" else "no_suitable_agent"
            self.intentgpt.remember(prediction, selected)
        except OmniIntentsError as error:
            result.status = "planning_error" if result.task_plan is None else "selection_error"
            result.errors.append({"code": error.code, "message": str(error)})
        return result

    def run(self, data: MultimodalInput, *, selection: str | int | None = None) -> OmniIntentsResult:
        return self.run_with_trace(data, selection=selection)[0]

    def run_with_trace(
        self, data: MultimodalInput, *, selection: str | int | None = None,
    ) -> tuple[OmniIntentsResult, PipelineTrace]:
        trace = PipelineTrace(self.config.enable_trace)
        if selection is not None and selection != "top1" and type(selection) is not int:
            raise ValidationError("selection must be None, top1, or a candidate index")
        try:
            prediction = self.predict(data, trace)
        except OmniIntentsError as error:
            return OmniIntentsResult("prediction_error", errors=[{"code": error.code, "message": str(error)}]), trace
        if not prediction.candidates:
            result = OmniIntentsResult("needs_clarification", prediction)
        elif selection is None and (len(prediction.candidates) > 1 or prediction.ambiguity != "unambiguous"):
            result = OmniIntentsResult("needs_selection", prediction)
        else:
            index = 0 if selection is None or selection == "top1" else selection
            result = self.plan_selected(prediction, index, trace=trace)
        trace.add("result", result.to_json_dict())
        return result, trace

    def reset_session(self, session_id: str | None = None) -> None:
        self.intentgpt.memory.clear(session_id)
