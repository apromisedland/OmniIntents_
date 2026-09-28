"""TaskGPT revisits original images when available, conditioned on a selection."""

from __future__ import annotations

from ..config import OmniIntentsConfig
from ..llm.base import LLMClient, generate_json
from ..prompts import request_for
from ..types import Intent, PipelineTrace, StructuredText, TaskPlan
from ..utils.image_utils import choose_keyframes
from ..utils.validate import parse_task_plan
from .ifas import IntentFocusedAttentionShifter


class TaskGPT:
    def __init__(self, llm: LLMClient, config: OmniIntentsConfig | None = None):
        self.llm = llm
        self.config = config or OmniIntentsConfig()
        self.ifas = IntentFocusedAttentionShifter()

    def plan(self, *, structured: StructuredText, intent: Intent, trace: PipelineTrace | None = None) -> TaskPlan:
        focus = self.ifas.shift(structured, intent) if self.config.ifas_enabled else ""
        payload = {
            "input": structured.prompt_dict(), "selected_intent": intent.to_json_dict(),
            "focus_instruction": focus,
            "ifas_mode": ("raw_images" if structured.image_paths else "preextracted") if self.config.ifas_enabled else "disabled",
        }
        request = request_for("task", payload, self.config, choose_keyframes(structured.image_paths))
        plan = parse_task_plan(generate_json(self.llm, request, self.config, trace))
        if trace is not None:
            trace.add("task_plan", plan.to_json_dict())
        return plan
