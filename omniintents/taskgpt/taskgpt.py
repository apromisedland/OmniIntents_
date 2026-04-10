from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..config import OmniIntentsConfig
from ..llm.base import LLMClient
from ..types import Capability, Intent, PipelineTrace, StructuredText, TaskPlan, TaskStep
from ..utils.json_utils import extract_first_json_object, truncate, dumps_pretty
from ..utils.validate import parse_capabilities, validate_task_plan_json
from .ifas import IntentFocusedAttentionShifter


@dataclass
class TaskGPT:
    """TaskGPT module (paper 6.2).

    - Takes intent from IntentGPT
    - Produces a task plan
    - Uses IFAS to improve precision by focusing attention on intent-relevant inputs
    """

    llm: LLMClient
    config: OmniIntentsConfig = OmniIntentsConfig()
    ifas: Optional[IntentFocusedAttentionShifter] = None

    def __post_init__(self) -> None:
        self.ifas = self.ifas or IntentFocusedAttentionShifter()

    def plan(self, *, structured: StructuredText, intent: Intent, trace: Optional[PipelineTrace] = None) -> TaskPlan:
        filtered: Dict[str, Any] = structured.to_json_dict()
        focus_instruction = ""
        if self.config.ifas_enabled and self.ifas is not None:
            filtered, focus_instruction = self.ifas.shift(structured=structured, intent=intent)

        prompt = self._build_prompt(filtered, intent, focus_instruction)
        if trace is not None and self.config.enable_trace:
            trace.add("taskgpt.prompt", {"prompt": truncate(prompt, self.config.max_prompt_chars_in_trace)})

        llm_result = self.llm.complete(prompt, temperature=0.2, max_tokens=1000)
        if trace is not None and self.config.enable_trace:
            trace.add("taskgpt.llm_output", {"text": truncate(llm_result.text, self.config.max_prompt_chars_in_trace)})

        data = extract_first_json_object(llm_result.text) or {}
        ok, errors = validate_task_plan_json(data)

        if not ok:
            plan = self._heuristic_plan(intent)
            if trace is not None and self.config.enable_trace:
                trace.add("taskgpt.fallback", {"errors": errors, "task_plan": plan.to_json_dict()})
            return plan

        # Parse
        goal = str(data.get("goal", "")).strip() or f"Handle intent: {intent.label}"
        steps_raw = data.get("steps", []) or []
        steps: List[TaskStep] = []
        for s in steps_raw:
            if not isinstance(s, dict):
                continue
            try:
                step_id = int(s.get("step_id"))
            except Exception:
                step_id = len(steps) + 1
            instr = str(s.get("instruction", "")).strip()
            caps = parse_capabilities(s.get("required_capabilities", []))
            steps.append(TaskStep(step_id=step_id, instruction=instr, required_capabilities=caps))

        plan = TaskPlan(goal=goal, steps=steps)
        if trace is not None and self.config.enable_trace:
            trace.add("taskgpt.task_plan", plan.to_json_dict())
        return plan

    def _build_prompt(self, filtered_structured: Dict[str, Any], intent: Intent, focus_instruction: str) -> str:
        instruction = (
            "You are TaskGPT. Create an actionable, step-by-step task plan that matches the user's intent. "
            "Avoid losing crucial details (especially spatial relationships) in complex scenes. "
            "Follow the focus instruction to allocate attention to intent-relevant inputs."
        )
        prompt = "\n\n".join(
            [
                instruction,
                "TASK_PLAN_JSON",
                f"INTENT_LABEL: {intent.label}",
                f"INTENT_DESCRIPTION: {intent.description}",
                f"FOCUS_INSTRUCTION: {focus_instruction}",
                "FILTERED_STRUCTURED_TEXT_JSON:",
                dumps_pretty(filtered_structured),
                "### OUTPUT FORMAT",
                "Return a JSON object with keys:",
                "- goal: string",
                "- steps: list of {step_id: int, instruction: string, required_capabilities: list[string]}",
                "Valid capability strings are:",
                ", ".join([c.value for c in Capability]),
            ]
        )
        return prompt

    def _heuristic_plan(self, intent: Intent) -> TaskPlan:
        label = (intent.label or "").lower()
        if "search" in label or "query" in label:
            return TaskPlan(
                goal="Find and present the requested information.",
                steps=[
                    TaskStep(1, "Identify the referenced entity/object from context.", [Capability.VISION_PERCEPTION, Capability.SPEECH_IO]),
                    TaskStep(2, "Construct a search query.", [Capability.TEXT_IO]),
                    TaskStep(3, "Retrieve information from a digital source.", [Capability.TEXT_IO]),
                    TaskStep(4, "Present a concise answer.", [Capability.SPEECH_IO, Capability.TEXT_IO]),
                ],
            )

        if "collaborator" in label or "navigate" in label:
            return TaskPlan(
                goal="Assist the user with navigation/collaboration in the environment.",
                steps=[
                    TaskStep(1, "Clarify the destination/target and constraints.", [Capability.SPEECH_IO]),
                    TaskStep(2, "Perceive obstacles/landmarks and plan a safe path.", [Capability.VISION_PERCEPTION]),
                    TaskStep(3, "Provide step-by-step guidance (optionally visual).", [Capability.SPEECH_IO, Capability.VISUAL_EXPRESSION]),
                ],
            )

        if "physical" in label or intent.requires_physical:
            return TaskPlan(
                goal="Execute the requested physical task safely.",
                steps=[
                    TaskStep(1, "Perceive the workspace and locate target.", [Capability.VISION_PERCEPTION]),
                    TaskStep(2, "Plan a safe physical interaction sequence.", [Capability.PHYSICAL_INTERACTION, Capability.NAVIGATION]),
                    TaskStep(3, "Execute the action while monitoring safety.", [Capability.PHYSICAL_INTERACTION, Capability.VISION_PERCEPTION]),
                    TaskStep(4, "Confirm completion.", [Capability.SPEECH_IO]),
                ],
            )

        if "companion" in label:
            return TaskPlan(
                goal="Engage in a helpful conversation.",
                steps=[TaskStep(1, "Respond and ask a clarifying follow-up question.", [Capability.SPEECH_IO])],
            )

        return TaskPlan(goal=f"Handle intent: {intent.label}", steps=[TaskStep(1, "Ask a clarification question.", [Capability.SPEECH_IO])])
