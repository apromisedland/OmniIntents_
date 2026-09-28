"""Deterministic demonstration rules, never a scientific model substitute."""

from __future__ import annotations

import json

from ..errors import ValidationError
from .base import LLMRequest, LLMResult


RULES = (
    ("Top-priority Actions", ("stop", "halt", "cancel immediately")),
    ("Rescue", ("rescue", "safe place", "emergency")),
    ("Request Remote Control", ("remote", "underground mine")),
    ("Error Correction", ("mistake", "correct my")),
    ("Shopping Guide", ("buy", "shopping", "look good")),
    ("Location Guide", ("guide me", "navigate", "attraction", "directions")),
    ("Teaching", ("teach", "explain how", "learn")),
    ("Listen to Companion", ("story", "tell me a joke")),
    ("Watch with Companion", ("watch", "movie together")),
    ("Interact with Companion", ("chat", "companion", "talk with", "toast", "cheers")),
    ("Repair", ("repair", "fix the", "broken")),
    ("Check", ("inspect", "check the")),
    ("Replace", ("replace", "substitute")),
    ("Clean", ("clean", "wash")),
    ("Save Documents", ("save", "archive")),
    ("Share Documents", ("share", "send my")),
    ("Edit Documents", ("edit", "augment", "trim the")),
    ("Generate Documents", ("draw", "generate", "create", "compose", "produce")),
    ("Search/Recognize Physical Items", ("where is", "locate", "find the")),
    ("Search/Recognize Digital Information", ("search", "information", "weather", "nutrient", "compare", "recipe", "query")),
    ("Request Collaborator", ("bring", "pour", "cut", "grab", "turn", "switch", "pass", "carry")),
)


class MockLLMClient:
    backend = "mock"

    def generate(self, request: LLMRequest) -> LLMResult:
        handlers = {
            "intent": self._intent, "task": self._task, "agent": self._agent,
            "agent_flags": self._flags, "vision": self._vision,
        }
        if request.stage not in handlers:
            raise ValidationError("Unknown mock request stage")
        output = handlers[request.stage](request.payload)
        return LLMResult(json.dumps(output, ensure_ascii=False, allow_nan=False),
                         {"backend": "mock", "model": "deterministic-demo-v0.3", "synthetic": True})

    def _intent(self, payload: dict) -> dict:
        structured = payload["input"]
        visual = structured.get("visual", {})
        speech = structured.get("audio", {}).get("speech", {})
        report = structured.get("credibility_report", {})
        voice_reliable = report.get("voice", {}).get("credible", True)
        text = (structured.get("utterance") or speech.get("transcript") or "") if voice_reliable else ""
        lowered = text.casefold()
        candidates = []
        for label, keywords in RULES:
            if any(keyword in lowered for keyword in keywords):
                candidates = [{"specific_label": label, "description": text, "confidence": 0.8}]
                break
        target = visual.get("hand_target") or visual.get("eye_target")
        if not candidates and target and "wine" in target.casefold():
            candidates = [
                {"specific_label": "Request Collaborator", "description": "Pour wine into the glass", "confidence": 0.6},
                {"specific_label": "Interact with Companion", "description": "Toast with the user", "confidence": 0.5},
            ]
        if not candidates and structured.get("history"):
            previous = structured["history"][-1]["specific_label"]
            if "continue" in lowered:
                candidates = [{"specific_label": previous, "description": "Continue the previous activity", "confidence": 0.5}]
        complementary = bool(text and target and (not visual.get("eye_target") or visual.get("eye_target") == target))
        return {
            "candidates": candidates,
            "complementarity": "complementary" if complementary else "non_complementary",
            "ambiguity": "ambiguous" if len(candidates) > 1 else "unambiguous" if candidates else "unknown",
            "evidence_summary": "Deterministic lexical and context rules for software testing; not model reasoning.",
        }

    def _task(self, payload: dict) -> dict:
        intent = payload["selected_intent"]
        label = intent["specific_label"]
        description = intent["description"]
        lowered = description.casefold()
        capabilities = ["speech_io"]
        instruction = f"Respond to the selected request: {description}"
        if label == "Generate Documents":
            capabilities += ["text_io", "creativity"]
        elif label in {"Save Documents", "Share Documents", "Edit Documents"}:
            capabilities += ["text_io"]
        elif label == "Location Guide":
            capabilities += ["visual_expression", "navigation_guidance"]
        elif label in {"Interact with Companion", "Watch with Companion"}:
            capabilities += ["visual_expression"]
        elif label in {"Repair", "Check", "Replace", "Clean", "Rescue", "Request Remote Control"}:
            capabilities += ["vision_perception", "physical_interaction", "physical_movement"]
        elif label == "Request Collaborator" and not any(word in lowered for word in ("turn", "switch", "order online")):
            capabilities += ["vision_perception", "physical_interaction", "physical_movement"]
        structured = payload["input"]
        visual = structured.get("visual", {})
        target = visual.get("hand_target") or visual.get("eye_target")
        if payload.get("focus_instruction") and target:
            instruction += f"; ground the plan in the observed target: {target}"
        return {"goal": description, "steps": [
            {"step_id": 1, "instruction": instruction, "required_capabilities": capabilities},
        ]}

    def _flags(self, payload: dict) -> dict:
        capabilities = {capability for step in payload["task_plan"]["steps"] for capability in step["required_capabilities"]}
        return {
            "speech_expression": bool(capabilities & {"speech_io", "text_io"}),
            "visual_expression": bool(capabilities & {"visual_expression", "navigation_guidance"}),
            "physical_interaction": bool(capabilities & {"physical_interaction", "physical_movement"}),
            "creativity": "creativity" in capabilities,
        }

    def _agent(self, payload: dict) -> dict:
        capabilities = {capability for step in payload["task_plan"]["steps"] for capability in step["required_capabilities"]}
        satisfying = [
            agent for agent, available in payload["agent_capabilities"].items()
            if capabilities <= set(available)
        ]
        if satisfying:
            agent = min(satisfying, key=lambda name: (payload["relative_costs"][name], name))
        else:
            agent = "physical_agent"
        return {"agent_type": agent, "rationale": "Deterministic demonstration: capability coverage followed by relative cost."}

    def _vision(self, payload: dict) -> dict:
        return {"activity": None, "location": None, "objects": [], "eye_state": None,
                "eye_target": None, "hand_state": None, "hand_target": None, "scene_description": None}
