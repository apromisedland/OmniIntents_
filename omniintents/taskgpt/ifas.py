from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from ..types import Intent, StructuredText


@dataclass
class IntentFocusedAttentionShifter:
    """Intent-Focused Attention Shifter (IFAS) (paper 6.2.1).

    Goal:
    - re-allocate TaskGPT's attention toward intent-relevant details
    - reduce over-focus on irrelevant details in complex scenes
    """

    def shift(self, *, structured: StructuredText, intent: Intent) -> Tuple[Dict[str, Any], str]:
        label = (intent.label or "").lower()

        credibility = structured.credibility_report or {}
        hint = str(credibility.get("attention_hint", "")) if isinstance(credibility, dict) else ""

        # Default: keep everything (but still provide generic instruction)
        filtered: Dict[str, Any] = structured.to_json_dict()

        if "search" in label or "query" in label:
            focus = (
                "Focus on identifying the referenced object/entity (e.g., book title/author) "
                "from visual context (objects, scene description, interaction target) and the utterance. "
                "Ignore unrelated scene details. "
            )
            if hint:
                focus += f"Credibility hint: {hint}"

            filtered = {
                "visual": {
                    "context": structured.visual.get("context", {}),
                    "objects": structured.visual.get("objects", []),
                    "interaction_target": structured.visual.get("interaction_target", ""),
                    "scene_description": structured.visual.get("scene_description", ""),
                },
                "audio": {"speech": structured.audio.get("speech", {})},
                "credibility_report": structured.credibility_report,
                "history_summary": structured.history_summary,
                "history_retrieval": structured.history_retrieval,
            }
            return filtered, focus

        if "collaborator" in label or "navigate" in label or "transport" in label or "warehouse" in label:
            focus = (
                "Focus on spatially relevant information (layout clues, targets, obstacles) and the user's constraints. "
                "Prioritize path-critical details to avoid collisions/delays. "
            )
            if hint:
                focus += f"Credibility hint: {hint}"
            filtered = {
                "visual": {
                    "brightness": structured.visual.get("brightness", {}),
                    "context": structured.visual.get("context", {}),
                    "objects": structured.visual.get("objects", []),
                    "interaction_target": structured.visual.get("interaction_target", ""),
                    "scene_description": structured.visual.get("scene_description", ""),
                },
                "audio": {"speech": structured.audio.get("speech", {})},
                "credibility_report": structured.credibility_report,
                "history_summary": structured.history_summary,
                "history_retrieval": structured.history_retrieval,
            }
            return filtered, focus

        if "physical" in label or intent.requires_physical:
            focus = (
                "Focus on the physical task's safety-critical details: tools, target objects, workspace constraints, "
                "and a safe manipulation sequence. "
            )
            if hint:
                focus += f"Credibility hint: {hint}"
            filtered = {
                "visual": {
                    "context": structured.visual.get("context", {}),
                    "objects": structured.visual.get("objects", []),
                    "interaction_target": structured.visual.get("interaction_target", ""),
                    "scene_description": structured.visual.get("scene_description", ""),
                },
                "audio": {"speech": structured.audio.get("speech", {})},
                "credibility_report": structured.credibility_report,
                "history_summary": structured.history_summary,
            }
            return filtered, focus

        if "companion" in label:
            focus = (
                "Focus on the user's expressed emotion and conversational intent. "
                "Respond naturally and ask small clarifying questions if needed. "
            )
            if hint:
                focus += f"Credibility hint: {hint}"
            filtered = {
                "audio": {"speech": structured.audio.get("speech", {})},
                "credibility_report": structured.credibility_report,
                "history_summary": structured.history_summary,
            }
            return filtered, focus

        focus = (
            "Focus on details necessary to execute the user's intent. "
            "Use the credibility report to prioritize reliable modalities. "
        )
        if hint:
            focus += f"Credibility hint: {hint}"

        return filtered, focus
