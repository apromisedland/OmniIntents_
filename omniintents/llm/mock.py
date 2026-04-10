from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .base import LLMResult
from ..utils.json_utils import dumps_pretty, extract_first_json


def _lower(x: str) -> str:
    return (x or "").lower()


def _contains_any(text: str, keywords) -> bool:
    t = _lower(text)
    return any(k in t for k in keywords)


def _safe_get(d: Dict[str, Any], *keys, default=None):
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
    return cur if cur is not None else default


@dataclass
class MockLLMClient:
    """Offline deterministic LLM replacement.

    This is designed to make the whole pipeline runnable **without keys**.

    Behavior:
    - Detects special markers in prompts to output JSON with a stable schema.
    - Uses simple heuristics based on utterance + (optionally) embedded structured JSON.
    """

    seed: int = 0

    def complete(self, prompt: str, *, temperature: float = 0.2, max_tokens: int = 800) -> LLMResult:
        if "INTENT_PREDICTION_JSON" in prompt:
            return LLMResult(text=dumps_pretty(self._predict_intent(prompt)))

        if "TASK_PLAN_JSON" in prompt:
            return LLMResult(text=dumps_pretty(self._plan_tasks(prompt)))

        if "AGENT_RECOMMENDATION_JSON" in prompt:
            return LLMResult(text=dumps_pretty(self._select_agent(prompt)))

        return LLMResult(text="{}")

    def vision_describe(self, image_path: str, prompt: str, *, max_tokens: int = 300) -> LLMResult:
        # No real vision; provide a placeholder caption.
        return LLMResult(text=f"A scene captured in {image_path}. (Mock vision caption)")

    # -------------------------
    # Mock behaviors (heuristic)
    # -------------------------
    def _extract_user_utterance(self, prompt: str) -> str:
        m = re.search(r"USER_UTTERANCE:\s*(.*)", prompt)
        return m.group(1).strip() if m else ""

    def _extract_structured_json(self, prompt: str) -> Dict[str, Any]:
        # The prompt includes 'STRUCTURED_TEXT_JSON:' or 'FILTERED_STRUCTURED_TEXT_JSON:'
        idx = prompt.find("STRUCTURED_TEXT_JSON:")
        if idx == -1:
            idx = prompt.find("FILTERED_STRUCTURED_TEXT_JSON:")
        if idx == -1:
            return {}
        chunk = prompt[idx:]
        parsed = extract_first_json(chunk)
        return parsed if isinstance(parsed, dict) else {}

    def _predict_intent(self, prompt: str) -> Dict[str, Any]:
        utter = self._extract_user_utterance(prompt)
        structured = self._extract_structured_json(prompt)
        ctx_loc = _safe_get(structured, "visual", "context", "location", default="") or ""
        objects = _safe_get(structured, "visual", "objects", default=[]) or []
        credibility = _safe_get(structured, "credibility_report", default={}) or {}
        vis_cred = bool(_safe_get(credibility, "visual", "credible", default=True))
        voice_cred = bool(_safe_get(credibility, "voice", "credible", default=True))

        # Canonical intents referenced in the excerpt and earlier text:
        # - Request Search
        # - Request Collaborator (navigation / manipulation help)
        # - Request Companion (social / emotional)
        # - Request Physical Help (direct physical execution)
        # - Unknown
        t = _lower(utter)

        # If voice is not credible and utterance is empty, rely on context.
        if (not voice_cred) and (not utter.strip()):
            if "library" in _lower(ctx_loc) or "book" in [str(o).lower() for o in objects]:
                return {
                    "label": "Request Search",
                    "description": "User likely wants to query information about an object in the environment.",
                    "confidence": 0.55,
                    "entities": {"context": ctx_loc, "objects": objects},
                    "requires_visual": True,
                    "requires_audio": False,
                    "requires_physical": False,
                }
            return {
                "label": "Unknown",
                "description": "Speech is unreliable and there is insufficient context; ask for clarification.",
                "confidence": 0.35,
                "entities": {"context": ctx_loc},
                "requires_visual": False,
                "requires_audio": False,
                "requires_physical": False,
            }

        if _contains_any(t, ["query", "search", "look up", "information", "info", "查", "查询", "信息"]):
            requires_visual = _contains_any(t, ["this", "这", "此"]) or ("book" in t) or ("library" in _lower(ctx_loc))
            return {
                "label": "Request Search",
                "description": "User wants to query / search for information.",
                "confidence": 0.78,
                "entities": {"query": utter, "context_location": ctx_loc, "objects": objects},
                "requires_visual": bool(requires_visual and vis_cred),
                "requires_audio": bool(voice_cred),
                "requires_physical": False,
            }

        if _contains_any(t, ["navigate", "go to", "find", "walk", "move", "去", "找", "带我", "导航"]):
            return {
                "label": "Request Collaborator",
                "description": "User requests assistance to navigate or find something in the environment.",
                "confidence": 0.72,
                "entities": {"goal": utter, "context_location": ctx_loc},
                "requires_visual": bool(vis_cred),
                "requires_audio": bool(voice_cred),
                "requires_physical": False,
            }

        if _contains_any(t, ["cut", "slice", "chop", "切", "拿", "抓", "倒水", "pour", "grab"]):
            return {
                "label": "Request Physical Help",
                "description": "User requests a physical action to be executed.",
                "confidence": 0.85,
                "entities": {"request": utter},
                "requires_visual": bool(vis_cred),
                "requires_audio": bool(voice_cred),
                "requires_physical": True,
            }

        if _contains_any(t, ["cheers", "toast", "开心", "陪我", "chat", "talk", "聊天"]):
            return {
                "label": "Request Companion",
                "description": "User requests social interaction/companionship.",
                "confidence": 0.7,
                "entities": {"utterance": utter},
                "requires_visual": False,
                "requires_audio": bool(voice_cred),
                "requires_physical": False,
            }

        return {
            "label": "Unknown",
            "description": "Unable to confidently infer the user's intent from the provided inputs.",
            "confidence": 0.4,
            "entities": {"utterance": utter, "context_location": ctx_loc, "objects": objects},
            "requires_visual": False,
            "requires_audio": False,
            "requires_physical": False,
        }

    def _plan_tasks(self, prompt: str) -> Dict[str, Any]:
        # Extract intent label
        m = re.search(r"INTENT_LABEL:\s*(.*)", prompt)
        label = m.group(1).strip() if m else ""

        if "Request Search" in label:
            return {
                "goal": "Find and present the requested information.",
                "steps": [
                    {
                        "step_id": 1,
                        "instruction": "Identify the referenced entity/object from visual context (title/label/appearance) and the utterance.",
                        "required_capabilities": ["vision_perception", "speech_io"],
                    },
                    {
                        "step_id": 2,
                        "instruction": "Formulate a precise digital query (e.g., book title + author) and choose the appropriate source (catalog/web).",
                        "required_capabilities": ["text_io"],
                    },
                    {
                        "step_id": 3,
                        "instruction": "Retrieve key facts and disambiguate results using context/history if needed.",
                        "required_capabilities": ["text_io"],
                    },
                    {
                        "step_id": 4,
                        "instruction": "Answer the user clearly and concisely; cite source if applicable.",
                        "required_capabilities": ["speech_io", "text_io"],
                    },
                ],
            }

        if "Request Collaborator" in label:
            return {
                "goal": "Assist the user in navigating/finding the target safely.",
                "steps": [
                    {
                        "step_id": 1,
                        "instruction": "Clarify the target destination/object and constraints (time, accessibility, safety).",
                        "required_capabilities": ["speech_io"],
                    },
                    {
                        "step_id": 2,
                        "instruction": "Perceive environment layout and locate obstacles/landmarks relevant to the path.",
                        "required_capabilities": ["vision_perception"],
                    },
                    {
                        "step_id": 3,
                        "instruction": "Provide step-by-step guidance; if possible, augment with visual cues (AR arrows/map).",
                        "required_capabilities": ["speech_io", "visual_expression"],
                    },
                ],
            }

        if "Request Physical Help" in label:
            return {
                "goal": "Execute the requested physical task safely.",
                "steps": [
                    {
                        "step_id": 1,
                        "instruction": "Perceive the workspace and locate the target objects/tools.",
                        "required_capabilities": ["vision_perception"],
                    },
                    {
                        "step_id": 2,
                        "instruction": "Plan a safe manipulation sequence (grasp/move/act) with collision avoidance.",
                        "required_capabilities": ["physical_interaction", "navigation"],
                    },
                    {
                        "step_id": 3,
                        "instruction": "Perform the action while monitoring safety constraints; stop if uncertain.",
                        "required_capabilities": ["physical_interaction", "vision_perception"],
                    },
                    {
                        "step_id": 4,
                        "instruction": "Confirm completion and ask if any further assistance is needed.",
                        "required_capabilities": ["speech_io"],
                    },
                ],
            }

        if "Request Companion" in label:
            return {
                "goal": "Engage in a helpful, context-aware conversation with the user.",
                "steps": [
                    {
                        "step_id": 1,
                        "instruction": "Respond empathetically and ask a light follow-up question to clarify what the user wants.",
                        "required_capabilities": ["speech_io"],
                    },
                    {"step_id": 2, "instruction": "Provide the requested conversation/help.", "required_capabilities": ["speech_io"]},
                ],
            }

        return {
            "goal": "Handle the user's request.",
            "steps": [
                {
                    "step_id": 1,
                    "instruction": "Ask a clarification question; request missing context or confirm the user's goal.",
                    "required_capabilities": ["speech_io"],
                }
            ],
        }

    def _select_agent(self, prompt: str) -> Dict[str, Any]:
        # Use keyword detection on required capabilities.
        p = _lower(prompt)
        if "physical_interaction" in p or "navigation" in p:
            return {"agent_type": "physical_robot", "rationale": "Task requires physical interaction/navigation."}
        if "vision_perception" in p and "visual_expression" in p:
            return {"agent_type": "ar_companion", "rationale": "Task benefits from both visual perception and visual expression."}
        if "vision_perception" in p:
            return {"agent_type": "multimodal_assistant", "rationale": "Task requires visual perception."}
        if "text_io" in p:
            return {"agent_type": "digital_assistant", "rationale": "Task is primarily digital text processing."}
        return {"agent_type": "voice_assistant", "rationale": "Cheapest option for speech-only tasks."}
