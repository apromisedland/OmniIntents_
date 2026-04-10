from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..types import Intent, StructuredText
from .taxonomy import IntentTaxonomy


def _lower(x: str) -> str:
    return (x or "").lower()


def _contains_any(text: str, keywords: List[str]) -> bool:
    t = _lower(text)
    return any(k.lower() in t for k in keywords)


@dataclass
class IntentHeuristicPredictor:
    """Rule-based fallback intent predictor.

    Used when:
    - LLM output is invalid / missing required fields
    - You want deterministic behavior for offline runs
    """

    taxonomy: IntentTaxonomy

    def predict(self, *, user_utterance: str, structured: StructuredText) -> Intent:
        utter = user_utterance or ""
        t = _lower(utter)

        ctx_loc = str(((structured.visual.get("context") or {}).get("location")) or "")
        objects = structured.visual.get("objects") or []
        obj_lower = [str(o).lower() for o in objects] if isinstance(objects, list) else []

        cred = structured.credibility_report or {}
        vis_cred = bool(((cred.get("visual") or {}).get("credible")) if isinstance(cred, dict) else True)
        voice_cred = bool(((cred.get("voice") or {}).get("credible")) if isinstance(cred, dict) else True)

        # Keyword matching in priority order
        for it in self.taxonomy.intents:
            if it.label == "Unknown":
                continue
            if it.keywords and _contains_any(t, it.keywords):
                return Intent(
                    label=it.label,
                    description=it.description,
                    confidence=0.65,
                    entities={"utterance": utter, "context_location": ctx_loc, "objects": objects},
                    requires_visual=bool(it.requires_visual and vis_cred),
                    requires_audio=bool(it.requires_audio and voice_cred),
                    requires_physical=it.requires_physical,
                )

        # Context-based heuristic: library+book -> search
        if ("library" in _lower(ctx_loc) or "书" in ctx_loc) and ("book" in obj_lower or "书" in obj_lower):
            it = self.taxonomy.get("Request Search")
            if it:
                return Intent(
                    label=it.label,
                    description=it.description,
                    confidence=0.55,
                    entities={"context_location": ctx_loc, "objects": objects},
                    requires_visual=bool(it.requires_visual and vis_cred),
                    requires_audio=bool(it.requires_audio and voice_cred),
                    requires_physical=it.requires_physical,
                )

        # If voice not credible and no utterance, ask clarification
        if (not voice_cred) and (not utter.strip()):
            return Intent(
                label="Unknown",
                description="Speech is unreliable and no utterance content is available; ask clarification.",
                confidence=0.35,
                entities={"context_location": ctx_loc},
                requires_visual=False,
                requires_audio=False,
                requires_physical=False,
            )

        # Default
        return Intent(
            label="Unknown",
            description="Unable to infer intent; ask clarification.",
            confidence=0.4,
            entities={"utterance": utter, "context_location": ctx_loc, "objects": objects},
            requires_visual=False,
            requires_audio=False,
            requires_physical=False,
        )
