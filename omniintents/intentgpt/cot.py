from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..types import StructuredText
from ..utils.json_utils import dumps_pretty
from .taxonomy import IntentTaxonomy


@dataclass
class CoTPromptBuilder:
    """Few-shot + CoT prompt builder for IntentGPT (paper 6.1.3).

    The paper's Appendix A.1 template isn't provided in the excerpt,
    so we implement a practical approximation that:
    - includes intent taxonomy
    - includes diary-study-like few-shot examples
    - includes credibility report + attention hint
    - asks the model to output STRICT JSON only
    """

    examples_path: Optional[str] = None
    max_examples: int = 3
    taxonomy: Optional[IntentTaxonomy] = None

    def __post_init__(self) -> None:
        self.taxonomy = self.taxonomy or IntentTaxonomy()

    def load_examples(self) -> List[Dict[str, Any]]:
        if self.examples_path:
            p = Path(self.examples_path)
        else:
            here = Path(__file__).resolve().parent.parent
            p = here / "data" / "cot_examples.json"
        if not p.exists():
            return []
        return json.loads(p.read_text(encoding="utf-8"))

    def build_intent_prompt(self, *, structured: StructuredText, user_utterance: str) -> str:
        examples = self.load_examples()[: self.max_examples]
        taxonomy_text = self.taxonomy.to_prompt_list() if self.taxonomy else ""

        instruction = (
            "You are IntentGPT. Infer the user's intent by reasoning over multimodal inputs "
            "(vision, audio, context, history). Use intermodal complementarity to resolve ambiguity "
            "(e.g., 'this' can be disambiguated by gaze/pointing). "
            "Use the credibility report to down-weight unreliable modalities. "
            "Return ONLY a JSON object (no markdown) with the required keys."
        )

        few_shot_blocks: List[str] = []
        for ex in examples:
            few_shot_blocks.append(
                "\n".join(
                    [
                        "### FEW-SHOT EXAMPLE",
                        f"Scenario: {ex.get('scenario','')}",
                        "Structured hint JSON:",
                        dumps_pretty(ex.get("structured_hint", {})),
                        f"Participant reasoning: {ex.get('participant_reasoning','')}",
                        "Expected intent JSON:",
                        dumps_pretty(ex.get("ground_truth_intent", {})),
                        f"CoT (short): {ex.get('cot_example','')}",
                    ]
                )
            )
        few_shot = "\n\n".join(few_shot_blocks)

        structured_json = dumps_pretty(structured.to_json_dict())

        # Marker token allows MockLLM routing.
        prompt = "\n\n".join(
            [
                instruction,
                "INTENT_PREDICTION_JSON",
                "### INTENT TAXONOMY",
                taxonomy_text,
                few_shot,
                "### INPUT",
                f"USER_UTTERANCE: {user_utterance}",
                "STRUCTURED_TEXT_JSON:",
                structured_json,
                "### OUTPUT FORMAT",
                "Return JSON with keys:",
                "- label: string (one of taxonomy labels when possible)",
                "- description: string",
                "- confidence: number in [0,1]",
                "- entities: object",
                "- requires_visual: boolean",
                "- requires_audio: boolean",
                "- requires_physical: boolean",
            ]
        )
        return prompt
