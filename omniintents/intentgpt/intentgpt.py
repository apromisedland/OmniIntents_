from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ..config import OmniIntentsConfig
from ..llm.base import LLMClient
from ..types import Intent, MultimodalInput, PipelineTrace, StructuredText
from ..utils.json_utils import extract_first_json_object, truncate
from ..utils.validate import validate_intent_json
from .audio import AudioProcessor
from .credibility import CredibilityBasedAttentionShifter
from .cot import CoTPromptBuilder
from .heuristics import IntentHeuristicPredictor
from .memory import ContextualMemoryTracker
from .taxonomy import IntentTaxonomy
from .visual import VisualProcessor


@dataclass
class IntentGPT:
    """IntentGPT module (paper 6.1).

    Steps:
    1) Data Conversion: multimodal -> structured text (visual + audio + credibility + history)
    2) Intermediate Reasoning: CoT prompt based on examples (few-shot)
    3) Intent Prediction: LLM outputs intent JSON (validated) else heuristic fallback
    """

    llm: LLMClient
    config: OmniIntentsConfig = OmniIntentsConfig()

    memory: Optional[ContextualMemoryTracker] = None
    cot: Optional[CoTPromptBuilder] = None
    taxonomy: Optional[IntentTaxonomy] = None
    heuristic: Optional[IntentHeuristicPredictor] = None

    visual_processor: Optional[VisualProcessor] = None
    audio_processor: Optional[AudioProcessor] = None
    credibility_shifter: Optional[CredibilityBasedAttentionShifter] = None

    def __post_init__(self) -> None:
        self.taxonomy = self.taxonomy or IntentTaxonomy()
        self.heuristic = self.heuristic or IntentHeuristicPredictor(self.taxonomy)

        if self.memory is None:
            self.memory = ContextualMemoryTracker(
                max_entries=self.config.max_history_entries,
                retrieval_top_k=self.config.memory_retrieval_top_k,
            )

        if self.cot is None:
            self.cot = CoTPromptBuilder(taxonomy=self.taxonomy)

        if self.visual_processor is None:
            self.visual_processor = VisualProcessor(llm=self.llm)

        if self.audio_processor is None:
            self.audio_processor = AudioProcessor(
                frame_hz=self.config.audio_frame_hz,
                voiced_rms_threshold=self.config.voiced_rms_threshold,
                db_calibration=self.config.db_calibration,
                sound_event_confidence_threshold=self.config.sound_event_confidence_threshold,
            )

        if self.credibility_shifter is None:
            self.credibility_shifter = CredibilityBasedAttentionShifter(
                brightness_credible_threshold=self.config.brightness_credible_threshold,
                speech_confidence_credible_threshold=self.config.speech_confidence_credible_threshold,
                missing_visual_is_incredible=self.config.missing_visual_is_incredible,
                missing_audio_is_incredible=self.config.missing_audio_is_incredible,
            )

    def convert_to_structured_text(self, mm: MultimodalInput, trace: Optional[PipelineTrace] = None) -> StructuredText:
        visual = self.visual_processor.process(mm) if self.visual_processor else {}
        audio = self.audio_processor.process(mm) if self.audio_processor else {}
        credibility = self.credibility_shifter.evaluate(visual, audio) if self.credibility_shifter else {}

        history_summary = self.memory.summarize() if self.memory else ""
        history_retrieval = self.memory.retrieve(mm.utterance) if self.memory and mm.utterance else []

        structured = StructuredText(
            visual=visual,
            audio=audio,
            credibility_report=credibility,
            history_summary=history_summary,
            history_retrieval=history_retrieval,
            raw={"timestamp": mm.timestamp.isoformat()},
        )

        if trace is not None and self.config.enable_trace:
            trace.add("intentgpt.structured_text", structured.to_json_dict())

        return structured

    def predict_intent(self, mm: MultimodalInput, trace: Optional[PipelineTrace] = None) -> Tuple[StructuredText, Intent]:
        structured = self.convert_to_structured_text(mm, trace=trace)

        prompt = self.cot.build_intent_prompt(structured=structured, user_utterance=mm.utterance)  # type: ignore[union-attr]
        if trace is not None and self.config.enable_trace:
            trace.add("intentgpt.prompt", {"prompt": truncate(prompt, self.config.max_prompt_chars_in_trace)})

        llm_result = self.llm.complete(prompt, temperature=0.2, max_tokens=700)
        if trace is not None and self.config.enable_trace:
            trace.add("intentgpt.llm_output", {"text": truncate(llm_result.text, self.config.max_prompt_chars_in_trace)})

        data = extract_first_json_object(llm_result.text) or {}
        ok, errors = validate_intent_json(data)

        if not ok:
            # fallback
            intent = self.heuristic.predict(user_utterance=mm.utterance, structured=structured)  # type: ignore[union-attr]
            if trace is not None and self.config.enable_trace:
                trace.add("intentgpt.fallback", {"errors": errors, "intent": intent.to_json_dict()})
        else:
            intent = Intent(
                label=str(data.get("label", "Unknown")),
                description=str(data.get("description", "")),
                confidence=float(data.get("confidence", 0.5)),
                entities=dict(data.get("entities", {}) or {}),
                requires_visual=bool(data.get("requires_visual", False)),
                requires_audio=bool(data.get("requires_audio", False)),
                requires_physical=bool(data.get("requires_physical", False)),
            )

        # Update memory tracker with context/location/activity if available
        if self.memory is not None:
            ctx = structured.visual.get("context") or {}
            loc = str(ctx.get("location") or "unknown") if isinstance(ctx, dict) else "unknown"
            act = str(ctx.get("activity") or "unknown") if isinstance(ctx, dict) else "unknown"
            self.memory.add_entry(
                timestamp=mm.timestamp,
                location=loc,
                activity=act,
                intent_label=intent.label,
                utterance=mm.utterance,
            )

        if trace is not None and self.config.enable_trace:
            trace.add("intentgpt.intent", intent.to_json_dict())

        return structured, intent
