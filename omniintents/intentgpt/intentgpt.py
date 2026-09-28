"""Paper taxonomy inference with session-scoped deterministic memory."""

from __future__ import annotations

from ..ablations import prepare_ablation_input, project_structured
from ..config import OmniIntentsConfig
from ..llm.base import LLMClient, generate_json
from ..prompts import request_for
from ..types import Intent, IntentPrediction, MultimodalInput, PipelineTrace, StructuredText
from ..utils.validate import parse_intents
from .audio import AudioProcessor
from .credibility import CredibilityBasedAttentionShifter
from .memory import ContextualMemoryTracker, HistoryEntry
from .visual import VisualProcessor


class IntentGPT:
    def __init__(
        self, llm: LLMClient, config: OmniIntentsConfig | None = None, *,
        memory: ContextualMemoryTracker | None = None, audio_processor: AudioProcessor | None = None,
    ):
        self.llm = llm
        self.config = config or OmniIntentsConfig()
        self.memory = memory or ContextualMemoryTracker(self.config.max_history_entries)
        self.visual = VisualProcessor(llm, self.config)
        self.audio = audio_processor or AudioProcessor(self.config)
        self.credibility = CredibilityBasedAttentionShifter(
            self.config.brightness_credible_threshold, self.config.speech_confidence_credible_threshold,
        )

    def convert_to_structured_text(self, data: MultimodalInput, trace: PipelineTrace | None = None) -> StructuredText:
        prepared = prepare_ablation_input(data, self.config)
        visual = self.visual.process(prepared, trace)
        audio = self.audio.process(prepared)
        report = self.credibility.evaluate(visual, audio) if self.config.cbas_enabled else {}
        history = self.memory.previous(
            data.session_id, data.sequence_index, data.timestamp, self.config.history_rounds,
        )
        structured = StructuredText(
            utterance=prepared.utterance, visual=visual, audio=audio, credibility_report=report,
            history=history, history_summary=self.memory.summarize(history),
            source_mode="raw_images" if prepared.image_paths else "preextracted",
            image_paths=prepared.image_paths[:],
        )
        structured = project_structured(structured, self.config.removed_modalities, self.config.structured_ablation_mode)
        if trace is not None:
            trace.add("structured_input", structured.prompt_dict())
        return structured

    def predict(self, data: MultimodalInput, trace: PipelineTrace | None = None) -> IntentPrediction:
        structured = self.convert_to_structured_text(data, trace)
        request = request_for("intent", {"input": structured.prompt_dict()}, self.config)
        output = generate_json(self.llm, request, self.config, trace)
        candidates, complementarity, ambiguity = parse_intents(output)
        return IntentPrediction(
            sample_id=data.sample_id, session_id=data.session_id, sequence_index=data.sequence_index,
            timestamp=data.timestamp, structured_text=structured, candidates=candidates,
            complementarity=complementarity, ambiguity=ambiguity, backend=self.llm.backend,
            evidence_summary=output.get("evidence_summary", ""),
        )

    def remember(self, prediction: IntentPrediction, selected: Intent) -> None:
        if not self.config.history_rounds or self.config.removed_modalities or self.config.structured_ablation_mode:
            return
        context = prediction.structured_text.visual.get("context", {})
        self.memory.add_entry(prediction.session_id, HistoryEntry(
            prediction.sample_id, prediction.sequence_index, prediction.timestamp,
            context.get("location"), context.get("activity"), selected.specific_label,
        ))
