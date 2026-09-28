"""Conservative structured-field projections for modality experiments."""

from __future__ import annotations

import copy
import re
from dataclasses import replace
from typing import Any

from .config import OmniIntentsConfig
from .types import MultimodalInput, StructuredText


def prepare_ablation_input(data: MultimodalInput, config: OmniIntentsConfig) -> MultimodalInput:
    if not config.removed_modalities and not config.structured_ablation_mode:
        return data
    return replace(data, image_paths=[], audio_path=None, scene_description=None)


def project_structured(structured: StructuredText, removed: tuple[str, ...], conservative: bool = False) -> StructuredText:
    result = copy.deepcopy(structured)
    if not removed and not conservative:
        return result
    result.image_paths = []
    result.history = []
    result.history_summary = ""
    result.visual.pop("scene_description", None)
    result.source_mode = "preextracted_ablation"
    erased: list[str] = []
    def erase(container: dict[str, Any], key: str) -> None:
        value = container.pop(key, None)
        def collect(item: Any) -> None:
            if isinstance(item, str) and len(item.strip()) >= 3 and item.casefold() not in {"none", "unknown", "missing"}:
                erased.append(item)
            elif isinstance(item, dict):
                for nested in item.values():
                    collect(nested)
            elif isinstance(item, list):
                for nested in item:
                    collect(nested)
        collect(value)
    for modality in removed:
        if modality == "context":
            erase(result.visual, "context")
        elif modality == "objects":
            erase(result.visual, "objects")
        elif modality == "hand_eye":
            for key in ("hand_state", "hand_target", "eye_state", "eye_target"):
                erase(result.visual, key)
        elif modality == "speech":
            erase(result.audio, "speech")
            erase(result.audio, "voice_target")
            result.utterance = ""
            result.credibility_report.pop("voice", None)
        elif modality == "audio_labels":
            erase(result.audio, "sound_classification")
        elif modality == "targets":
            for key in ("hand_target", "eye_target"):
                erase(result.visual, key)
            erase(result.audio, "voice_target")
    result.credibility_report.pop("report", None)
    result.credibility_report.pop("attention_hint", None)
    def scrub(value: Any) -> Any:
        if isinstance(value, str):
            for token in sorted(set(erased), key=len, reverse=True):
                value = re.sub(re.escape(token), "[removed]", value, flags=re.IGNORECASE)
            return value
        if isinstance(value, dict):
            return {key: scrub(item) for key, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value
    result.visual = scrub(result.visual)
    result.audio = scrub(result.audio)
    result.utterance = scrub(result.utterance)
    return result


def ablation_configs(base: OmniIntentsConfig) -> dict[str, OmniIntentsConfig]:
    variants = {"baseline": replace(base)}
    variants["modality_reference"] = replace(base, removed_modalities=(), structured_ablation_mode=True)
    for modality in ("context", "objects", "hand_eye", "speech", "targets", "audio_labels"):
        variants[f"without_{modality}"] = replace(base, removed_modalities=(modality,), structured_ablation_mode=True)
    variants["without_cbas"] = replace(base, cbas_enabled=False)
    variants["without_ifas"] = replace(base, ifas_enabled=False)
    for rounds in range(6):
        variants[f"history_{rounds}"] = replace(base, history_rounds=rounds, max_history_entries=5)
    variants["agent_explicit"] = replace(base, agent_strategy="explicit")
    variants["agent_implicit"] = replace(base, agent_strategy="implicit")
    return variants
