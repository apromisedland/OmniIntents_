"""Appendix A.1 thresholds; equality is conservatively unreliable."""

from dataclasses import dataclass
from typing import Any


@dataclass
class CredibilityBasedAttentionShifter:
    brightness_credible_threshold: float = 0.2
    speech_confidence_credible_threshold: float = 0.8

    def evaluate(self, visual: dict[str, Any], audio: dict[str, Any]) -> dict[str, Any]:
        brightness = visual.get("brightness")
        speech = audio.get("speech", {})
        confidence = speech.get("confidence")
        visual_status = "missing" if brightness is None else (
            "credible" if brightness > self.brightness_credible_threshold else "low_credibility"
        )
        if speech.get("source") == "typed_text" and speech.get("transcript"):
            voice_status = "typed_text"
        elif not speech.get("transcript"):
            voice_status = "missing"
        elif confidence is None:
            voice_status = "unknown_confidence"
        else:
            voice_status = "credible" if confidence > self.speech_confidence_credible_threshold else "low_credibility"
        visual_credible = visual_status == "credible"
        voice_credible = voice_status in {"credible", "typed_text"}
        if visual_credible and not voice_credible:
            hint = "Prioritize visual evidence; speech is missing or unreliable."
        elif voice_credible and not visual_credible:
            hint = "Prioritize language evidence; visual reliability is missing or low."
        elif visual_credible and voice_credible:
            hint = "Combine modalities, keeping conflicting targets separate."
        else:
            hint = "Evidence is insufficient or unreliable; return alternatives or request clarification."
        return {
            "visual": {"status": visual_status, "credible": visual_credible, "brightness": brightness},
            "voice": {"status": voice_status, "credible": voice_credible, "confidence": confidence},
            "report": f"Visual {'credible' if visual_credible else 'incredible'}; Voice {'credible' if voice_credible else 'incredible'}.",
            "attention_hint": hint,
        }
