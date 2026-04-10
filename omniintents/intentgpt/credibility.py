from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class CredibilityBasedAttentionShifter:
    """Credibility-Based Attention Shifter (paper 6.1.1.3).

    The excerpt explicitly mentions two credibility checks:
    - Visual credibility: brightness threshold (>= 0.15 credible)
    - Voice credibility: speech recognition confidence threshold (>= 0.70 credible)

    This enhanced implementation also:
    - handles missing modalities (no image/audio) as 'incredible' when configured
    - emits explicit reasons + an attention hint for downstream prompts
    """

    brightness_credible_threshold: float = 0.15
    speech_confidence_credible_threshold: float = 0.70
    missing_visual_is_incredible: bool = True
    missing_audio_is_incredible: bool = True

    def evaluate(self, visual: Dict[str, Any], audio: Dict[str, Any]) -> Dict[str, Any]:
        # --- Visual credibility ---
        brightness_val = self._get_float(visual, ("brightness", "value"))
        has_visual_signal = brightness_val is not None or bool(visual.get("scene_description")) or bool(visual.get("objects"))

        if not has_visual_signal and self.missing_visual_is_incredible:
            visual_credible = False
            visual_reason = "No visual signal provided."
            brightness = None
        else:
            brightness = float(brightness_val or 0.0)
            visual_credible = brightness >= self.brightness_credible_threshold
            visual_reason = (
                f"Brightness {brightness:.3f} >= {self.brightness_credible_threshold:.2f}."
                if visual_credible
                else f"Brightness {brightness:.3f} < {self.brightness_credible_threshold:.2f} (too dark)."
            )

        # --- Voice credibility ---
        speech_conf = self._get_float(audio, ("speech", "confidence"))
        transcript = self._get_str(audio, ("speech", "transcript")) or ""
        has_voice_signal = bool(transcript.strip()) or (speech_conf is not None)

        if not has_voice_signal and self.missing_audio_is_incredible:
            voice_credible = False
            voice_reason = "No speech signal/transcript provided."
            speech_conf_val = None
        else:
            speech_conf_val = float(speech_conf or 0.0)
            voice_credible = speech_conf_val >= self.speech_confidence_credible_threshold
            voice_reason = (
                f"Speech confidence {speech_conf_val:.2f} >= {self.speech_confidence_credible_threshold:.2f}."
                if voice_credible
                else f"Speech confidence {speech_conf_val:.2f} < {self.speech_confidence_credible_threshold:.2f}."
            )

        report_text = f"Visual {'credible' if visual_credible else 'incredible'}; Voice {'credible' if voice_credible else 'incredible'}."  # noqa: E501

        if not visual_credible and voice_credible:
            hint = "Prioritize auditory/speech information over visual details due to low visual credibility."
        elif visual_credible and not voice_credible:
            hint = "Prioritize visual information over speech due to low speech recognition credibility."
        elif not visual_credible and not voice_credible:
            hint = "Both visual and speech signals are unreliable. Ask clarifying questions and rely on context/history."
        else:
            hint = "Both visual and speech signals are credible. Use intermodal complementarity to resolve ambiguity."

        return {
            "visual": {"credible": bool(visual_credible), "brightness": brightness, "reason": visual_reason},
            "voice": {"credible": bool(voice_credible), "speech_confidence": speech_conf_val, "reason": voice_reason},
            "report": report_text,
            "attention_hint": hint,
        }

    @staticmethod
    def _get_float(d: Dict[str, Any], path) -> Optional[float]:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        try:
            return float(cur)
        except Exception:
            return None

    @staticmethod
    def _get_str(d: Dict[str, Any], path) -> Optional[str]:
        cur: Any = d
        for k in path:
            if not isinstance(cur, dict):
                return None
            cur = cur.get(k)
        if cur is None:
            return None
        return str(cur)
