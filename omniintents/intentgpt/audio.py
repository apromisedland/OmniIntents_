from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np

from ..types import MultimodalInput
from ..utils.audio_utils import load_audio, voiced_rms_db, voiced_ratio


class SoundClassifier(Protocol):
    """Interface for YAMNet-like sound classifiers."""

    def classify_topk(self, samples: np.ndarray, sample_rate: int, k: int = 3) -> List[Tuple[str, float]]:
        ...


class SpeechTranscriber(Protocol):
    """Interface for a speech-to-text component (e.g., Google Speech API, Whisper)."""

    def transcribe(self, audio_path: str) -> Tuple[str, float]:
        ...


@dataclass
class HeuristicSoundClassifier:
    """Fallback sound classifier when YAMNet isn't available."""

    def classify_topk(self, samples: np.ndarray, sample_rate: int, k: int = 3) -> List[Tuple[str, float]]:
        energy = float(np.mean(np.square(samples))) if samples.size else 0.0
        if energy < 1e-6:
            return [("Silence", 0.95), ("Background noise", 0.1), ("Speech", 0.05)][:k]
        # A crude proxy: voiced ratio suggests speech-ish content
        return [("Background noise", 0.55), ("Speech", 0.45), ("Music", 0.2)][:k]


@dataclass
class AudioProcessor:
    """Extract auditory information into structured text fields (paper 6.1.1.2)."""

    frame_hz: int = 20
    voiced_rms_threshold: float = 0.01
    db_calibration: float = 0.0
    sound_event_confidence_threshold: float = 0.70

    sound_classifier: Optional[SoundClassifier] = None
    speech_transcriber: Optional[SpeechTranscriber] = None

    def __post_init__(self) -> None:
        if self.sound_classifier is None:
            self.sound_classifier = HeuristicSoundClassifier()

    def process(self, mm: MultimodalInput) -> Dict[str, object]:
        # --- Speech transcription ---
        transcript = mm.speech_transcript
        speech_conf = mm.speech_confidence

        if transcript is None and mm.audio_path and self.speech_transcriber is not None:
            try:
                transcript, speech_conf = self.speech_transcriber.transcribe(mm.audio_path)
            except Exception:
                transcript, speech_conf = "", 0.0

        transcript = transcript or ""
        speech_conf = float(speech_conf) if speech_conf is not None else 0.0

        # --- Volume & sound classification ---
        volume_db: Optional[float] = None
        vratio: Optional[float] = None
        topk: List[Tuple[str, float]] = []

        if mm.audio_path:
            try:
                audio = load_audio(mm.audio_path)
                volume_db = voiced_rms_db(
                    audio.samples,
                    audio.sample_rate,
                    frame_hz=self.frame_hz,
                    voiced_rms_threshold=self.voiced_rms_threshold,
                    calibration=self.db_calibration,
                )
                vratio = voiced_ratio(
                    audio.samples,
                    audio.sample_rate,
                    frame_hz=self.frame_hz,
                    voiced_rms_threshold=self.voiced_rms_threshold,
                )
                topk = self.sound_classifier.classify_topk(audio.samples, audio.sample_rate, k=3) if self.sound_classifier else []
            except Exception:
                volume_db = None
                vratio = None
                topk = []

        likely_event = ""
        if topk:
            lab, conf = topk[0]
            if conf >= self.sound_event_confidence_threshold:
                likely_event = lab

        audio_text: Dict[str, object] = {
            "volume": {
                "decibels": None if volume_db is None else float(volume_db),
                "voiced_ratio": None if vratio is None else float(vratio),
                "text": (
                    "The environmental volume is approximately unknown decibels."
                    if volume_db is None
                    else f"The environmental volume is approximately {volume_db:.2f} decibels."
                ),
            },
            "sound_classification": {
                "top3": [{"label": lab, "confidence": float(conf)} for lab, conf in topk],
                "likely_event": likely_event,
            },
            "speech": {
                "transcript": transcript,
                "confidence": float(speech_conf),
                "text": f"The current speech recognition result is {transcript!r}, with a confidence level of {speech_conf:.2f}.",
            },
        }
        return audio_text
