"""Audio measurements and explicit optional service adapters."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from ..config import OmniIntentsConfig
from ..errors import ProviderError, ValidationError
from ..types import MultimodalInput, finite_number
from ..utils.audio_utils import load_audio, voiced_ratio, voiced_rms_db


class SoundClassifier(Protocol):
    def classify_topk(self, samples: np.ndarray, sample_rate: int, k: int = 3) -> list[tuple[str, float]]:
        ...


class SpeechTranscriber(Protocol):
    def transcribe(self, audio_path: str) -> tuple[str, float | None]:
        ...


class GoogleSpeechTranscriber:
    def __init__(self, language_code: str = "en-US", client: Any = None):
        try:
            from google.cloud import speech
        except ImportError as error:
            raise ValidationError("Install the google-speech extra to use Google Speech") from error
        self.speech = speech
        try:
            self.client = client or speech.SpeechClient()
        except Exception as error:
            raise ProviderError("speech_credentials", "Google Speech client initialization failed; check ADC configuration") from error
        self.language_code = language_code

    def transcribe(self, audio_path: str) -> tuple[str, float | None]:
        try:
            response = self.client.recognize(
                config=self.speech.RecognitionConfig(language_code=self.language_code),
                audio=self.speech.RecognitionAudio(content=Path(audio_path).read_bytes()),
                timeout=60,
                retry=None,
            )
        except OSError as error:
            raise ValidationError("Unreadable speech audio input") from error
        except Exception as error:
            raise ProviderError("speech_service", "Google Speech recognition failed") from error
        alternatives = [result.alternatives[0] for result in response.results if result.alternatives]
        if not alternatives:
            return "", None
        weights = [max(1, len(item.transcript.split())) for item in alternatives]
        confidence = sum(item.confidence * weight for item, weight in zip(alternatives, weights)) / sum(weights)
        return " ".join(item.transcript for item in alternatives), float(confidence)


class YamNetClassifier:
    def __init__(self, model_path: str, labels_path: str, model: Any = None):
        with Path(labels_path).open(encoding="utf-8", newline="") as stream:
            self.labels = [row["display_name"] for row in csv.DictReader(stream)]
        if len(self.labels) != 521:
            raise ValidationError("YAMNet class map must contain 521 labels")
        if model is None:
            if not Path(model_path).is_dir():
                raise ValidationError("Supply a local YAMNet SavedModel directory")
            try:
                import tensorflow_hub
            except ImportError as error:
                raise ValidationError("Install the yamnet extra to use YAMNet") from error
            try:
                model = tensorflow_hub.load(model_path)
            except Exception as error:
                raise ProviderError("sound_model", "Cannot load the local YAMNet model") from error
        self.model = model

    def classify_topk(self, samples: np.ndarray, sample_rate: int, k: int = 3) -> list[tuple[str, float]]:
        if sample_rate != 16000:
            raise ValidationError("YAMNet requires pre-resampled 16 kHz mono audio")
        if samples.size == 0:
            return []
        try:
            scores, _, _ = self.model(np.asarray(samples, dtype=np.float32))
            mean_scores = np.asarray(scores).mean(axis=0)
        except Exception as error:
            raise ProviderError("sound_model", "YAMNet inference failed") from error
        if mean_scores.shape != (521,) or not np.all(np.isfinite(mean_scores)):
            raise ProviderError("sound_model", "Invalid YAMNet scores")
        indices = np.argsort(-mean_scores, kind="stable")[:k]
        return [(self.labels[index], float(mean_scores[index])) for index in indices]


class AudioProcessor:
    def __init__(
        self, config: OmniIntentsConfig | None = None, *,
        sound_classifier: SoundClassifier | None = None, speech_transcriber: SpeechTranscriber | None = None,
    ):
        self.config = config or OmniIntentsConfig()
        self.sound_classifier = sound_classifier
        self.speech_transcriber = speech_transcriber

    def process(self, data: MultimodalInput) -> dict[str, Any]:
        transcript, confidence = data.speech_transcript, data.speech_confidence
        source = "preextracted" if transcript is not None else "missing"
        volume, ratio = data.volume_db, None
        events = list(data.sound_events or [])
        if data.audio_path:
            audio = load_audio(data.audio_path)
            if transcript is None:
                if self.speech_transcriber is None:
                    raise ProviderError("speech_unavailable", "Raw audio requires a speech adapter or supplied transcript")
                transcript, confidence = self.speech_transcriber.transcribe(data.audio_path)
                source = "google_speech_or_injected_adapter"
            if data.sound_events is None:
                if self.sound_classifier is None:
                    raise ProviderError("sound_unavailable", "Raw audio requires a sound adapter or supplied sound_events")
                events = [{"label": label, "confidence": score} for label, score in
                          self.sound_classifier.classify_topk(audio.samples, audio.sample_rate)]
            if volume is None:
                volume = voiced_rms_db(
                    audio.samples, audio.sample_rate, self.config.audio_frame_hz,
                    self.config.voiced_rms_threshold, self.config.db_calibration,
                )
            ratio = voiced_ratio(
                audio.samples, audio.sample_rate, self.config.audio_frame_hz, self.config.voiced_rms_threshold,
            )
        if transcript is None and data.utterance:
            transcript, source = data.utterance, "typed_text"
        if confidence is not None:
            finite_number(confidence, "speech confidence", 0, 1)
        for event in events:
            finite_number(event["confidence"], "sound confidence", 0, 1)
        events.sort(key=lambda event: -event["confidence"])
        events = events[:3]
        likely = events[0]["label"] if events and events[0]["confidence"] > self.config.sound_event_confidence_threshold else None
        return {
            "status": "present" if transcript or events or data.audio_path else "missing",
            "volume": {"value": volume, "unit": "preextracted_dB_unspecified" if data.volume_db is not None else "dBFS+calibration", "calibration": self.config.db_calibration,
                       "voiced_ratio": ratio},
            "sound_classification": {"top3": events, "likely_event": likely},
            "speech": {"transcript": transcript, "confidence": confidence, "source": source},
            "voice_target": data.voice_target,
        }
