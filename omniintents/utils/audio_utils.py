"""Audio measurement; uncalibrated levels are dBFS, not physical SPL."""

from __future__ import annotations

import math
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..errors import ValidationError


@dataclass
class AudioData:
    samples: np.ndarray
    sample_rate: int


def load_audio(path: str) -> AudioData:
    try:
        if Path(path).suffix.lower() == ".wav":
            with wave.open(str(path), "rb") as stream:
                width = stream.getsampwidth()
                channels = stream.getnchannels()
                rate = stream.getframerate()
                raw = stream.readframes(stream.getnframes())
            if width == 1:
                samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128) / 128
            elif width in (2, 4):
                samples = np.frombuffer(raw, dtype=f"<i{width}").astype(np.float32) / (2 ** (8 * width - 1))
            else:
                raise ValidationError("Use 8-, 16-, or 32-bit PCM WAV, or install the audio extra for other formats")
            return AudioData(samples.reshape(-1, channels).mean(axis=1), rate)
        try:
            import soundfile
        except ImportError as error:
            raise ValidationError("Non-WAV audio requires the audio extra") from error
        samples, rate = soundfile.read(path, dtype="float32", always_2d=True)
        return AudioData(samples.mean(axis=1), rate)
    except (OSError, wave.Error, ValueError) as error:
        raise ValidationError("Unreadable or unsupported audio input") from error


def rms(samples: np.ndarray) -> float:
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValidationError("Audio must be a finite mono waveform")
    return float(np.sqrt(np.mean(values ** 2))) if values.size else 0.0


def rms_to_db(rms_value: float, calibration: float = 0.0) -> float:
    if not math.isfinite(rms_value) or rms_value < 0 or not math.isfinite(calibration):
        raise ValidationError("Invalid RMS or calibration")
    return 20 * math.log10(max(rms_value, 1e-12)) + calibration


def frame_rms(samples: np.ndarray, sample_rate: int, frame_hz: int = 20) -> np.ndarray:
    if type(sample_rate) is not int or sample_rate <= 0 or type(frame_hz) is not int or frame_hz <= 0:
        raise ValidationError("Sample rate and frame rate must be positive integers")
    values = np.asarray(samples, dtype=np.float64)
    rms(values)
    frame_length = max(1, round(sample_rate / frame_hz))
    return np.array([rms(values[start:start + frame_length]) for start in range(0, len(values), frame_length)])


def voiced_rms_db(
    samples: np.ndarray, sample_rate: int, frame_hz: int = 20,
    voiced_rms_threshold: float = 0.01, calibration: float = 0.0,
) -> float | None:
    values = frame_rms(samples, sample_rate, frame_hz)
    mask = values >= voiced_rms_threshold
    if not np.any(mask):
        return None
    frame_length = max(1, round(sample_rate / frame_hz))
    lengths = np.minimum(frame_length, len(samples) - np.arange(len(values)) * frame_length)
    combined = float(np.sqrt(np.average(values[mask] ** 2, weights=lengths[mask])))
    return rms_to_db(combined, calibration)


def voiced_ratio(samples: np.ndarray, sample_rate: int, frame_hz: int = 20, voiced_rms_threshold: float = 0.01) -> float:
    values = frame_rms(samples, sample_rate, frame_hz)
    return float(np.mean(values >= voiced_rms_threshold)) if len(values) else 0.0
