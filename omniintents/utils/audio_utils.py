from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore


@dataclass
class AudioData:
    samples: np.ndarray  # mono float32 in [-1,1] typical
    sample_rate: int


def load_audio(path: str) -> AudioData:
    """Load audio from file.

    Prefers soundfile. If unavailable but librosa is available, librosa can load common formats.
    """
    if sf is not None:
        samples, sr = sf.read(path, always_2d=False)
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        return AudioData(samples=samples, sample_rate=int(sr))

    if librosa is not None:  # pragma: no cover
        samples, sr = librosa.load(path, sr=None, mono=True)
        samples = np.asarray(samples, dtype=np.float32)
        return AudioData(samples=samples, sample_rate=int(sr))

    raise RuntimeError("No audio loader available. Install 'soundfile' or 'librosa'.")


def rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(x))))


def rms_to_db(rms_value: float, calibration: float = 0.0) -> float:
    eps = 1e-12
    return float(20.0 * math.log10(max(rms_value, eps)) + calibration)


def frame_rms(samples: np.ndarray, sample_rate: int, frame_hz: int = 20) -> np.ndarray:
    """Compute RMS per frame at `frame_hz`.

    This mirrors the paper's Web Audio API approach (20 fps).
    """
    samples = np.asarray(samples, dtype=np.float32)
    frame_len = max(1, int(sample_rate / max(1, frame_hz)))
    n_frames = int(math.ceil(samples.size / frame_len))
    out = np.zeros((n_frames,), dtype=np.float32)
    for i in range(n_frames):
        start = i * frame_len
        end = min(samples.size, (i + 1) * frame_len)
        out[i] = rms(samples[start:end])
    return out


def voiced_rms_db(
    samples: np.ndarray,
    sample_rate: int,
    frame_hz: int = 20,
    voiced_rms_threshold: float = 0.01,
    calibration: float = 0.0,
) -> Optional[float]:
    """Approximate 'voiced segments RMS -> dB' pipeline.

    - Per-frame RMS at `frame_hz`
    - Filter out frames below threshold
    - Mean RMS across voiced frames
    - Convert to dB
    """
    per_frame = frame_rms(samples, sample_rate, frame_hz=frame_hz)
    voiced = per_frame[per_frame >= voiced_rms_threshold]
    if voiced.size == 0:
        return None
    voiced_rms_val = float(np.mean(voiced))
    return rms_to_db(voiced_rms_val, calibration=calibration)


def voiced_ratio(
    samples: np.ndarray,
    sample_rate: int,
    frame_hz: int = 20,
    voiced_rms_threshold: float = 0.01,
) -> float:
    """Fraction of frames considered 'voiced' by RMS threshold."""
    per_frame = frame_rms(samples, sample_rate, frame_hz=frame_hz)
    if per_frame.size == 0:
        return 0.0
    return float(np.mean(per_frame >= voiced_rms_threshold))
