import math
import wave

import numpy as np
import pytest
from PIL import Image

from omniintents import MultimodalInput
from omniintents.intentgpt.audio import AudioProcessor
from omniintents.utils.audio_utils import frame_rms, load_audio, rms, voiced_rms_db
from omniintents.utils.image_utils import (
    blur_score_variance_of_laplacian, choose_keyframes, relative_luminance_from_rgb, smoothed_brightness,
)


def test_brightness_formula_and_missing():
    assert relative_luminance_from_rgb(np.array([[[255, 0, 0]]])) == pytest.approx(0.2126)
    assert relative_luminance_from_rgb(np.full((2, 2, 3), 255)) == pytest.approx(1)
    assert smoothed_brightness([]) is None


def test_five_keyframes_and_smoothing(tmp_path):
    paths = []
    for index in range(7):
        path = tmp_path / f"{index}.png"
        Image.new("RGB", (2, 2), color=(index * 40,) * 3).save(path)
        paths.append(str(path))
    selected = choose_keyframes(paths)
    assert len(selected) == 5 and selected[0] == paths[0] and selected[-1] == paths[-1]
    expected = np.mean([0, 40, 120, 160, 240]) / 255
    assert smoothed_brightness(paths) == pytest.approx(expected)


def test_vectorized_laplacian_matches_independent_kernel():
    image = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
    gray = image @ np.array([0.2126, 0.7152, 0.0722])
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    reference = [np.sum(gray[row:row + 3, column:column + 3] * kernel)
                 for row in range(3) for column in range(3)]
    assert blur_score_variance_of_laplacian(image) == pytest.approx(np.var(reference), abs=1e-20)


def test_audio_20hz_and_weighted_partial_frame():
    samples = np.r_[np.ones(50), np.full(25, 0.5)]
    assert frame_rms(samples, 1000).tolist() == [1.0, 0.5]
    assert voiced_rms_db(samples, 1000) == pytest.approx(20 * math.log10(math.sqrt(56.25 / 75)))
    assert voiced_rms_db(np.zeros(100), 1000) is None
    assert rms(np.array([])) == 0


def test_pcm_wav_and_supplied_features(tmp_path):
    path = tmp_path / "audio.wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(16000)
        stream.writeframes(np.full(800, 16384, dtype="<i2").tobytes())
    audio = load_audio(str(path))
    assert audio.sample_rate == 16000
    assert rms(audio.samples) == 0.5
    result = AudioProcessor().process(MultimodalInput(
        audio_path=str(path), speech_transcript="hello", speech_confidence=0.9,
        sound_events=[{"label": "Speech", "confidence": 0.7}],
    ))
    assert result["sound_classification"]["likely_event"] is None
    assert result["volume"]["value"] == pytest.approx(20 * math.log10(0.5))
