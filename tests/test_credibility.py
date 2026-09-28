import pytest

from omniintents import MultimodalInput, OmniIntentsConfig, OmniIntentsPipeline
from omniintents.errors import ValidationError
from omniintents.intentgpt.credibility import CredibilityBasedAttentionShifter
from omniintents.llm import MockLLMClient


@pytest.mark.parametrize("brightness,confidence,vision,voice", [
    (0.2, 0.8, False, False), (0.20001, 0.80001, True, True),
    (0.1, 0.9, False, True), (0.8, 0.1, True, False),
])
def test_published_thresholds_and_equality(brightness, confidence, vision, voice):
    report = CredibilityBasedAttentionShifter().evaluate(
        {"brightness": brightness}, {"speech": {"transcript": "a request", "confidence": confidence}},
    )
    assert report["visual"]["credible"] is vision
    assert report["voice"]["credible"] is voice


def test_missing_not_dark():
    report = CredibilityBasedAttentionShifter().evaluate({}, {})
    assert report["visual"]["brightness"] is None
    assert report["visual"]["status"] == "missing"


def test_typed_text_is_not_fake_asr_confidence():
    prediction = OmniIntentsPipeline(MockLLMClient()).predict(MultimodalInput(utterance="Teach me geometry"))
    assert prediction.structured_text.audio["speech"]["confidence"] is None
    assert prediction.structured_text.credibility_report["voice"]["status"] == "typed_text"


def test_cbas_disabled_in_current_input_and_examples(capture_client):
    pipe = OmniIntentsPipeline(capture_client, OmniIntentsConfig(cbas_enabled=False))
    prediction = pipe.predict(MultimodalInput(utterance="Teach me geometry"))
    assert prediction.structured_text.credibility_report == {}
    assert all("credibility_report" not in example["input"] for example in capture_client.requests[0].payload["examples"])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 1.1, True, "0.9"])
def test_confidence_validation(value):
    with pytest.raises(ValidationError):
        MultimodalInput(speech_confidence=value)


@pytest.mark.parametrize("kwargs", [
    {"history_rounds": -1}, {"history_rounds": 6}, {"max_history_entries": 6},
    {"history_rounds": True}, {"cbas_enabled": "false"}, {"audio_frame_hz": 0},
    {"temperature": float("nan")}, {"removed_modalities": ("unknown",)},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValidationError):
        OmniIntentsConfig(**kwargs)
