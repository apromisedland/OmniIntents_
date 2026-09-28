import json
from types import SimpleNamespace

import pytest
import requests
from PIL import Image

from omniintents.errors import ProviderError, ValidationError
from omniintents.llm.base import LLMRequest, RecordingClient
from omniintents.llm.openai_compatible import OpenAICompatibleClient


class FakeResponse:
    def __init__(self, status=200, data=None):
        self.status_code = status
        self.data = data if data is not None else {
            "model": "fixture-model-snapshot", "choices": [
                {"message": {"content": '{"ok":true}'}, "finish_reason": "stop"},
            ], "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
        }

    def json(self):
        return self.data


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def client_for(session, **kwargs):
    return OpenAICompatibleClient(
        model="test-model", api_key="DO_NOT_LOG_THIS_KEY", base_url="https://example.invalid/v1",
        session=session, **kwargs,
    )


def test_protocol_url_text_usage_and_model():
    session = FakeSession([FakeResponse()])
    recorder = RecordingClient(client_for(session, token_parameter="max_completion_tokens", send_seed=True))
    recorder.generate(LLMRequest("intent", "Return JSON", {"input": "hello"}, seed=7))
    url, request = session.calls[0]
    assert url == "https://example.invalid/v1/chat/completions"
    assert "/v1/v1/" not in url
    assert request["json"]["max_completion_tokens"] == 1600
    assert request["json"]["seed"] == 7
    assert request["json"]["messages"][0]["role"] == "system"
    assert recorder.calls[0]["model"] == "fixture-model-snapshot"
    assert recorder.calls[0]["usage"]["total_tokens"] == 14
    assert "DO_NOT_LOG_THIS_KEY" not in json.dumps(recorder.calls)


def test_jpeg_mime_and_vision_model(tmp_path):
    filename = tmp_path / "image.jpg"
    Image.new("RGB", (4, 4), color="white").save(filename)
    session = FakeSession([FakeResponse()])
    client_for(session, vision_model="test-vision").generate(LLMRequest(
        "vision", "Return JSON", {}, image_paths=[str(filename)],
    ))
    payload = session.calls[0][1]["json"]
    assert payload["model"] == "test-vision"
    assert payload["messages"][1]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


@pytest.mark.parametrize("failure", [FakeResponse(429), FakeResponse(503), requests.Timeout(), requests.ConnectionError()])
def test_retryable_failure_is_bounded(monkeypatch, failure):
    monkeypatch.setattr("omniintents.llm.openai_compatible.time.sleep", lambda duration: None)
    session = FakeSession([failure, failure, failure])
    with pytest.raises(ProviderError) as caught:
        client_for(session).generate(LLMRequest("intent", "Return JSON", {}))
    assert len(session.calls) == 3
    assert "DO_NOT_LOG_THIS_KEY" not in str(caught.value)


def test_retry_can_recover(monkeypatch):
    monkeypatch.setattr("omniintents.llm.openai_compatible.time.sleep", lambda duration: None)
    session = FakeSession([FakeResponse(429), FakeResponse()])
    result = client_for(session).generate(LLMRequest("intent", "Return JSON", {}))
    assert result.metadata["attempts"] == 2


@pytest.mark.parametrize("status,code", [(400, "request_rejected"), (401, "authentication"), (403, "authentication")])
def test_nonretryable_status(status, code):
    session = FakeSession([FakeResponse(status)])
    with pytest.raises(ProviderError) as caught:
        client_for(session).generate(LLMRequest("intent", "Return JSON", {}))
    assert caught.value.code == code and len(session.calls) == 1


@pytest.mark.parametrize("data", [{}, {"choices": []}, {"choices": [{"message": {"content": None}}]}])
def test_malformed_provider_response(data):
    with pytest.raises(ProviderError):
        client_for(FakeSession([FakeResponse(data=data)])).generate(LLMRequest("intent", "Return JSON", {}))


def test_live_model_is_not_silently_chosen(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    with pytest.raises(ValidationError):
        OpenAICompatibleClient(api_key="dummy")


def test_yamnet_adapter_521_classes(tmp_path):
    import numpy as np
    from omniintents.intentgpt.audio import YamNetClassifier

    labels = tmp_path / "labels.csv"
    labels.write_text("index,mid,display_name\n" + "".join(
        f"{index},id{index},class{index}\n" for index in range(521)
    ), encoding="utf-8")
    scores = np.zeros((2, 521))
    scores[:, 4], scores[:, 3], scores[:, 2] = 0.9, 0.8, 0.7
    classifier = YamNetClassifier("unused", str(labels), model=lambda samples: (scores, None, None))
    assert classifier.classify_topk(np.ones(1600), 16000) == [
        ("class4", 0.9), ("class3", 0.8), ("class2", 0.7),
    ]
    with pytest.raises(ValidationError):
        classifier.classify_topk(np.ones(1600), 44100)


def test_google_speech_adapter_preserves_confidence(tmp_path, monkeypatch):
    import sys
    import types
    from omniintents.intentgpt.audio import GoogleSpeechTranscriber

    google = types.ModuleType("google")
    cloud = types.ModuleType("google.cloud")
    speech = SimpleNamespace(RecognitionConfig=lambda **kwargs: kwargs, RecognitionAudio=lambda **kwargs: kwargs)
    cloud.speech = speech
    google.cloud = cloud
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.cloud", cloud)
    calls = []
    def recognize(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(results=[
            SimpleNamespace(alternatives=[SimpleNamespace(transcript="one", confidence=0.6)]),
            SimpleNamespace(alternatives=[SimpleNamespace(transcript="two words", confidence=0.9)]),
        ])
    path = tmp_path / "speech.wav"
    path.write_bytes(b"fixture bytes")
    transcriber = GoogleSpeechTranscriber(client=SimpleNamespace(recognize=recognize))
    transcript, confidence = transcriber.transcribe(str(path))
    assert transcript == "one two words"
    assert confidence == pytest.approx(0.8)
    assert calls[0]["audio"]["content"] == b"fixture bytes"
