import json
from datetime import datetime, timezone

import pytest
from PIL import Image

from omniintents import MultimodalInput, OmniIntentsConfig, OmniIntentsPipeline
from omniintents.intentgpt.memory import HistoryEntry


@pytest.mark.parametrize("modality,field", [
    ("context", "context_location"), ("objects", "objects"), ("hand_eye", "hand_state"),
    ("speech", "speech_transcript"), ("targets", "hand_target"), ("audio_labels", "sound_events"),
])
def test_ablation_removes_all_duplicate_channels(capture_client, modality, field):
    needle = "UNIQUE_REMOVED_SIGNAL"
    values = {
        "utterance": f"Search for information about {needle}",
        "speech_transcript": f"Search for information about {needle}", "speech_confidence": 0.95,
        "brightness": 0.8, "scene_description": f"Raw caption repeats {needle}",
        "image_paths": ["must-not-be-opened.png"], "audio_path": "must-not-be-opened.wav",
        "sample_id": "current", "session_id": "session", "sequence_index": 1,
        "timestamp": datetime(2026, 1, 2, tzinfo=timezone.utc),
    }
    values[field] = [needle] if field == "objects" else [
        {"label": needle, "confidence": 0.9},
    ] if field == "sound_events" else needle
    pipe = OmniIntentsPipeline(capture_client, OmniIntentsConfig(removed_modalities=(modality,)))
    pipe.intentgpt.memory.add_entry("session", HistoryEntry(
        "previous", 0, datetime(2026, 1, 1, tzinfo=timezone.utc), needle, needle, "Clean",
    ))
    pipe.run(MultimodalInput(**values), selection="top1")
    for request in capture_client.requests:
        assert request.image_paths == []
        assert request.stage != "vision"
        assert needle not in json.dumps(request.payload)
        if "input" in request.payload:
            assert request.payload["input"]["history"] == []
            assert request.payload["input"]["history_summary"] == ""


def test_ifas_has_original_images_and_selected_intent(tmp_path, capture_client):
    image = tmp_path / "book.png"
    Image.new("RGB", (4, 4), color="white").save(image)
    pipe = OmniIntentsPipeline(capture_client)
    result = pipe.run(MultimodalInput(
        utterance="Search for information about this book", image_paths=[str(image)], hand_target="the blue book",
    ))
    assert result.status == "completed"
    task_request = next(request for request in capture_client.requests if request.stage == "task")
    assert task_request.image_paths == [str(image)]
    assert task_request.payload["ifas_mode"] == "raw_images"
    assert "selected_intent" in task_request.payload
    assert "Reinspect" in task_request.payload["focus_instruction"]


def test_ifas_disabled_keeps_comparison_inputs(capture_client):
    pipe = OmniIntentsPipeline(capture_client, OmniIntentsConfig(ifas_enabled=False))
    pipe.run(MultimodalInput(utterance="Search for information about a blue book", hand_target="blue book"))
    request = next(request for request in capture_client.requests if request.stage == "task")
    assert request.payload["focus_instruction"] == ""
    assert request.payload["ifas_mode"] == "disabled"
    assert request.payload["input"]["visual"]["hand_target"] == "blue book"


def test_preextracted_ifas_is_labeled(capture_client):
    OmniIntentsPipeline(capture_client).run(MultimodalInput(utterance="Search for a recipe"))
    request = next(request for request in capture_client.requests if request.stage == "task")
    assert request.payload["ifas_mode"] == "preextracted"
    assert request.image_paths == []


def test_modality_reference_has_matching_media_and_history_controls(capture_client):
    from omniintents.ablations import ablation_configs

    config = ablation_configs(OmniIntentsConfig())["modality_reference"]
    data = MultimodalInput(
        utterance="Search for a book", context_location="library",
        image_paths=["not-accessed.png"], scene_description="excluded description",
    )
    prediction = OmniIntentsPipeline(capture_client, config).predict(data)
    assert prediction.structured_text.visual["context"]["location"] == "library"
    assert "scene_description" not in prediction.structured_text.visual
    assert prediction.structured_text.history == []
    assert all(request.image_paths == [] for request in capture_client.requests)
