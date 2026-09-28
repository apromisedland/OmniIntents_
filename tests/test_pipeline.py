from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest

from omniintents import MultimodalInput, OmniIntentsConfig, OmniIntentsPipeline
from omniintents.dataset import load_dataset
from omniintents.errors import ProviderError, ValidationError
from omniintents.llm import MockLLMClient
from omniintents.types import AgentType
from conftest import CaptureClient


def test_paper_taxonomy():
    from omniintents.intentgpt.taxonomy import GENERAL_LABELS, SPECIFIC_LABELS, SPECIFIC_TO_GENERAL

    assert len(GENERAL_LABELS) == 8
    assert len(SPECIFIC_LABELS) == 21
    assert len(AgentType) == 4
    assert SPECIFIC_TO_GENERAL["Generate Documents"] == "Request Information Management"


@pytest.mark.parametrize("utterance,agent", [
    ("Tell me the weather tomorrow", AgentType.VOICE_ASSISTANT),
    ("Guide me to the art gallery", AgentType.AR_AGENT),
    ("Generate a new illustration", AgentType.GENERATIVE_AI_AGENT),
    ("Bring a mug from the kitchen", AgentType.PHYSICAL_AGENT),
])
def test_end_to_end_categories(utterance, agent):
    result, trace = OmniIntentsPipeline(MockLLMClient()).run_with_trace(MultimodalInput(utterance=utterance))
    assert result.status == "completed"
    assert result.agent.agent_type == agent
    assert result.prediction.schema_version == "0.3"
    assert trace.events == []


def test_ambiguous_prediction_requires_selection(capture_client):
    samples, _ = load_dataset()
    pipe = OmniIntentsPipeline(capture_client)
    result = pipe.run(samples[-1].input)
    assert result.status == "needs_selection"
    assert len(result.prediction.candidates) == 2
    assert result.task_plan is None and result.agent is None
    assert [request.stage for request in capture_client.requests] == ["intent"]
    assert pipe.intentgpt.memory.sessions == {}
    selected = pipe.plan_selected(result.prediction, 1)
    assert selected.intent.specific_label == "Interact with Companion"
    assert selected.agent.agent_type == AgentType.AR_AGENT


def test_top1_is_explicit_batch_policy():
    samples, _ = load_dataset()
    result = OmniIntentsPipeline(MockLLMClient()).run(samples[-1].input, selection="top1")
    assert result.selected_index == 0 and result.status == "completed"


@pytest.mark.parametrize("index", [-1, 3, True, "zero"])
def test_invalid_selection_is_rejected(index):
    pipe = OmniIntentsPipeline(MockLLMClient())
    prediction = pipe.predict(MultimodalInput(utterance="Teach me geometry"))
    with pytest.raises(ValidationError):
        pipe.plan_selected(prediction, index)


def test_no_signal_is_clarification():
    result = OmniIntentsPipeline(MockLLMClient()).run(MultimodalInput())
    assert result.status == "needs_clarification"
    assert result.intent is None


def test_trace_switch():
    result, trace = OmniIntentsPipeline(MockLLMClient(), OmniIntentsConfig(enable_trace=True)).run_with_trace(
        MultimodalInput(utterance="Teach me geometry"),
    )
    assert result.status == "completed"
    assert any(event["name"] == "structured_input" for event in trace.events)


@pytest.mark.parametrize("stage,status", [
    ("intent", "prediction_error"), ("task", "planning_error"), ("agent", "selection_error"),
])
def test_provider_failures_are_explicit(stage, status):
    client = CaptureClient({stage: ProviderError("timeout", "Provider timeout")})
    result = OmniIntentsPipeline(client).run(MultimodalInput(utterance="Teach me geometry"))
    assert result.status == status
    assert result.errors[0]["code"] == "timeout"


def test_history_all_five_rounds_and_isolation(capture_client):
    config = OmniIntentsConfig(history_rounds=5)
    pipe = OmniIntentsPipeline(capture_client, config)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index in range(7):
        result = pipe.run(MultimodalInput(
            utterance=f"Save document {index}", sample_id=f"entry-{index}", session_id="alice",
            sequence_index=index, timestamp=start + timedelta(minutes=index), context_activity=f"activity-{index}",
        ))
        assert result.status == "completed"
        expected = min(index, 5)
        assert len(result.prediction.structured_text.history) == expected
        assert result.prediction.structured_text.history_summary.count("Round ") == expected
    prediction = pipe.predict(MultimodalInput(
        utterance="Continue my work", sample_id="bob", session_id="bob",
        sequence_index=100, timestamp=start + timedelta(hours=1),
    ))
    assert prediction.structured_text.history == []
    assert len(pipe.intentgpt.memory.sessions["alice"]) == 5
    pipe.reset_session("alice")
    assert pipe.intentgpt.memory.sessions == {}


def test_zero_history_has_no_hidden_state():
    pipe = OmniIntentsPipeline(MockLLMClient(), OmniIntentsConfig(history_rounds=0))
    for index in range(2):
        result = pipe.run(MultimodalInput(utterance="Save a note", sequence_index=index, sample_id=str(index)))
        assert not result.prediction.structured_text.history
        assert not result.prediction.structured_text.history_summary
    assert pipe.intentgpt.memory.sessions == {}


def test_future_history_is_not_read():
    from omniintents.intentgpt.memory import ContextualMemoryTracker, HistoryEntry

    memory = ContextualMemoryTracker()
    future = datetime(2026, 1, 2, tzinfo=timezone.utc)
    memory.add_entry("session", HistoryEntry("future", 4, future, "secret", "future activity", "Clean"))
    assert memory.previous("session", 3, future - timedelta(days=1), 2) == []
    with pytest.raises(ValidationError):
        memory.add_entry("session", HistoryEntry("earlier", 3, future, None, None, "Clean"))


def test_label_only_changes_do_not_change_predictions(capture_client):
    from omniintents.evaluation import evaluate

    samples, _ = load_dataset()
    original = samples[:1]
    changed = [replace(original[0], intent_labels=["Clean"], agent_label=AgentType.PHYSICAL_AGENT)]
    first = evaluate(original, capture_client, OmniIntentsConfig(), task="pipeline")
    second = evaluate(changed, capture_client, OmniIntentsConfig(), task="pipeline")
    assert first["predictions"][0]["output"] == second["predictions"][0]["output"]
    assert first["metrics"]["agents"]["accuracy"] != second["metrics"]["agents"]["accuracy"]
