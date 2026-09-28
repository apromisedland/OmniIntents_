import copy
import json

import pytest

from omniintents import OmniIntentsConfig
from omniintents.dataset import _validate_rows, load_dataset, select_split
from omniintents.errors import ProviderError, ValidationError
from omniintents.evaluation import evaluate, reproduce
from omniintents.llm import MockLLMClient
from omniintents.prompts import load_resource, prompt_hashes
from conftest import CaptureClient


def row():
    return {
        "schema_version": "0.3", "sample_id": "test-one", "session_id": "test-session",
        "sequence_index": 0, "timestamp": "2026-01-01T00:00:00Z", "split": "test",
        "provenance": "synthetic test", "input": {"utterance": "Clean the classroom"},
        "labels": {"intents": ["Clean"], "agent": "physical_agent"},
    }


@pytest.mark.parametrize("mutation", [
    lambda value: value["labels"].update(intents=[]),
    lambda value: value["labels"].update(intents=["Unknown"]),
    lambda value: value["input"].update(labels=["Clean"]),
    lambda value: value.update(timestamp="yesterday"),
    lambda value: value.update(schema_version="0.2"),
    lambda value: value.update(sample_id="supp-intent-01"),
    lambda value: value["input"].update(utterance="turn on the light"),
])
def test_invalid_dataset_rows(mutation):
    value = row()
    mutation(value)
    with pytest.raises(ValidationError):
        _validate_rows([value])


def test_duplicate_and_cross_split_leakage():
    first = row()
    second = copy.deepcopy(first)
    second.update(sample_id="second", sequence_index=1, split="validation")
    second["input"]["utterance"] = "Wash another classroom"
    with pytest.raises(ValidationError, match="cross"):
        _validate_rows([first, second])
    second.update(session_id="other", split="test")
    second["input"] = first["input"]
    with pytest.raises(ValidationError, match="Duplicate input"):
        _validate_rows([first, second])


def test_reversed_sequence_and_empty_dataset():
    with pytest.raises(ValidationError):
        _validate_rows([])
    first, second = row(), row()
    second["sample_id"] = "other"
    second["input"]["utterance"] = "Clean another room"
    with pytest.raises(ValidationError, match="ordered"):
        _validate_rows([first, second])


def test_bundled_examples_are_complete_and_disjoint():
    samples, digest = load_dataset()
    assert len(samples) == 28 and len(digest) == 64
    assert [len(load_resource(f"prompts/{name}_examples.json")) for name in ("intent", "task", "agent")] == [7, 3, 4]
    assert len(prompt_hashes()) == 4
    assert {label for sample in samples for label in sample.intent_labels} == set(load_resource("prompts/definitions.json"))


def test_failures_do_not_disappear():
    samples, digest = load_dataset()
    client = CaptureClient({"intent": ProviderError("timeout", "Timeout")})
    report = evaluate(samples[:2], client, OmniIntentsConfig(), dataset_sha256=digest)
    assert report["sample_count"] == 2
    assert report["metrics"]["specific_intents"]["samples"] == 2
    assert report["metrics"]["specific_intents"]["overlap_acc"] == 0
    assert report["errors"] == {"timeout": 2}
    assert len(report["predictions"]) == 2


def test_invalid_model_output_is_counted():
    samples, _ = load_dataset()
    report = evaluate(samples[:1], CaptureClient({"intent": "invalid"}), OmniIntentsConfig())
    assert report["errors"] == {"validation_error": 1}
    assert report["metrics"]["specific_intents"]["overlap_acc"] == 0


def test_evaluation_disallows_demo_fallback_and_train():
    samples, _ = load_dataset()
    with pytest.raises(ValidationError):
        evaluate(samples, MockLLMClient(), OmniIntentsConfig(failure_policy="demo"))
    with pytest.raises(ValidationError):
        select_split(samples, "train")


def strip_timing(value):
    if isinstance(value, dict):
        return {key: strip_timing(item) for key, item in value.items() if key not in {"timing", "duration_s"}}
    if isinstance(value, list):
        return [strip_timing(item) for item in value]
    return value


def test_repeated_offline_results_are_deterministic():
    samples, digest = load_dataset()
    first = evaluate(samples, MockLLMClient(), OmniIntentsConfig(), task="pipeline", dataset_sha256=digest)
    second = evaluate(samples, MockLLMClient(), OmniIntentsConfig(), task="pipeline", dataset_sha256=digest)
    assert strip_timing(first) == strip_timing(second)
    assert first["claim"] == "synthetic_sanity_only"


def test_manifest_uses_one_resolved_source_root(monkeypatch):
    from pathlib import Path
    from omniintents import evaluation

    original = evaluation.environment_manifest()
    repository = Path(evaluation.__file__).resolve().parent.parent
    monkeypatch.chdir(repository)
    monkeypatch.setattr(evaluation, "__file__", "omniintents/evaluation.py")
    relative = evaluation.environment_manifest()
    assert relative["source_sha256"] == original["source_sha256"]


def test_full_ablation_artifacts(tmp_path):
    samples, digest = load_dataset()
    root = tmp_path / "experiment"
    summary = reproduce(samples, MockLLMClient(), OmniIntentsConfig(), root, dataset_sha256=digest, suite="ablations")
    assert len(summary["runs"]) == 54
    manifest = json.loads((root / "baseline" / "intent" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment"]["dataset_sha256"] == digest
    assert manifest["environment"]["source_sha256"]
    baseline = json.loads((root / "baseline" / "pipeline" / "metrics.json").read_text(encoding="utf-8"))
    assert baseline["errors"] == {}
    assert (root / "without_speech" / "intent" / "predictions.jsonl").is_file()
    with pytest.raises(ValidationError, match="empty"):
        reproduce(samples, MockLLMClient(), OmniIntentsConfig(), root)
