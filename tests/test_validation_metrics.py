import math

import pytest

from omniintents.errors import ValidationError
from omniintents.metrics import agent_metrics, multilabel_metrics
from omniintents.utils.json_utils import extract_first_json_object, require_json_object, strict_loads
from omniintents.utils.validate import parse_intents, parse_task_plan


def test_json_recovery_nested_braces_and_multiple_objects():
    assert extract_first_json_object('prefix {"text":"a } brace","nested":{"ok":true}} then {"second":2}') == {
        "text": "a } brace", "nested": {"ok": True},
    }
    assert extract_first_json_object(' ```json\n{"items":[{"value":1}]}\n``` ') == {"items": [{"value": 1}]}


@pytest.mark.parametrize("text", ['{"value":NaN}', '{"value":Infinity}', '{"value":1,"value":2}', "no JSON"])
def test_invalid_json_is_rejected(text):
    with pytest.raises(ValidationError):
        require_json_object(text)


def test_invalid_json_document():
    with pytest.raises(ValidationError):
        strict_loads('{"a":1,}')


@pytest.mark.parametrize("candidate", [
    {"specific_label": "Invalid", "description": "x", "confidence": 0.5},
    {"specific_label": "Clean", "description": "x", "confidence": math.nan},
    {"specific_label": "Clean", "description": "", "confidence": 0.5},
    {"specific_label": "Clean", "general_label": "Request Guide", "description": "x", "confidence": 0.5},
])
def test_invalid_intent(candidate):
    with pytest.raises(ValidationError):
        parse_intents({"candidates": [candidate], "complementarity": "unknown", "ambiguity": "unknown"})


@pytest.mark.parametrize("field", ["complementarity", "ambiguity"])
@pytest.mark.parametrize("value", [[], {}, None, False])
def test_wrong_type_enum_is_a_validation_error(field, value):
    output = {"candidates": [], "complementarity": "unknown", "ambiguity": "unknown"}
    output[field] = value
    with pytest.raises(ValidationError):
        parse_intents(output)


def test_candidate_rank_and_maximum():
    candidates = [
        {"specific_label": "Clean", "description": str(index), "confidence": index / 10}
        for index in range(3)
    ]
    parsed, _, _ = parse_intents({"candidates": candidates, "complementarity": "unknown", "ambiguity": "ambiguous"})
    assert [candidate.confidence for candidate in parsed] == [0.2, 0.1, 0.0]
    with pytest.raises(ValidationError):
        parse_intents({"candidates": candidates + candidates[:1], "complementarity": "unknown", "ambiguity": "ambiguous"})


@pytest.mark.parametrize("step", [
    {"step_id": "1", "instruction": "x", "required_capabilities": ["speech_io"]},
    {"step_id": 1, "instruction": "x", "required_capabilities": ["flying"]},
    {"step_id": 1, "instruction": "x", "required_capabilities": []},
])
def test_invalid_task_plan(step):
    with pytest.raises(ValidationError):
        parse_task_plan({"goal": "goal", "steps": [step]})


def test_overlap_is_not_exact_match():
    report = multilabel_metrics([{"a", "b"}, {"a"}], [{"a"}, set()], ["a", "b", "c"])
    assert report["overlap_acc"] == 0.5
    assert report["strict_set_accuracy"] == 0
    assert report["macro_f1"] == pytest.approx((2 / 3) / 3)
    assert report["per_class"]["c"]["f1"] == 0


def test_failed_agent_predictions_stay_in_denominator():
    report = agent_metrics(["voice", "robot", "robot"], ["voice", None, "voice"], ["voice", "robot"])
    assert report["accuracy"] == pytest.approx(1 / 3)
    assert report["macro_f1"] == pytest.approx(1 / 3)
    assert report["weighted_f1"] == pytest.approx(2 / 9)
    assert report["invalid_predictions"] == 1
    assert sum(sum(row) for row in report["confusion_matrix"]["counts"]) == 3


@pytest.mark.parametrize("truth,prediction", [([], []), ([set()], [set()]), ([{"a"}], []), ([{"z"}], [set()])])
def test_invalid_metric_inputs(truth, prediction):
    with pytest.raises(ValidationError):
        multilabel_metrics(truth, prediction, ["a"])
