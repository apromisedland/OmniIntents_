"""Finite-sample metrics on fixed label universes, with failures in the denominator."""

from __future__ import annotations

from collections import Counter
from typing import Sequence

from .errors import ValidationError


def _check_lengths(truth: Sequence, prediction: Sequence) -> None:
    if not truth or len(truth) != len(prediction):
        raise ValidationError("Metric inputs must be non-empty and have equal lengths")


def multilabel_metrics(
    truth: Sequence[set[str]], prediction: Sequence[set[str]], labels: Sequence[str],
) -> dict:
    _check_lengths(truth, prediction)
    universe = set(labels)
    if any(not target or not target <= universe for target in truth):
        raise ValidationError("Ground-truth sets must be non-empty and use known labels")
    if any(not predicted <= universe for predicted in prediction):
        raise ValidationError("Predicted sets contain unknown labels")
    overlaps = [
        len(target & predicted) / min(len(target), len(predicted)) if predicted else 0.0
        for target, predicted in zip(truth, prediction)
    ]
    per_class = {}
    for label in labels:
        true_positive = sum(label in target and label in predicted for target, predicted in zip(truth, prediction))
        false_positive = sum(label not in target and label in predicted for target, predicted in zip(truth, prediction))
        false_negative = sum(label in target and label not in predicted for target, predicted in zip(truth, prediction))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class[label] = {
            "f1": 2 * true_positive / denominator if denominator else 0.0,
            "support": sum(label in target for target in truth),
        }
    return {
        "samples": len(truth), "overlap_acc": sum(overlaps) / len(truth),
        "strict_set_accuracy": sum(target == predicted for target, predicted in zip(truth, prediction)) / len(truth),
        "macro_f1": sum(value["f1"] for value in per_class.values()) / len(labels),
        "per_class": per_class, "scale": "0..1", "zero_division": 0,
    }


def agent_metrics(truth: Sequence[str], prediction: Sequence[str | None], labels: Sequence[str]) -> dict:
    _check_lengths(truth, prediction)
    if any(target not in labels for target in truth) or any(
        predicted is not None and predicted not in labels for predicted in prediction
    ):
        raise ValidationError("Unknown agent metric label")
    columns = list(labels) + ["invalid"]
    matrix = [[0 for _ in columns] for _ in labels]
    per_class = {}
    counts = Counter(truth)
    for target, predicted in zip(truth, prediction):
        matrix[labels.index(target)][columns.index(predicted if predicted is not None else "invalid")] += 1
    for label in labels:
        true_positive = sum(target == label and predicted == label for target, predicted in zip(truth, prediction))
        false_positive = sum(target != label and predicted == label for target, predicted in zip(truth, prediction))
        false_negative = sum(target == label and predicted != label for target, predicted in zip(truth, prediction))
        denominator = 2 * true_positive + false_positive + false_negative
        per_class[label] = 2 * true_positive / denominator if denominator else 0.0
    normalized = [[value / counts[label] if counts[label] else 0.0 for value in row] for label, row in zip(labels, matrix)]
    return {
        "samples": len(truth), "accuracy": sum(target == predicted for target, predicted in zip(truth, prediction)) / len(truth),
        "macro_f1": sum(per_class.values()) / len(labels),
        "weighted_f1": sum(per_class[label] * counts[label] for label in labels) / len(truth),
        "confusion_matrix": {"rows": list(labels), "columns": columns, "counts": matrix, "row_normalized": normalized},
        "invalid_predictions": sum(predicted is None for predicted in prediction),
        "scale": "0..1", "zero_division": 0,
    }
