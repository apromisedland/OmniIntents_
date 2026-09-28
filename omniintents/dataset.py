"""Versioned JSONL datasets with labels isolated from inference inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

from .errors import ValidationError
from .intentgpt.taxonomy import SPECIFIC_LABELS
from .prompts import load_resource
from .types import AgentType, MultimodalInput, SCHEMA_VERSION, TaskPlan
from .utils.json_utils import strict_loads
from .utils.validate import parse_task_plan, require_fields


@dataclass
class Sample:
    input: MultimodalInput
    intent_labels: list[str]
    agent_label: AgentType | None
    task_plan: TaskPlan | None
    split: str
    provenance: str


def content_fingerprint(data: dict[str, Any]) -> str:
    values = {key: value for key, value in data.items() if key not in {
        "sample_id", "session_id", "sequence_index", "timestamp",
    }}
    return hashlib.sha256(json.dumps(values, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _validate_rows(rows: list[dict[str, Any]], base: Path | None = None) -> list[Sample]:
    if not rows:
        raise ValidationError("Dataset must not be empty")
    samples = []
    sample_ids = set()
    fingerprints: dict[str, str] = {}
    sessions: dict[str, tuple[int, str, Any]] = {}
    prompt_ids = {
        example["id"] for resource in ("intent_examples.json", "task_examples.json", "agent_examples.json")
        for example in load_resource(f"prompts/{resource}")
    }
    published_utterances = {
        example["input"]["utterance"].strip().casefold()
        for example in load_resource("prompts/intent_examples.json")
        if example["input"]["utterance"]
    }
    for number, row in enumerate(rows, 1):
        try:
            require_fields(row, {"schema_version", "sample_id", "session_id", "sequence_index", "timestamp",
                                 "split", "provenance", "input", "labels"}, {"task_plan"})
            if row["schema_version"] != SCHEMA_VERSION:
                raise ValidationError("Unsupported dataset schema_version")
            if row["split"] not in {"train", "test", "validation", "fewshot", "demo"}:
                raise ValidationError("Unknown split")
            if not isinstance(row["provenance"], str) or not row["provenance"].strip():
                raise ValidationError("provenance must describe the data source")
            if row["sample_id"] in sample_ids or row["sample_id"] in prompt_ids:
                raise ValidationError("Duplicate or reserved prompt sample_id")
            if not isinstance(row["input"], dict):
                raise ValidationError("input must be an object")
            values = dict(row["input"])
            if set(values) & {"sample_id", "session_id", "sequence_index", "timestamp"}:
                raise ValidationError("Identity fields belong at the top level")
            values.update({key: row[key] for key in ("sample_id", "session_id", "sequence_index", "timestamp")})
            data = MultimodalInput.from_json_dict(values)
            if data.session_id in sessions:
                previous_index, previous_split, previous_time = sessions[data.session_id]
                if data.sequence_index <= previous_index or data.timestamp < previous_time:
                    raise ValidationError("Session entries must be chronological and strictly ordered")
                if previous_split != row["split"]:
                    raise ValidationError("A session cannot cross train, few-shot, or evaluation splits")
            sessions[data.session_id] = (data.sequence_index, row["split"], data.timestamp)
            fingerprint = content_fingerprint(data.to_json_dict())
            if fingerprint in fingerprints:
                raise ValidationError("Duplicate input content, including cross-split duplicates")
            fingerprints[fingerprint] = row["split"]
            if row["split"] in {"test", "validation", "demo"} and (
                (data.utterance or data.speech_transcript or "").strip().casefold() in published_utterances
            ):
                raise ValidationError("Evaluation utterance duplicates a bundled few-shot example")
            require_fields(row["labels"], {"intents"}, {"agent"})
            labels = row["labels"]["intents"]
            if not isinstance(labels, list) or not 1 <= len(labels) <= 3:
                raise ValidationError("Ground truth requires one to three non-empty intent labels")
            if any(label not in SPECIFIC_LABELS for label in labels) or len(set(labels)) != len(labels):
                raise ValidationError("Unknown or duplicate ground-truth intent")
            agent = row["labels"].get("agent")
            try:
                agent = AgentType(agent) if agent is not None else None
            except ValueError as error:
                raise ValidationError("Unknown ground-truth agent") from error
            plan = parse_task_plan(row["task_plan"]) if "task_plan" in row else None
            sample_ids.add(data.sample_id)
            if base is not None:
                data.image_paths = [str((base / path).resolve()) if not Path(path).is_absolute() else path for path in data.image_paths]
                if data.audio_path and not Path(data.audio_path).is_absolute():
                    data.audio_path = str((base / data.audio_path).resolve())
            samples.append(Sample(data, labels, agent, plan, row["split"], row["provenance"]))
        except (ValidationError, TypeError, KeyError) as error:
            raise ValidationError(f"Dataset row {number}: {error}") from error
    return samples


def load_dataset(path: str | Path | None = None) -> tuple[list[Sample], str]:
    source = Path(path) if path is not None else files("omniintents").joinpath("data", "smoke.jsonl")
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise ValidationError("Cannot read dataset") from error
    try:
        lines = raw.decode("utf-8-sig").splitlines()
    except UnicodeDecodeError as error:
        raise ValidationError("Dataset must be UTF-8 JSONL") from error
    rows = [strict_loads(line) for line in lines if line.strip()]
    samples = _validate_rows(rows, Path(path).resolve().parent if path is not None else None)
    return samples, hashlib.sha256(raw).hexdigest()


def select_split(samples: list[Sample], split: str) -> list[Sample]:
    selected = [sample for sample in samples if sample.split == split]
    if not selected:
        raise ValidationError(f"No rows in split {split!r}")
    if split in {"train", "fewshot"}:
        raise ValidationError("Evaluation cannot use train or fewshot splits")
    return selected
