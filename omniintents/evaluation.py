"""Auditable stage-specific evaluations and deterministic offline reproduction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

from . import __version__
from .ablations import ablation_configs
from .agentgpt.agentgpt import AgentGPT
from .config import OmniIntentsConfig
from .dataset import Sample
from .errors import OmniIntentsError, ValidationError
from .intentgpt.taxonomy import GENERAL_LABELS, SPECIFIC_LABELS, SPECIFIC_TO_GENERAL
from .llm.base import LLMClient, RecordingClient
from .metrics import agent_metrics, multilabel_metrics
from .pipeline import OmniIntentsPipeline
from .prompts import prompt_hashes
from .types import AgentType
from .utils.json_utils import dumps_pretty


def evaluate(
    samples: list[Sample], client: LLMClient, config: OmniIntentsConfig, *,
    task: str = "intent", dataset_sha256: str = "", audio_processor: Any = None,
) -> dict[str, Any]:
    if task not in {"intent", "agent", "pipeline"}:
        raise ValidationError("task must be intent, agent, or pipeline")
    if not samples:
        raise ValidationError("No evaluation samples")
    if config.failure_policy != "error":
        raise ValidationError("Scientific evaluation disallows demo fallbacks")
    if task in {"agent", "pipeline"} and any(sample.agent_label is None for sample in samples):
        raise ValidationError("Agent and pipeline evaluation require an agent label for every sample")
    if task == "agent" and any(sample.task_plan is None for sample in samples):
        raise ValidationError("Given-task agent evaluation requires task_plan for every sample")
    if (config.removed_modalities or config.structured_ablation_mode) and any(
        (sample.input.image_paths or sample.input.audio_path)
        and not any(value is not None for value in (
            sample.input.objects, sample.input.context_activity, sample.input.context_location,
            sample.input.hand_state, sample.input.eye_state, sample.input.speech_transcript,
        )) for sample in samples
    ):
        raise ValidationError("Modality ablations require pre-extracted features; raw media are masked")
    recorder = RecordingClient(client)
    pipeline = OmniIntentsPipeline(recorder, config, audio_processor=audio_processor)
    records = []
    errors = Counter()
    for sample in samples:
        start = time.perf_counter()
        call_start = len(recorder.calls)
        predicted: set[str] = set()
        selected_agent = None
        error_code = None
        outcome = None
        try:
            if task == "intent":
                prediction = pipeline.predict(sample.input)
                predicted = {intent.specific_label for intent in prediction.candidates}
                outcome = prediction.to_json_dict()
                if prediction.candidates:
                    pipeline.intentgpt.remember(prediction, prediction.candidates[0])
                else:
                    error_code = "empty_prediction"
            elif task == "agent":
                recommendation = AgentGPT(recorder, config).recommend(sample.task_plan)
                selected_agent = recommendation.agent_type.value if recommendation.agent_type else None
                outcome = recommendation.to_json_dict()
                if recommendation.status != "selected":
                    error_code = recommendation.status
            else:
                result = pipeline.run(sample.input, selection="top1")
                outcome = result.to_json_dict()
                predicted = {intent.specific_label for intent in result.prediction.candidates} if result.prediction else set()
                selected_agent = result.agent.agent_type.value if result.agent and result.agent.agent_type else None
                if result.status != "completed":
                    error_code = result.errors[0]["code"] if result.errors else result.status
        except OmniIntentsError as error:
            error_code = error.code
        if error_code:
            errors[error_code] += 1
        records.append({
            "sample_id": sample.input.sample_id, "session_id": sample.input.session_id,
            "predicted_intents": sorted(predicted), "predicted_agent": selected_agent,
            "error_code": error_code, "duration_s": time.perf_counter() - start,
            "calls": recorder.calls[call_start:], "output": outcome,
        })
    scores: dict[str, Any] = {}
    if task in {"intent", "pipeline"}:
        truth = [set(sample.intent_labels) for sample in samples]
        prediction = [set(record["predicted_intents"]) for record in records]
        scores["specific_intents"] = multilabel_metrics(truth, prediction, SPECIFIC_LABELS)
        scores["general_intents"] = multilabel_metrics(
            [{SPECIFIC_TO_GENERAL[label] for label in labels} for labels in truth],
            [{SPECIFIC_TO_GENERAL[label] for label in labels} for labels in prediction], GENERAL_LABELS,
        )
    if task in {"agent", "pipeline"}:
        scores["agents"] = agent_metrics(
            [sample.agent_label.value for sample in samples],
            [record["predicted_agent"] for record in records], [agent.value for agent in AgentType],
        )
    if task == "pipeline":
        scores["pipeline_completion_rate"] = sum(record["error_code"] is None for record in records) / len(records)
    latencies: dict[str, list[float]] = {}
    for call in recorder.calls:
        latencies.setdefault(call["stage"], []).append(call["duration_s"])
    return {
        "schema_version": "0.3", "task": task, "sample_count": len(samples),
        "claim": "synthetic_sanity_only" if client.backend == "mock" else "new_run_not_original_paper_reproduction",
        "backend": client.backend, "config": config.to_json_dict(),
        "dataset_sha256": dataset_sha256, "prompt_sha256": prompt_hashes(),
        "history_method": "deterministic_reference", "history_labels": "previous_predictions_never_ground_truth",
        "sample_ids": [sample.input.sample_id for sample in samples],
        "metrics": scores, "errors": dict(errors), "predictions": records,
        "timing": {
            "sample_mean_s": statistics.mean(record["duration_s"] for record in records),
            "model_request_mean_s": {stage: statistics.mean(values) for stage, values in latencies.items()},
            "definition": "Wall-clock request to complete response, including adapter retries; sample time also includes preprocessing.",
        },
    }


def environment_manifest() -> dict[str, Any]:
    versions = {}
    for name in ("omniintents", "numpy", "Pillow", "requests"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "unavailable"
    revision = None
    try:
        package_root = Path(__file__).resolve().parent.parent
        if (package_root / ".git").exists():
            revision = subprocess.run(
                ["git", "-C", str(package_root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True,
            ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        pass
    source_hash = hashlib.sha256()
    for path in sorted(Path(__file__).resolve().parent.rglob("*.py")):
        source_hash.update(str(path.relative_to(Path(__file__).parent)).replace("\\", "/").encode())
        source_hash.update(path.read_bytes())
    return {
        "package_version": __version__, "python": platform.python_version(),
        "platform": platform.system(), "dependencies": versions, "git_revision": revision,
        "source_sha256": source_hash.hexdigest(),
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> Path:
    directory = Path(output_dir)
    if directory.exists() and any(directory.iterdir()):
        raise ValidationError("Output directory must be absent or empty")
    directory.mkdir(parents=True, exist_ok=True)
    payload = dict(report)
    predictions = payload.pop("predictions", [])
    manifest = {"environment": environment_manifest(), "experiment": {
        key: payload[key] for key in ("schema_version", "task", "backend", "config", "dataset_sha256", "prompt_sha256", "sample_ids")
    }}
    (directory / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in predictions),
        encoding="utf-8",
    )
    (directory / "metrics.json").write_text(dumps_pretty(payload) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(dumps_pretty(manifest) + "\n", encoding="utf-8")
    return directory


def reproduce(
    samples: list[Sample], client: LLMClient, config: OmniIntentsConfig, output_dir: str | Path, *,
    dataset_sha256: str = "", suite: str = "smoke", audio_processor: Any = None,
) -> dict[str, Any]:
    if suite not in {"smoke", "ablations"}:
        raise ValidationError("suite must be smoke or ablations")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise ValidationError("Reproduction output directory must be absent or empty")
    variants = ablation_configs(config) if suite == "ablations" else {"baseline": config}
    summary = {"schema_version": "0.3", "suite": suite, "backend": client.backend, "runs": []}
    for variant, selected_config in variants.items():
        for task in ("intent", "agent", "pipeline"):
            current_audio = audio_processor
            if current_audio is not None:
                from .intentgpt.audio import AudioProcessor

                current_audio = AudioProcessor(selected_config, sound_classifier=current_audio.sound_classifier,
                                               speech_transcriber=current_audio.speech_transcriber)
            report = evaluate(samples, client, selected_config, task=task, dataset_sha256=dataset_sha256,
                              audio_processor=current_audio)
            relative = f"{variant}/{task}"
            write_report(report, root / relative)
            reference = "modality_reference" if selected_config.removed_modalities else "baseline"
            summary["runs"].append({"variant": variant, "reference": reference, "task": task, "path": relative,
                                    "metrics": report["metrics"], "errors": report["errors"]})
    (root / "summary.json").write_text(dumps_pretty(summary) + "\n", encoding="utf-8")
    return summary
