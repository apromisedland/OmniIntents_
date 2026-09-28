"""Installed command-line entry points for demonstration and experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

from .config import OmniIntentsConfig
from .dataset import load_dataset, select_split
from .errors import OmniIntentsError, ValidationError
from .evaluation import evaluate, reproduce, write_report
from .intentgpt.audio import AudioProcessor, GoogleSpeechTranscriber, YamNetClassifier
from .llm.mock import MockLLMClient
from .llm.openai_compatible import OpenAICompatibleClient
from .pipeline import OmniIntentsPipeline
from .prompts import load_resource
from .types import MultimodalInput
from .utils.json_utils import dumps_pretty, strict_loads


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OmniIntents research reference. Mock scores are software checks only.")
    parser.add_argument("--version", action="version", version="omniintents 0.3.0")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("demo", "validate-data", "evaluate", "ablate", "reproduce"):
        command = commands.add_parser(name)
        command.add_argument("--dataset", help="UTF-8 JSONL path; defaults to bundled synthetic smoke data")
        command.add_argument("--split", default="demo", choices=["demo", "test", "validation"])
        if name == "validate-data":
            continue
        command.add_argument("--backend", default="mock", choices=["mock", "openai-compatible"])
        command.add_argument("--model", help="Explicit live model name (or OPENAI_MODEL)")
        command.add_argument("--vision-model")
        command.add_argument("--base-url", help="API root including any /v1 suffix")
        command.add_argument("--config", help="JSON configuration path")
        command.add_argument("--history-rounds", type=int)
        command.add_argument("--agent-strategy", choices=["implicit", "explicit"])
        command.add_argument("--speech-backend", choices=["none", "google"], default="none")
        command.add_argument("--language", default="en-US")
        command.add_argument("--yamnet-model", help="Local YAMNet SavedModel directory")
        command.add_argument("--yamnet-labels", help="YAMNet 521-class CSV")
        command.add_argument("--timeout", type=float, default=60)
        command.add_argument("--retries", type=int, default=2)
        command.add_argument("--token-parameter", choices=["max_tokens", "max_completion_tokens"], default="max_tokens")
        command.add_argument("--send-seed", action="store_true", help="Enable only when the provider supports seed")
        if name == "demo":
            command.add_argument("--utterance")
            command.add_argument("--input", help="JSON MultimodalInput file")
            command.add_argument("--selection", help="top1 or a zero-based candidate index")
            command.add_argument("--trace-out", help="Optional local trace file")
        else:
            command.add_argument("--out", required=True, help="A new or empty output directory; results may contain private inputs")
        if name == "evaluate":
            command.add_argument("--task", default="intent", choices=["intent", "agent", "pipeline"])
        if name == "reproduce":
            command.add_argument("--suite", default="smoke", choices=["smoke", "ablations"])
    return parser


def _components(args: argparse.Namespace):
    values = strict_loads(Path(args.config).read_text(encoding="utf-8")) if args.config else load_resource("configs/default.json")
    if not isinstance(values, dict):
        raise ValidationError("Configuration must be an object")
    config = OmniIntentsConfig.from_json_dict(values)
    if args.history_rounds is not None:
        config = replace(config, history_rounds=args.history_rounds)
    if args.agent_strategy is not None:
        config = replace(config, agent_strategy=args.agent_strategy)
    client = MockLLMClient() if args.backend == "mock" else OpenAICompatibleClient(
        model=args.model, vision_model=args.vision_model, base_url=args.base_url,
        timeout_s=args.timeout, retries=args.retries, token_parameter=args.token_parameter, send_seed=args.send_seed,
    )
    if args.backend == "mock" and (args.speech_backend != "none" or args.yamnet_model):
        raise ValidationError("Mock mode does not initialize live speech or sound services")
    if bool(args.yamnet_model) != bool(args.yamnet_labels):
        raise ValidationError("Provide both --yamnet-model and --yamnet-labels")
    speech = GoogleSpeechTranscriber(args.language) if args.speech_backend == "google" else None
    sound = YamNetClassifier(args.yamnet_model, args.yamnet_labels) if args.yamnet_model else None
    return config, client, AudioProcessor(config, sound_classifier=sound, speech_transcriber=speech)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "validate-data":
            samples, digest = load_dataset(args.dataset)
            selected = select_split(samples, args.split)
            print(dumps_pretty({"valid": True, "rows": len(samples), "selected_rows": len(selected), "sha256": digest}))
            return 0
        config, client, audio = _components(args)
        if args.command == "demo":
            if args.input and args.utterance is not None:
                raise ValidationError("--input and --utterance are mutually exclusive")
            if args.input:
                data = MultimodalInput.from_json_dict(strict_loads(Path(args.input).read_text(encoding="utf-8")))
            elif args.utterance is not None:
                data = MultimodalInput(utterance=args.utterance)
            else:
                samples, _ = load_dataset(args.dataset)
                data = select_split(samples, args.split)[0].input
            selection = args.selection
            if selection is not None and selection != "top1":
                try:
                    selection = int(selection)
                except ValueError as error:
                    raise ValidationError("--selection must be top1 or an integer") from error
            if args.trace_out:
                config = replace(config, enable_trace=True)
            pipeline = OmniIntentsPipeline(client, config, audio_processor=audio)
            result, trace = pipeline.run_with_trace(data, selection=selection)
            print(dumps_pretty(result.to_json_dict()))
            if args.trace_out:
                target = Path(args.trace_out)
                if target.exists():
                    raise ValidationError("Trace output already exists")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(dumps_pretty(trace.to_json_dict()) + "\n", encoding="utf-8")
            return 1 if result.status.endswith("_error") or result.status == "no_suitable_agent" else 0
        samples, digest = load_dataset(args.dataset)
        samples = select_split(samples, args.split)
        if args.command == "evaluate":
            report = evaluate(samples, client, config, task=args.task, dataset_sha256=digest, audio_processor=audio)
            write_report(report, args.out)
            print(dumps_pretty({key: report[key] for key in ("claim", "sample_count", "metrics", "errors")}))
        else:
            suite = "ablations" if args.command == "ablate" else args.suite
            summary = reproduce(samples, client, config, args.out, dataset_sha256=digest, suite=suite, audio_processor=audio)
            print(dumps_pretty({"output": args.out, "suite": suite, "runs": len(summary["runs"]), "backend": client.backend}))
        return 0
    except (OmniIntentsError, OSError) as error:
        print(json.dumps({"error": getattr(error, "code", "filesystem_error"), "message": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
