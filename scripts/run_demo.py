from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

from omniintents import MultimodalInput, OmniIntentsPipeline
from omniintents.llm import MockLLMClient
from omniintents.utils.json_utils import dumps_pretty


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run OmniIntents pipeline demo.")
    p.add_argument("--utterance", type=str, default="", help="User utterance text.")
    p.add_argument("--image", type=str, action="append", default=[], help="Path to an image (repeatable).")
    p.add_argument("--audio", type=str, default=None, help="Path to an audio file.")
    p.add_argument("--objects", type=str, default=None, help="Comma-separated object list (optional).")
    p.add_argument("--location", type=str, default=None, help="Context location (optional).")
    p.add_argument("--activity", type=str, default=None, help="Context activity (optional).")
    p.add_argument("--hand_state", type=str, default=None, help="Injected hand_state (optional).")
    p.add_argument("--eye_state", type=str, default=None, help="Injected eye_state (optional).")
    p.add_argument("--target", type=str, default=None, help="Injected interaction_target (optional).")
    p.add_argument("--speech_transcript", type=str, default=None, help="Injected speech transcript (optional).")
    p.add_argument("--speech_confidence", type=float, default=None, help="Injected speech confidence (optional).")
    p.add_argument("--trace_out", type=str, default=None, help="Write debug trace JSON to this path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    image_paths: List[str] = [p for p in args.image if p]

    objects = None
    if args.objects:
        objects = [o.strip() for o in args.objects.split(",") if o.strip()]

    mm = MultimodalInput(
        utterance=args.utterance or "",
        image_paths=image_paths,
        audio_path=args.audio,
        objects=objects,
        context_location=args.location,
        context_activity=args.activity,
        hand_state=args.hand_state,
        eye_state=args.eye_state,
        interaction_target=args.target,
        speech_transcript=args.speech_transcript,
        speech_confidence=args.speech_confidence,
    )

    llm = MockLLMClient()
    pipeline = OmniIntentsPipeline(llm=llm)
    result, trace = pipeline.run_with_trace(mm)

    print(dumps_pretty(result.to_json_dict()))

    if args.trace_out:
        Path(args.trace_out).write_text(dumps_pretty(trace.to_json_dict()), encoding="utf-8")
        print(f"\n[trace written to {args.trace_out}]")  # noqa: T201


if __name__ == "__main__":
    main()
