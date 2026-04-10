from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from ..llm.base import LLMClient
from ..types import MultimodalInput
from ..utils.image_utils import blur_score_variance_of_laplacian, load_image, smoothed_brightness, choose_keyframes


class ObjectDetector(Protocol):
    def detect_objects(self, image_paths: Sequence[str], mm: MultimodalInput) -> List[str]:
        ...


class ContextDetector(Protocol):
    def detect_context(self, image_paths: Sequence[str], mm: MultimodalInput) -> Tuple[str, str]:
        ...


class HandEyeTargetDetector(Protocol):
    def detect(self, image_paths: Sequence[str], mm: MultimodalInput) -> Tuple[str, str, str]:
        ...


@dataclass
class InjectedObjectDetector:
    def detect_objects(self, image_paths: Sequence[str], mm: MultimodalInput) -> List[str]:
        return mm.objects or []


@dataclass
class InjectedContextDetector:
    def detect_context(self, image_paths: Sequence[str], mm: MultimodalInput) -> Tuple[str, str]:
        return (mm.context_location or "unknown", mm.context_activity or "unknown")


@dataclass
class InjectedHandEyeTargetDetector:
    def detect(self, image_paths: Sequence[str], mm: MultimodalInput) -> Tuple[str, str, str]:
        return (mm.hand_state or "unknown", mm.eye_state or "unknown", mm.interaction_target or "unknown")


@dataclass
class VisualProcessor:
    """Extract visual information into structured text fields (paper 6.1.1.1).

    Implemented fields:
    - brightness: relative luminance (Eq.1) with temporal smoothing across 5 keyframes
    - hand_state, eye_state, interaction_target (injected or detector)
    - context: location/activity (injected or detector)
    - objects: (injected or detector)
    - optional: scene_description (via multimodal LLM)
    - optional: image_quality (blur score) for debugging/extension (NOT used by default thresholds)
    """

    llm: LLMClient
    object_detector: Optional[ObjectDetector] = None
    context_detector: Optional[ContextDetector] = None
    hand_eye_target_detector: Optional[HandEyeTargetDetector] = None

    describe_with_llm: bool = True
    max_describe_frames: int = 1  # keep low by default

    def __post_init__(self) -> None:
        self.object_detector = self.object_detector or InjectedObjectDetector()
        self.context_detector = self.context_detector or InjectedContextDetector()
        self.hand_eye_target_detector = self.hand_eye_target_detector or InjectedHandEyeTargetDetector()

    def process(self, mm: MultimodalInput) -> Dict[str, object]:
        image_paths: Sequence[str] = mm.image_paths or []
        keyframes = choose_keyframes(image_paths, k=5) if image_paths else []
        brightness = smoothed_brightness(keyframes, k=5) if keyframes else 0.0

        # Optional quality metric on first frame
        blur_score = None
        if keyframes:
            try:
                img0 = load_image(keyframes[0])
                blur_score = blur_score_variance_of_laplacian(img0.rgb)
            except Exception:
                blur_score = None

        # Scene description (optional)
        scene_description = ""
        if self.describe_with_llm and keyframes:
            prompt = (
                "Describe the scene in detail, focusing on objects, context (location/activity), "
                "and any interaction target the user may be referring to."
            )
            # Only describe a few frames to limit cost.
            descs = []
            for p in keyframes[: self.max_describe_frames]:
                try:
                    descs.append(self.llm.vision_describe(p, prompt, max_tokens=250).text.strip())
                except Exception:
                    continue
            scene_description = "\n".join([d for d in descs if d])

        # Detectors / injected metadata
        objects = self.object_detector.detect_objects(keyframes, mm) if self.object_detector else (mm.objects or [])
        loc, act = self.context_detector.detect_context(keyframes, mm) if self.context_detector else (
            mm.context_location or "unknown",
            mm.context_activity or "unknown",
        )
        hand_state, eye_state, target = self.hand_eye_target_detector.detect(keyframes, mm) if self.hand_eye_target_detector else (
            mm.hand_state or "unknown",
            mm.eye_state or "unknown",
            mm.interaction_target or "unknown",
        )

        visual_text: Dict[str, object] = {
            "brightness": {
                "value": float(brightness),
                "text": f"The current environmental brightness is {brightness:.3f}, on a scale of 0 to 1.",
            },
            "scene_description": scene_description,
            "hand_state": hand_state,
            "eye_state": eye_state,
            "interaction_target": target,
            "context": {"location": loc, "activity": act},
            "objects": objects,
        }
        if blur_score is not None:
            visual_text["image_quality"] = {"blur_score_var_laplacian": float(blur_score)}
        return visual_text
