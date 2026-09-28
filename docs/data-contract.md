# JSONL contract, schema 0.3

One UTF-8 JSON object per line. Labels are outside `input`. Each row is validated before any provider call.

```json
{
  "schema_version": "0.3",
  "sample_id": "example-001",
  "session_id": "session-A",
  "sequence_index": 0,
  "timestamp": "2026-01-01T12:00:00+00:00",
  "split": "test",
  "provenance": "Describe the authorized source here",
  "input": {
    "utterance": "Guide me to the engineering gallery.",
    "speech_transcript": "Guide me to the engineering gallery.",
    "speech_confidence": 0.95,
    "brightness": 0.6,
    "objects": ["sign", "door"],
    "context_location": "museum",
    "context_activity": "looking for the engineering gallery",
    "eye_state": "looking at a sign",
    "eye_target": "engineering sign",
    "hand_state": "pointing",
    "hand_target": "engineering sign",
    "voice_target": "engineering gallery"
  },
  "labels": {
    "intents": ["Location Guide"],
    "agent": "ar_agent"
  },
  "task_plan": {
    "goal": "Guide the user to the engineering gallery",
    "steps": [
      {
        "step_id": 1,
        "instruction": "Provide visual route guidance with spoken directions",
        "required_capabilities": ["speech_io", "visual_expression", "navigation_guidance"]
      }
    ]
  }
}
```

Required top-level fields are shown; `task_plan` is optional except for standalone agent evaluation. `labels.agent` is optional for intent-only evaluation and required for agent/pipeline evaluation. The task plan describes a given task for that baseline and is never passed to the intent predictor.

## Input fields

All modality fields are optional. Use null/absence for unknown values, not invented zero confidence or zero brightness.

| Fields | Type and interpretation |
| --- | --- |
| utterance | Typed or directly supplied text; a supplied speech transcript remains separately represented |
| image_paths | List of paths to ordered keyframe images; relative to the dataset directory |
| audio_path | Audio file path, relative to the dataset directory |
| speech_transcript, speech_confidence | String and finite number in [0,1], or null |
| brightness | Finite number in [0,1], or null; overrides image-computed brightness when explicitly supplied |
| volume_db | Finite pre-extracted value, units/calibration unspecified by source |
| sound_events | List of objects with string label and finite confidence in [0,1]; an empty list explicitly supplies no events |
| objects | List of strings |
| context_location, context_activity | Strings or null |
| eye_state, eye_target, hand_state, hand_target, voice_target | Separate strings or null; do not collapse conflicting targets |
| scene_description | Optional free description; excluded in conservative modality experiments |

Explicitly provided visual fields override model-extracted fields. Missing fields may be filled from raw images. Raw images are still extracted when present, even if partial metadata is injected; choose pre-extracted mode by omitting raw paths.

Timestamps must include a timezone. Sequence indices are non-negative integers, strictly increase within each session, and timestamps cannot decrease. Session entries cannot cross splits. Splits are `train`, `validation`, `test`, `fewshot` and `demo`; evaluation accepts only the last three applicable evaluation splits: validation, test and demo. Few-shot and training rows are never automatically promoted to model examples.

Sample IDs must be unique, may not use bundled prompt IDs, and identical normalized inputs are rejected even across sessions/splits. Evaluation utterances may not duplicate the normalized exact utterance of a bundled few-shot example. These checks catch exact duplicates, not paraphrase-level leakage; review any imported dataset's provenance.

Ground-truth intent sets require one to three distinct labels from the canonical 21 categories. General categories are derived from these labels. An empty predicted set is a valid abstention and receives zero overlap against non-empty truth; an empty ground-truth set is invalid.

Four agent values: `voice_assistant`, `ar_agent`, `generative_ai_agent`, `physical_agent`.

Capability vocabulary: `speech_io`, `text_io`, `vision_perception`, `visual_expression`, `navigation_guidance`, `physical_movement`, `physical_interaction`, `creativity`. Steps must have non-empty instructions and capabilities, with integer IDs 1, 2, ... . Unknown capabilities are errors.

## Scope and storage

The package ships only attributed public prompt adaptations and synthetic validation data. Put later private inputs under ignored `private/` or outside the checkout. Raw media and model checkpoints are user-supplied and not downloaded implicitly. Reports can contain derived participant information, so retain private runs locally.
