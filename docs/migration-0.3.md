# Migration from 0.2 to 0.3

This is an intentional research-interface change. Re-evaluate downstream assumptions instead of mapping old predictions into the new label space silently.

| 0.2 | 0.3 |
| --- | --- |
| One intent returned immediately | Up to three ranked candidates plus complementarity/ambiguity; select before planning if ambiguous |
| Approximate Request Search / Physical Help / Unknown labels | Exact 21 specific labels and derived eight general categories; no evidence returns zero candidates |
| Five agent categories | Four paper categories; digital_assistant and multimodal_assistant have no unconditional one-to-one mapping |
| voice_assistant / ar_companion / physical_robot | voice_assistant / ar_agent / physical_agent; generative_ai_agent added |
| Generic navigation capability | navigation_guidance and physical_movement distinguished |
| Shared implicit single-session history | session_id, sequence_index and timezone-aware timestamp; deterministic 0–5 prior rounds |
| LSTM placeholder and optional Torch import | No Torch dependency; explicit deterministic reference memory |
| Brightness 0.15 / ASR 0.70 | Supplement thresholds 0.2 / 0.8 with strict greater-than; sound-event threshold stays 0.70 |
| Prompt-marker complete/vision_describe interface | LLMClient.generate(LLMRequest) with explicit stage, system, payload and image_paths |
| Invalid output silently falls back | Typed errors and per-sample failure accounting; select MockLLMClient explicitly for offline demonstrations |
| Trace generally enabled | Trace disabled by default; run_with_trace returns an empty trace unless enabled |
| Source-relative example files | Installed package resources included in wheel and sdist |

`pipeline.run(input)` remains the convenience entry point, but callers must inspect `result.status`. `result.intent` is the selected candidate (or None), while `result.prediction.candidates` contains all predictions. Serialized results use `schema_version="0.3"`, `prediction`, `selected_index`, optional `task_plan`/`agent`, and explicit `errors`.

Use `predict(input)` followed by `plan_selected(prediction, index)` for interactive selection. Use `run(input, selection="top1")` only when that automatic batch policy is intended. `reset_session(session_id)` clears stored context. Supply strictly increasing indices to commit additional turns.

`MultimodalInput.interaction_target` is replaced by `eye_target`, `hand_target` and `voice_target`. This prevents agreement being assumed when channels point to different objects. Metadata brightness and audio event scores can be supplied without media.

Legacy `python -m scripts.run_demo` forwards to the installed demo CLI. Prefer `python -m omniintents`, which works after wheel installation outside the source tree. Legacy decision-tree evaluation forwards to the full ablation runner and requires an output directory; it no longer mislabels the first candidate as a cost-optimized prediction.
