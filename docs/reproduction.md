# Reproduction workflow

## Environment

Use a fresh Python 3.11 or 3.12 project environment, install `requirements.lock`, then install the repository with `--no-deps --no-build-isolation -e .`. Invoke pip, pytest, build and CLI through that environment's absolute or relative interpreter path. Do not install project dependencies into a shared bundled runtime.

The lock pins the base and development dependency closure. Optional service extras are separately installed and are not part of the locked offline environment. CI covers Windows/Linux and both Python versions, checks package data, and installs the wheel outside the source checkout.

## Offline verification

```sh
python -m omniintents validate-data
python -m omniintents reproduce --suite smoke --out runs/smoke
python -m omniintents reproduce --suite ablations --out runs/ablations
```

The smoke suite runs intent prediction, given-task agent selection and full pipeline evaluation separately. The ablation suite runs 18 configurations across those three tasks, yielding 54 runs. Conditions irrelevant to a task are retained as no-op controls; for example, IFAS does not change standalone intent prediction.

Every run contains:

- `manifest.json`: package/source fingerprint, Python/platform/dependency versions, configuration, dataset hash, prompt resource hashes and sample identities.
- `predictions.jsonl`: per-sample predictions, pipeline outputs, explicit errors, request fingerprints, backend/model/usage metadata and timings.
- `metrics.json`: fixed-label metrics, failure counts, claim classification and mean latencies.
- The suite root also contains `summary.json`, linking each configuration/task pair and its comparison reference.

No inference response cache is used. Repeated mock runs have identical functional outputs after removing `duration_s` and `timing`. Seed and temperature are recorded, but live providers are not guaranteed deterministic.

## Experimental conditions

| Family | Conditions and comparison |
| --- | --- |
| Main pipeline | Full available structured input, CBAS on, IFAS on, two prior rounds, implicit agent selection |
| Modality controls | `modality_reference` removes raw media, free scene descriptions and all historical fields from the control and every modality treatment |
| Modality removal | Remove context, objects, hand/eye states and targets, speech, all targets, or audio labels; compare with `modality_reference` |
| CBAS | Same input and examples, reliability report removed, compared with baseline |
| Memory | 0–5 prior rounds; ordered sessions; compare within the same sequence dataset |
| IFAS | Same selected intent and image inputs, focus instruction enabled/disabled; raw-image vs pre-extracted mode recorded |
| Agent strategy | Implicit model recommendation vs model-generated capability flags and explicit decision tree |

Modality experiments require pre-extracted features: they never silently re-extract masked information from original images or audio. The projector removes duplicate exact strings from remaining current-sample fields, including utterance text. It disables history in all modality controls because history labels or summaries can encode the removed observation. This is a deliberately conservative input-redaction experiment; removing duplicated evidence is not always a pure causal intervention on one sensor. Semantic paraphrases cannot be mechanically guaranteed absent, so inspect the input records before making scientific claims. Public examples remain constant task instructions, not current-sample evidence.

An externally supplied `task_plan` is used only for standalone agent evaluation. Pipeline evaluation creates its own task plan from the prediction; input labels never enter any model request. History uses previous predictions/selections, not evaluation ground truth. Sequences do not cross splits, and current/future entries cannot enter historical context.

TaskGPT/IFAS produces plans, not objective planning-quality metrics. The software reports pipeline completion separately from classification accuracy. It does not synthesize human Accuracy/Cognitive Load ratings from Appendix D.

## Private data

Private data integration was deferred for this release. The documented JSONL loader is ready for a later local dataset; use `private/` or an external directory. Select `--split test` explicitly. All CLI evaluation defaults use the clearly identified bundled `demo` split.

```sh
python -m omniintents validate-data --dataset private/diary.jsonl --split test
python -m omniintents evaluate --dataset private/diary.jsonl --split test --task pipeline --out runs/private-check
```

The default backend is mock even with private data. Such runs verify the workflow only. Model evaluation requires explicitly selecting a real backend. Predictions and trace files can contain private text and paths; they are not uploaded as CI artifacts. Keep them local.

## Optional services

### Compatible language/vision service

Configure `OPENAI_API_KEY`, `OPENAI_MODEL`, optional `OPENAI_BASE_URL` (API root including provider-specific version suffix), and optional `OPENAI_VISION_MODEL`. `.env` is not auto-loaded. No default model name is silently selected.

```sh
python -m omniintents evaluate --backend openai-compatible --model YOUR_MODEL_ID \
  --dataset private/diary.jsonl --split test --task intent --out runs/live
```

The adapter sends a system instruction and JSON payload via Chat Completions, with JSON-object output. Images use their actual MIME type. It supports `max_tokens` and the explicit `--token-parameter max_completion_tokens`; `--send-seed` is opt-in because compatibility varies. Unsupported options produce explicit provider errors. Timeouts, connection errors, 429 and 5xx have at most two retries by default; authentication and other request failures are not retried. Truncated and malformed completions fail visibly.

A temperature of zero is a reconstruction default, not a recovered paper setting. Provider/model snapshots, usage, attempts and wall-clock request durations are recorded. Request fingerprints and image hashes omit authorization headers. Streaming is not used; latency ends when the complete HTTP response is available.

Protocol reference: [official Chat Completions documentation](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).

### Google Cloud Speech

Install the `google-speech` extra in the project environment, configure Google Application Default Credentials externally, and select `--speech-backend google --language en-US`. The adapter uses synchronous recognition with a 60-second timeout and no hidden SDK retries. Use short audio supported by that API, or provide offline transcripts. Multiple recognized segments are combined; confidence is word-count weighted, an explicit reconstruction choice.

### YAMNet

Install the `yamnet` extra, supply a local YAMNet SavedModel and its 521-class `yamnet_class_map.csv`, then pass `--yamnet-model PATH --yamnet-labels PATH`. No automatic model download occurs. Input must already be mono 16 kHz audio. Class scores are averaged over time and the three highest are retained. The top event is reported only above 0.70.

Base installation reads 8/16/32-bit integer PCM WAV without extra packages. The `audio` extra adds SoundFile for non-WAV formats. For an offline WAV run, supply transcript/confidence and `sound_events` explicitly; absent services are reported instead of fabricated sound predictions.

Live Google/model calls and TensorFlow/YAMNet execution were not validated in this release. Contract tests use injected clients/models. Optional dependencies and model weights require a separate environment check before real experiments.

## Timing and failure accounting

Each request uses a monotonic wall-clock timer and includes retries. Per-sample timing additionally includes preprocessing and orchestration. These measurements are not sensor-to-robot completion time. A sample with an invalid model output, provider failure or missing required capability stays in the sample denominator. Stage-specific failures do not erase an earlier valid intent prediction from a pipeline report.

The paper's 371/374 ICL samples, 105 impaired-channel samples and 51 sequences of six entries are historical reported counts. New runs use their supplied IDs and split, and never assert those original selections without the missing source manifests.
