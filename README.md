# OmniIntents 0.3.0

[![Reproducible delivery](https://github.com/apromisedland/OmniIntents_/actions/workflows/ci.yml/badge.svg)](https://github.com/apromisedland/OmniIntents_/actions/workflows/ci.yml)

A paper-aligned Python reference for **OmniIntents: Enhancing Intent Prediction and Agent Selection through Real-World Multimodal Inputs and LLM Integration** (CHI 2025).

[Project page](https://apromisedland.github.io/omniintents-paper-page/) · [Manuscript](https://apromisedland.github.io/omniintents-paper-page/assets/omniintents-paper.pdf) · [Supplement](https://apromisedland.github.io/omniintents-paper-page/assets/supplement.pdf) · [Reproduction guide](docs/reproduction.md) · [Paper alignment](docs/paper-alignment.md)

## What this release delivers

- Eight general / 21 specific intent labels, up to three ranked candidate intents, and four paper agent categories.
- Separate prediction and user selection, followed by intent-conditioned task planning and implicit or explicit agent selection.
- Structured visual/audio processing, credibility prompts, original-image IFAS, and isolated multi-round sessions.
- An installed CLI, versioned JSONL data format, fixed-universe metrics, experiment manifests and 54 experiment/task combinations.
- A network-free deterministic demonstration and protocol-tested adapters for compatible model APIs, Google Speech and local YAMNet.
- Locked base/development dependencies, wheel/sdist checks and a Windows/Linux Python 3.11/3.12 CI matrix.

This release reconstructs the published method and provides an executable experimental workflow. It does **not** reproduce the original paper's numerical results. The private diary dataset, original splits, trained memory weights and historical model snapshots are not included. History summaries are deterministic reference summaries, not the paper's LSTM/seq2seq model. Mock outputs and their scores are synthetic software checks, not research evidence. Hardware execution, HoloLens/UDP integration, BERT/LLM training and human-study reruns are outside this release.

## Install in a project environment

Python 3.11 or 3.12 is the tested target. No shell activation is required.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation --no-deps -e .
.\.venv\Scripts\python.exe -m omniintents demo --utterance "What is the weather tomorrow?"
.\.venv\Scripts\python.exe -m omniintents reproduce --suite smoke --out runs/smoke
```

Linux/macOS:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation --no-deps -e .
.venv/bin/python -m omniintents demo --utterance "Generate an original illustration"
.venv/bin/python -m omniintents reproduce --suite smoke --out runs/smoke
```

For a minimal consumer installation, install the built wheel in a fresh environment with `pip install -c requirements.lock dist/omniintents-0.3.0-py3-none-any.whl`. The lock also contains test/build tools, but constraints install only the wheel's required dependencies. Torch and TensorFlow are not base dependencies.

The installed `omniintents` command and `python -m omniintents` expose identical commands. All examples below assume the project interpreter or installed command is on the explicitly selected path.

```sh
omniintents validate-data
omniintents evaluate --task intent --out runs/intent
omniintents evaluate --task agent --out runs/agent
omniintents evaluate --task pipeline --out runs/pipeline
omniintents ablate --out runs/ablations
```

Output directories must be new or empty. Local result files can contain input text, labels inferred by the model, and local media paths; `runs/`, `private/` and `.env` files are ignored by Git.

## Python API

```python
from omniintents import MultimodalInput, OmniIntentsPipeline
from omniintents.llm import MockLLMClient

pipeline = OmniIntentsPipeline(MockLLMClient())
prediction = pipeline.predict(MultimodalInput(
    utterance="unclear speech",
    speech_transcript="unclear speech",
    speech_confidence=0.2,
    brightness=0.8,
    hand_target="wine glass",
    eye_target="wine glass",
))

for index, candidate in enumerate(prediction.candidates):
    print(index, candidate.specific_label, candidate.description)

result = pipeline.plan_selected(prediction, selected_index=0)
print(result.to_json_dict())
```

The chosen index represents the user's selection. `pipeline.run(input)` proceeds automatically only for a single unambiguous candidate. It returns `needs_selection` or `needs_clarification` otherwise. For batch experiments, `selection="top1"` explicitly chooses the highest-ranked alternative. The package generates plans and recommendations; it does not execute those plans.

## Real service configuration

Use the explicit `openai-compatible` backend with `OPENAI_API_KEY`, `OPENAI_MODEL`, and optionally `OPENAI_BASE_URL` / `OPENAI_VISION_MODEL`. The model has no implicit default. `.env.example` documents names; the program does not automatically load dotenv files.

```sh
omniintents evaluate --backend openai-compatible --model YOUR_MODEL_ID \
  --dataset private/diary.jsonl --split test --task intent --out runs/live-intent
```

Using a live backend sends that run's inputs and public prompt examples to the configured provider. Credentials are read from the environment and excluded from reports. Use `--token-parameter max_completion_tokens` or `--send-seed` only when supported by that provider. The compatible endpoint uses Chat Completions with JSON-object output.

See [service setup](docs/reproduction.md#optional-services) for optional Google Speech and YAMNet configuration. Live services and real model quality were not exercised for this delivery; their protocols are tested using controlled doubles.

## Verify a delivery

```sh
python -m pytest -q
python -m omniintents reproduce --suite ablations --out runs/verification
python -m build --no-isolation
python tools/verify_distribution.py
```

Run these using the project interpreter. The distribution checker creates a separate temporary environment, installs the wheel, runs the console command and smoke suite outside the source tree, and rebuilds the sdist.

## Documentation

- [Paper correspondence and source discrepancies](docs/paper-alignment.md)
- [Reproduction, experimental conditions and timing](docs/reproduction.md)
- [JSONL dataset contract](docs/data-contract.md)
- [Metric definitions and failure accounting](docs/metrics.md)
- [Migration from 0.2](docs/migration-0.3.md)
- [Release changes](CHANGELOG.md)

MIT applies to this repository's software. Paper-derived examples are attributed in their resources; no ownership or new license over the original manuscript or private participant data is asserted.
