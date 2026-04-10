````markdown
# OmniIntents Pipeline


## Quick start

```bash
python -m scripts.run_demo --utterance "query information about this book" --objects "book" --location "library" --trace_out trace.json
````

Text-only input is also supported:

```bash
python -m scripts.run_demo --utterance "navigate to the entrance"
```

---

## Project structure

```
omniintents/
  pipeline.py              # End-to-end orchestration + Trace
  config.py                # Thresholds / switches / cost table
  types.py                 # Data structures (Intent / TaskPlan / Agent / Trace)
  utils/
    audio_utils.py         # RMS / dB / 20fps, etc.
    image_utils.py         # Brightness (paper formula) + optional quality metrics
    json_utils.py          # JSON extraction and validation helpers
    validate.py            # Structural validation and safe fallback
  llm/
    base.py                # LLMClient interface
    mock.py                # Offline mock LLM (controlled JSON output)
    openai_compatible.py   # Example adapter for OpenAI-compatible APIs (optional)
  intentgpt/
    intentgpt.py           # IntentGPT main pipeline
    visual.py              # Visual structuring (brightness / scene / context / objects)
    audio.py               # Audio structuring (volume / classification / speech)
    credibility.py         # Credibility assessment and attention hints
    memory.py              # History storage / summarization / retrieval
    cot.py                 # Few-shot + CoT prompt construction
    taxonomy.py            # Intent taxonomy & definitions
    heuristics.py          # Intent prediction fallback strategies
  taskgpt/
    taskgpt.py             # TaskGPT (with IFAS)
    ifas.py                # Intent-Focused Attention Shifter
  agentgpt/
    agentgpt.py            # AgentGPT
    decision_tree.py       # Implicit/explicit trees + cost optimization
    eval.py                # Evaluation tools (confusion matrix, etc.)
scripts/
  run_demo.py
  evaluate_decision_trees.py
tests/
  test_pipeline.py
  test_credibility.py
  test_agentgpt.py
```

---

## How to integrate real services 

* **LLM**: Implement `omniintents.llm.base.LLMClient` and replace `MockLLMClient`
* **Speech**: Implement `SpeechTranscriber` in `intentgpt/audio.py`
* **YAMNet**: Implement `SoundClassifier`
* **Vision detection**: Implement/inject `ObjectDetector`, `ContextDetector`, and `HandEyeTargetDetector` in `intentgpt/visual.py`

---

## License

MIT

```
```
