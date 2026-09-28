# Paper correspondence and limits

Sources: the [main manuscript](https://apromisedland.github.io/omniintents-paper-page/assets/omniintents-paper.pdf) and [supplement](https://apromisedland.github.io/omniintents-paper-page/assets/supplement.pdf) supplied with the project. These are anonymous manuscripts with placeholder publication metadata, not verified final proceedings files.

| Published component | 0.3 implementation | Important boundary |
| --- | --- | --- |
| Table 1: eight general and 21 specific intents | Canonical exact labels and validated parent mapping | No invented Physical Help or Unknown class |
| Appendix A.1: ranked alternatives, at most three | Ranked candidate descriptions and labels; complementarity/ambiguity; concise evidence summary | Confidence values are uncalibrated; multiple descriptions may share a class |
| Appendix A.1: seven examples | Seven attributed JSON adaptations, bundled in wheel | Example probabilities are illustrative additions; contradictory source reasoning is documented |
| Section 6.1: visual structuring | Original-image extraction of activity, location, objects, eye/hand state and separate targets | Gaze from imagery is an approximation; not a calibrated eye tracker |
| Brightness equation | Linear RGB weighted mean / 255, smoothed over up to five uniformly spaced supplied frames | Uses the paper equation, not gamma-corrected WCAG luminance; direct video decoding is not included |
| Section 6.1: audio | 20 Hz frame RMS, voiced threshold, dB conversion, top-three sounds, ASR confidence | Frame rounding and weighted voiced RMS are documented reconstruction choices |
| Appendix A.1: CBAS | Brightness > 0.2 and ASR confidence > 0.8; explicit textual reliability report | Equality is conservatively unreliable because the source leaves equality undefined |
| Sound event reporting | Highest of three events only when confidence > 0.70 | Separate from ASR credibility |
| Contextual Memory Tracker | Per-session previous location/activity/selected intent; configurable 0–5 rounds, default 2 | Deterministic replacement, not trained LSTM/seq2seq reproduction; no similarity retrieval |
| Appendix A.2: TaskGPT/IFAS | Three task examples, selected-intent focus prompt, original images sent again when available | Pre-extracted mode is explicitly labeled and cannot recover unseen details; no internal attention-weight edits |
| Appendix A.3: AgentGPT | Four examples; implicit model selection and explicit four-boolean decision tree | Relative costs are engineering ranks, not paper-measured prices |
| Section 7: experiment workflow | Separate intent, given-task agent and pipeline evaluations; controlled ablations and provenance | New execution on available data, not automatic recreation of original experimental subsets |
| Appendix D / Section 8: user studies | Public scenario descriptions inform reference examples | No reconstructed participant scores, cognitive-load scores or hardware task-success claims |

## Preserved discrepancies and explicit decisions

- The paper calls intent ACC exact match, but its formula is the overlap coefficient. Reports expose both `overlap_acc` and true `strict_set_accuracy`.
- The main text says five intent examples; Appendix A.1 contains seven. This release uses all seven. No automatic assumption that a new dataset has exactly 371 intent or 374 agent evaluation entries is made.
- Agent example wording suggests an ambiguous count per category; Appendix A.3 actually supplies four examples, which are used.
- Appendix A.1 has phone/book/wine inconsistencies, holding/pointing inconsistencies and a drawing example labeled Request Collaborator. The structured examples retain published inputs and labels, attach notes, and use short evidence summaries rather than propagating contradictory narrative text.
- Table 3 gives GPT-4o specific macro F1 as 63.4; Figure 10's full-input configuration gives 63.1. Both are preserved in `data/paper_reported.json`, never substituted for newly measured results.
- Figure 10 labels and the accompanying prose are not perfectly aligned. This release includes context, objects, hand/eye, speech, targets and audio-label removal conditions; the standalone target condition is an engineering extension.
- Appendix A.3's commented decision code rejects speech=False, conflicting with its physical-task example. The explicit reference tree prioritizes physical interaction, then creativity, then visual expression, then voice/text. A final capability check rejects an unsuitable agent.
- A visual observation made by IntentGPT does not automatically require visual embodiment from the executing agent. Navigation guidance is distinct from a robot moving its body.
- A task needing both physical manipulation and creativity can exceed all four configured agents. The program returns `no_suitable_agent`; it does not silently invent a fifth type.
- Typed text is usable language evidence but does not receive a fabricated ASR confidence.
- Reported dB from pre-extracted records has unspecified calibration; waveform calculations are labeled dBFS plus an explicit calibration offset, never assumed physical dB SPL.

## Reproduction claims

The public synthetic dataset has 28 hand-authored entries covering all labels, all agents, one six-round session and an ambiguous situation. It is deliberately easy for deterministic demo rules and is not representative of research accuracy. Few-shot example source IDs are reserved; normalized exact utterance duplicates are rejected in evaluation data.

No original private records, consent documents, split manifests, model weights or API response archives are redistributed. No learned model, BERT baseline, LLM fine-tuning, original neural memory, HoloLens UI or robot control is claimed to be recreated. Future real-data runs should be labeled new runs and preserve the same manifest discipline.
