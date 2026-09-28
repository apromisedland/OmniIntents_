# Metrics

All generated metrics use the scale **0 to 1**. The source-only `paper_reported.json` stores the paper's 0–100 values and seconds; it is never read to populate measured results.

## Intent prediction

For non-empty true label set G and predicted set P:

- **overlap_acc** = mean(|P intersect G| / min(|P|, |G|)); score zero when P is empty.
- **strict_set_accuracy** = mean(P equals G).
- **macro_f1** = mean per-class F1 over all 21 specific classes, or all eight general classes after parent mapping. Absent classes have F1 zero.
- Per-class F1 = 2 TP / (2 TP + FP + FN), with zero for a zero denominator.

The paper calls overlap ACC exact match. They differ: predicting one correct member of a three-label true set yields overlap 1 but strict accuracy 0. Returning several descriptions from the same category does not duplicate the label when computing metrics.

## Agent selection

Agent prediction is single-label over all four canonical categories. Reports include accuracy, fixed-four-class macro F1, support-weighted F1 and both raw/row-normalized confusion matrices. The extra `invalid` prediction column preserves failed or rejected predictions; it is not a fifth agent class.

The original Table 4 does not define F1 averaging. Both macro and weighted F1 are exposed; neither is presented as a proven reconstruction of its unnamed F1.

## Errors and stages

Unknown labels, malformed JSON, bad task capabilities, truncated responses and provider failures are explicit errors. No silent heuristic recovery is allowed during evaluation. Failed intent predictions become empty sets; failed agent predictions become null. They remain in all relevant denominators. Ground-truth input errors abort validation before evaluation rather than silently skipping records.

The intent task evaluates IntentGPT only. The agent task consumes an explicitly supplied task plan; it does not consume predicted intent. The pipeline task uses its own predicted intent and task plan, with explicit top1 selection. If later planning fails, earlier valid intent classification is still measurable; agent selection fails for that sample. `pipeline_completion_rate` records software completion without implying real-world task success or semantic planning accuracy.

Mean model-request timing includes adapter retries, transport and response completion. Mean sample timing also includes preprocessing and orchestration. Offline mock latency is software overhead and cannot be compared with historical GPT-4o latency. No confidence intervals or human rating statistics are fabricated.
