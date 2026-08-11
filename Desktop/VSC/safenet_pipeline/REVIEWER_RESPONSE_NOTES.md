# Reviewer Comment -> Pipeline Artifact Map

Use this as the basis for your response-to-reviewers letter. Every row
points at a concrete, generated artifact — not a promise.

## Review #1

| Comment | Addressed by |
|---|---|
| No README, unreproducible hardcoded path | `README.md`, `src/config.py` (no hardcoded paths, everything CLI-driven), `sampling_manifest.json` |
| Text / table / confusion-matrix contradict each other | `results_summary.md` — single source of truth, auto-generated from `results_fused.json`; every number in the paper should be transcribed from here |
| No latency-vs-load scaling graph | `fig_latency_scaling.png` + `latency_results.json` (batch sizes 1→1024, mean±std, real test rows) |
| Trivial binary classification collapses 33 attack types | `fig_per_attack_recall.png` + `per_attack_type_recall.json` (full 33-class recall table); `fig_multiclass_confusion.png` + `results_multiclass.json` (coarse-category multiclass head) |

## Review #2

| Comment | Addressed by |
|---|---|
| No competitive recent DL baselines (attention/transformer) | `baselines_deep.py` → CNN1D, CNN+BiLSTM+Attention, TinyTransformer, all in `fig_roc_comparison.png` / `fig_baseline_bar_comparison.png` |
| Possible temporal leakage in windowed sequences | Windowing removed entirely — see `NOTES_ON_REDESIGN.md`. Not mitigated, **eliminated** at the architecture level. |
| No ablation isolating AE vs. classifier contribution | `fig_ablation.png` + `ablation_summary.json` (raw-only / AE-only / fused, 3-way comparison) |
| No evaluation on unseen/novel attack types | `generalization_loao.py` → `fig_generalization_loao.png` + `loao_results.json` |

## Honesty checkpoint — do not omit this from the resubmission

The classical-baseline comparison in the ORIGINAL submission claimed
"None of these models matched the accuracy or effectiveness of the
proposed SAFE-Net," which was contradicted by the authors' own
`baseline_results.json` (RandomForest slightly outperformed the reported
SAFE-Net numbers). The new `results_summary.md` explicitly instructs:
*"Report every row above in the paper, including any baseline that
matches or exceeds SAFE-Net v2 — do not omit inconvenient results."*
If a classical baseline again matches/exceeds SAFE-Net v2 on raw
accuracy, frame the contribution around what SAFE-Net v2 genuinely adds
instead (unsupervised reconstruction signal, single-row real-time
inference cost, generalization to unseen attack types per the LOAO
results) rather than claiming an accuracy win that isn't there — this
is far more defensible to a reviewer than an unsupported superiority
claim.
