# SAFE-Net v2 — Reproducible Pipeline

This is a full rewrite of the SAFE-Net intrusion detection pipeline,
built specifically to resolve every issue raised by the two conference
reviews (see `REVIEWER_RESPONSE_NOTES.md`). Read `NOTES_ON_REDESIGN.md`
first — it explains the one big architectural change (no more LSTM /
sequence windowing) and why it was necessary.

## What changed vs. the original submission

- **No hardcoded personal paths anywhere.** Everything is a CLI flag
  with a relative default (`--data_dir`, `--artifacts_dir`, ...).
- **Dense Autoencoder instead of LSTM Autoencoder, no sliding windows.**
  See `NOTES_ON_REDESIGN.md`. This also makes the latency numbers honest
  (one row in, one prediction out — no more `seq_len` division tricks).
- **One `results.json` per model = single source of truth.** The paper's
  text, Table II, and confusion-matrix figure should all be transcribed
  from `results_summary.md`, generated at the end of the run — this is
  the direct fix for the reviewer's "text/table/figure contradict each
  other" complaint.
- **Reproducible sampling.** `sampling_manifest.json` documents exactly
  how the training sample was drawn from the full dataset (seeded,
  per-class targets logged).
- **Ablation study**: raw-features-only vs. AE-latent-only vs. fused.
- **Full baseline suite**: LogisticRegression, RandomForest, XGBoost,
  IsolationForest, OneClassSVM, CNN1D, CNN+BiLSTM+Attention, and a small
  Transformer encoder — all on the identical split/scaler.
- **Honest latency benchmark** with a throughput-vs-load scaling curve
  (Reviewer #1's request), using real test rows, not `torch.randn`.
- **Leave-one-attack-out generalization test** (Reviewer #2's request):
  does the model still flag attack types it was never trained on?
- **Per-attack-type recall table + a coarse-category multiclass head**,
  so the binary framing is no longer the *only* lens on the results
  (Reviewer #1's "trivial binary classification" complaint).
- **Publication-ready figures**: 300 DPI, large fonts, tight layout,
  sized to drop straight into a two-column paper with no enlargement.

## Setup

```bash
pip install -r requirements.txt --break-system-packages   # or use a venv
```

## Running

### 1. Smoke test first (always do this before the real run)
```bash
python run_all.py --quick
```
This generates a tiny synthetic dataset matching your real schema and
runs the entire pipeline in under two minutes, so you can catch any
environment problems before committing hours of compute.

### 2. The real run
```bash
python run_all.py --data_dir "C:\Users\tomoy\Downloads\MERGED_CSV\MERGED_CSV" --artifacts_dir ./artifacts
```
Point `--data_dir` at the folder containing your `Merged*.csv` files.
Everything is written to `artifacts/<run_name>/`.

You can also run any single stage on its own (useful if one stage fails
and you don't want to redo the others):
```bash
python src/train_main.py --data_dir ... --run_name run_20260804_120000
python src/baselines_classical.py --data_dir ... --run_name run_20260804_120000
python src/make_figures.py --data_dir ... --run_name run_20260804_120000
```
Just reuse the same `--run_name` so everything lands in one folder.

### Estimated runtime
On the hardware described in your paper (2-core/4-thread, 2.5GHz, no
GPU), expect the full run on 1.4M sampled rows to take **several hours**,
dominated by the classical `RandomForest`/`XGBoost` fits and the AE/
classifier training epochs. The `--quick` smoke test takes under two
minutes and is what you should use to validate the code first.

## What you get, and why you never need to re-run this

Everything is saved under `artifacts/<run_name>/`:

```
main/                    best_ae.pt, best_clf_fused.pt, best_clf_rawonly.pt,
                          best_clf_aeonly.pt, scaler.pkl, all results_*.json,
                          training curves, class_balance.json, run_manifest.json
baselines_classical/      results_<Model>.json for every classical baseline
baselines_deep/           results_<Model>.json + checkpoints for CNN1D,
                          CNN_BiLSTM_Attention, TinyTransformer
latency/                  latency_results.json (headline + full scaling curve)
generalization_loao/      loao_results.json
multiclass/                results_multiclass.json, best_clf_multiclass.pt
figures/                   every PNG figure, 300 DPI, print-ready
results_summary.md         <- open this file for every number the paper needs
sampling_manifest.json     <- documents exactly how the 1.4M-row sample was built
label_counts_full_dataset.json   <- label distribution across the FULL dataset
```

Model checkpoints (`.pt` files) let you reload any trained model without
retraining, in case you need additional analysis later:
```python
import torch
from src.models import DenseAutoencoder, FusionClassifier
ae = DenseAutoencoder(input_dim=39, hidden=128, latent=32)
ae.load_state_dict(torch.load("artifacts/<run_name>/main/best_ae.pt"))
```

## Repository hygiene for your resubmission

Include this whole folder in your camera-ready/resubmission repository
link, with a populated `artifacts/<run_name>/` from your real run (or at
least `results_summary.md`, the `figures/` folder, and all `results_*.json`
files, if the full checkpoints are too large for your repo host). This
directly answers the reviewer complaint that the original repo had no
README and unrunnable, unreproducible paths.
