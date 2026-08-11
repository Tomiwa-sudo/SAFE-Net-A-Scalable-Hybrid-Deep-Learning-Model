# SAFE-Net-A-Scalable-Hybrid-Deep-Learning-Model-for-Real-Time-Intrusion-Detection-in-IoT-Networks

# SAFE-Net: A Scalable Hybrid Deep Learning Model for Real-Time Intrusion Detection in IoT Networks

SAFE-Net (**S**calable **A**utoencoder **F**usion **E**mbedding **Net**work) is a hybrid intrusion detection system for IoT network traffic, combining an unsupervised Autoencoder with a supervised fusion classifier. This repository contains the full, reproducible pipeline used to train, evaluate, and statistically validate SAFE-Net on the CICIoT2023 dataset, along with every baseline comparison, ablation study, and diagnostic analysis reported in the accompanying paper.

Every result in the paper is generated directly by this code — no numbers are hand-derived or manually transcribed. Every model checkpoint, training curve, and evaluation result is logged to disk automatically, so the paper's tables, figures, and text all trace back to the same source files and cannot drift out of sync with one another.

## Key Findings

- **94.78% accuracy, 0.9626 AUC** on a stratified 1,343,613-row sample of the full 45,019,243-row CICIoT2023 dataset.
- RandomForest and XGBoost achieve a small but **statistically significant** accuracy and AUC advantage over SAFE-Net (paired significance testing across multiple independently resampled seeds, p < 0.01) — reported directly rather than in favor of an unsupported superiority claim.
- Fusing the Autoencoder's latent representation with raw features provides **no statistically significant accuracy benefit** over raw features alone (p > 0.1).
- Detection reliability varies sharply by attack category: **volumetric flooding attacks (DDoS, DoS, Mirai) are detected with near-perfect recall**, including on attack types withheld entirely from training, while **reconnaissance, spoofing, and injection attacks are detected considerably less reliably**, in some cases below 25% recall.
- A benign-only, fully unsupervised Autoencoder anomaly detector does **not** generalize to unseen attacks better than the supervised model.
- A targeted intervention (engineered flow-repetition feature + oversampling) **meaningfully improves spoofing-attack detection specifically**, without helping reconnaissance or injection attacks — pointing toward a concrete, testable direction for future feature engineering.
- Genuine single-sample inference latency of **0.73 ms**, scaling to **311,196 samples/second** at a concurrent batch size of 1024, measured on commodity CPU hardware (no GPU).

Full methodology, statistical tests, and discussion are in the paper (`paper/`, or see citation below).

## Repository Structure

```
safenet_pipeline/
├── README.md                      # this file
├── NOTES_ON_REDESIGN.md           # why the architecture changed from an LSTM-based
│                                   # sequence model to a dense Autoencoder, and exactly
│                                   # what changed in the paper as a result
├── REVIEWER_RESPONSE_NOTES.md     # maps every methodological concern to the specific
│                                   # artifact/analysis that addresses it
├── requirements.txt
├── run_all.py                     # orchestrates the entire pipeline end to end
├── src/
│   ├── config.py                  # all hyperparameters and CLI flags — no hardcoded
│   │                               # environment-specific paths anywhere in this repo
│   ├── data_utils.py               # CSV loading, seeded stratified sampling,
│   │                               # preprocessing, feature engineering
│   ├── models.py                  # Autoencoder, Fusion Classifier, and all baseline
│   │                               # deep learning architectures
│   ├── eval_utils.py               # single source of truth for all evaluation metrics
│   ├── train_main.py               # trains the Autoencoder + Fusion Classifier + ablation arms
│   ├── baselines_classical.py      # LogisticRegression, RandomForest, IsolationForest,
│   │                               # OneClassSVM, XGBoost + feature importance
│   ├── baselines_deep.py           # CNN1D, CNN+BiLSTM+Attention, Transformer
│   ├── latency_bench.py            # honest single-sample latency + throughput scaling curve
│   ├── generalization_loao.py      # leave-one-attack-out generalization test
│   ├── multiclass_eval.py          # coarse-category multiclass extension
│   ├── ae_anomaly_detector.py      # benign-only unsupervised anomaly detector
│   ├── improve_hard_classes.py     # targeted fix for hard-to-detect attack categories
│   ├── multi_seed_eval.py          # multi-seed evaluation + paired significance testing
│   └── make_figures.py             # generates every figure + results_summary.md
├── data/
│   └── merged_csv/                # place your CICIoT2023 Merged*.csv files here
└── artifacts/
    └── <run_name>/                # everything from a single run: model checkpoints,
                                    # training curves, results_*.json, figures/,
                                    # results_summary.md, sampling_manifest.json
```

## Dataset

This project uses [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) (Neto et al., *Sensors*, 2023), provided by the Canadian Institute for Cybersecurity. The dataset itself is **not included in this repository** — download it from the official source and place the CSV files under `data/merged_csv/` (or point `--data_dir` at wherever you keep them).

The full dataset is 45,019,243 rows across 63 files. Training directly on this scale is impractical on commodity hardware, so this pipeline draws a **reproducible, seeded, stratified sample** (default 1,343,613 rows) via per-class reservoir sampling — every parameter used to build the sample, including exact row counts per category, is logged to `sampling_manifest.json` for full auditability.

## Setup

```bash
git clone <this-repo>
cd safenet_pipeline
pip install -r requirements.txt
```

Requires Python 3.10+. Tested on Windows and Linux, CPU-only (no GPU required).

## Quick Start

**Always run the smoke test first**, before committing hours of compute on real data:

```bash
python run_all.py --quick
```

This generates a small synthetic dataset matching the real schema and runs the entire 10-stage pipeline in under two minutes, so you can confirm your environment is set up correctly.

**Full run on real data:**

```bash
python run_all.py --data_dir /path/to/your/merged_csv --artifacts_dir ./artifacts
```

On the hardware used in the paper (2-core/4-thread, 2.5 GHz, 16 GB RAM, no GPU), a full run takes several hours, dominated by RandomForest/XGBoost fitting and the multi-seed evaluation stage (which repeats the core training roughly `n_seeds` times — see below).

Every stage can also be run independently, reusing a previous run's cached data sample and trained models via `--run_name`:

```bash
python src/train_main.py --data_dir ... --run_name my_run
python src/baselines_classical.py --data_dir ... --run_name my_run
python src/multi_seed_eval.py --data_dir ... --run_name my_run --n_seeds 3
python src/make_figures.py --data_dir ... --run_name my_run
```

## Pipeline Stages

| Stage | Script | What it does |
|---|---|---|
| 1 | `train_main.py` | Trains the Dense Autoencoder + Fusion Classifier, plus two ablation arms (raw-features-only, latent-only) |
| 2 | `baselines_classical.py` | LogisticRegression, RandomForest, IsolationForest, OneClassSVM, XGBoost — identical split/scaling; extracts feature importance from tree models |
| 3 | `baselines_deep.py` | CNN1D, CNN+BiLSTM+Attention, compact Transformer encoder |
| 4 | `latency_bench.py` | Genuine single-sample latency + throughput-vs-load scaling curve, using real test rows on CPU |
| 5 | `generalization_loao.py` | Leave-one-attack-out: retrains excluding each of several representative attack types, tests detection on the truly unseen category |
| 6 | `multiclass_eval.py` | Coarse-category multiclass extension for finer-grained analysis |
| 7 | `ae_anomaly_detector.py` | A second Autoencoder trained on benign traffic only (zero attack labels used anywhere), tested as a standalone anomaly detector |
| 8 | `improve_hard_classes.py` | Automatically identifies attack categories with low validation recall, adds one engineered feature, oversamples those categories, and reports a rigorous before/after comparison |
| 9 | `multi_seed_eval.py` | Repeats key comparisons across multiple independently resampled seeds with paired significance testing (t-test + Wilcoxon) |
| 10 | `make_figures.py` | Generates every publication-ready figure (300 DPI) and `results_summary.md`, the single source of truth for every number in the paper |

## Reproducibility

- No hardcoded, machine-specific file paths anywhere in this repository — everything is a CLI flag with a sensible relative default.
- All random seeds are fixed and logged; the stratified sampling procedure is fully deterministic given a seed.
- `results_summary.md`, generated by `make_figures.py`, is the single source of truth for every number that appears in the paper's text, tables, and figures — nothing is transcribed by hand, so the narrative, the tables, and the figures cannot contradict each other.
- Model checkpoints (`.pt` files), full training curves, and every intermediate result are saved, so results can be inspected or reused without retraining.

## Citation

If you use this code or build on this work, please cite:

```bibtex
@article{safenet2026,
  title   = {SAFE-Net: A Scalable Hybrid Deep Learning Model for Real-Time Intrusion Detection in IoT Networks},
  author  = {<author names>},
  year    = {2026},
  note    = {Code and reproducibility materials: <this repository URL>}
}
```

*(Update with the final venue/DOI once available.)*

Dataset citation:
```bibtex
@article{neto2023ciciot2023,
  title   = {CI-CIoT2023: A real-time dataset and benchmark for large-scale attacks in IoT environment},
  author  = {Neto, E. C. P. and Dadkhah, S. and Ferreira, R. and Zohourian, A. and Lu, R. and Ghorbani, A. A.},
  journal = {Sensors},
  year    = {2023}
}
```

## License

*(Add your chosen license — e.g., MIT, Apache 2.0 — here.)*

## Acknowledgments

Built on the CICIoT2023 dataset provided by the Canadian Institute for Cybersecurity, University of New Brunswick.
