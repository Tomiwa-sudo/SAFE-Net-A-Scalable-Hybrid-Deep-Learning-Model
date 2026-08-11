"""
Runs the ENTIRE pipeline end to end, in the correct order, with one command.

    python run_all.py --data_dir "C:\\path\\to\\MERGED_CSV" --artifacts_dir ./artifacts
    python run_all.py --quick                     # fast synthetic smoke test first (recommended)

Everything from every stage is saved under artifacts/<run_name>/ — model
checkpoints, every training curve, every results_*.json, every figure,
and results_summary.md. Once this finishes, you never need to re-run
anything to get numbers/figures/tables for the paper — just open
results_summary.md and the figures/ folder.
"""
import subprocess
import sys
import os
import time

STAGES = [
    ("src/train_main.py",            "Main model: Dense AE + Fusion Classifier + Ablation"),
    ("src/baselines_classical.py",   "Classical ML baselines (LR, RF, XGBoost, IsolationForest, OCSVM) + feature importance"),
    ("src/baselines_deep.py",        "Deep learning baselines (CNN1D, CNN-BiLSTM-Attention, Transformer)"),
    ("src/latency_bench.py",         "Honest latency benchmark + throughput scaling curve"),
    ("src/generalization_loao.py",   "Leave-one-attack-out generalization test"),
    ("src/multiclass_eval.py",       "Multiclass (coarse-category) evaluation"),
    ("src/ae_anomaly_detector.py",   "Autoencoder as standalone zero-day detector vs. supervised LOAO"),
    ("src/improve_hard_classes.py",  "Targeted fix for hard-to-detect attack categories"),
    ("src/multi_seed_eval.py",       "Multi-seed evaluation + statistical significance testing (SLOW — repeats training n_seeds times)"),
    ("src/make_figures.py",          "Generate all figures + results_summary.md"),
]


def main():
    extra_args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))

    # Fix run_name across all stages so they all write into the same folder
    run_name = None
    for i, a in enumerate(extra_args):
        if a == "--run_name" and i + 1 < len(extra_args):
            run_name = extra_args[i + 1]
    if run_name is None:
        import datetime
        run_name = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        extra_args += ["--run_name", run_name]

    print(f"=== SAFE-Net v2 full pipeline — run_name = {run_name} ===\n")

    t0 = time.time()
    for script, desc in STAGES:
        print(f"\n{'='*70}\n[STAGE] {desc}\n[SCRIPT] {script}\n{'='*70}")
        cmd = [sys.executable, os.path.join(here, script)] + extra_args
        result = subprocess.run(cmd, cwd=here)
        if result.returncode != 0:
            print(f"\n[FATAL] Stage failed: {script} (exit code {result.returncode})")
            print("Fix the error above, then re-run this stage alone (or run_all.py again "
                  "with the same --run_name to resume; completed stages' artifacts are "
                  "untouched but WILL be overwritten if you re-run them).")
            sys.exit(1)

    print(f"\n\n{'='*70}")
    print(f"ALL STAGES COMPLETE in {time.time()-t0:.1f}s")
    print(f"Results: artifacts/{run_name}/results_summary.md")
    print(f"Figures: artifacts/{run_name}/figures/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
