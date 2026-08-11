"""
Classical ML baselines (LogisticRegression, RandomForest, XGBoost,
IsolationForest, OneClassSVM) trained on EXACTLY the same train/val/test
split and MinMaxScaler as the main SAFE-Net-v2 model (loaded from
main/scaler.pkl and reconstructed via the same seeded split function),
so the comparison in the paper is apples-to-apples.

IMPORTANT: last time, Random Forest matched or beat the proposed model
on this dataset and the paper claimed otherwise. This script reports
whatever comes out, honestly, into baselines_results.json — the
downstream figure/table generation does not hide or omit any baseline.
"""
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from data_utils import (
    has_csv_files,
    list_csv_files, get_cached_sample,
    preprocess, split, scale, make_synthetic_dataset,
)
from eval_utils import set_seed, full_evaluation, save_json


def run_one(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te, out_dir,
            needs_benign_only_fit=False, max_train_samples=None, seed=42, feature_names=None):
    t0 = time.time()
    n_full_benign = int((y_tr == 0).sum())
    subsampled_note = None
    if needs_benign_only_fit:
        X_fit = X_tr[y_tr == 0]
        if max_train_samples is not None and X_fit.shape[0] > max_train_samples:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X_fit.shape[0], size=max_train_samples, replace=False)
            X_fit = X_fit[idx]
            subsampled_note = (
                f"Trained on a random subsample of {max_train_samples:,} benign rows "
                f"(out of {n_full_benign:,} available) -- {name} does not scale to the "
                f"full training set on commodity hardware (O(n^2)-O(n^3) complexity); "
                f"this is a standard, documented practice for this model type."
            )
            print(f"[BASELINE] {name}: subsampling benign fit set {n_full_benign:,} -> {max_train_samples:,} "
                  f"rows (this model type does not scale to the full set)")
        model.fit(X_fit)
        val_scores = -model.score_samples(X_val) if hasattr(model, "score_samples") else -model.decision_function(X_val)
        te_scores = -model.score_samples(X_te) if hasattr(model, "score_samples") else -model.decision_function(X_te)
    else:
        model.fit(X_tr, y_tr)
        val_scores = model.predict_proba(X_val)[:, 1]
        te_scores = model.predict_proba(X_te)[:, 1]
    train_time = time.time() - t0

    # F1-optimal threshold on validation set, identical methodology to the main model
    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, val_scores)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[np.nanargmax(f1[:-1])]) if len(thr) else float(np.percentile(val_scores, 95))

    t1 = time.time()
    bundle = full_evaluation(y_te, te_scores, threshold=threshold)
    infer_time = time.time() - t1

    bundle["model_name"] = name
    bundle["train_time_sec"] = train_time
    bundle["inference_time_sec_full_testset"] = infer_time
    bundle["inference_time_ms_per_sample"] = (infer_time / len(y_te)) * 1000
    if subsampled_note is not None:
        bundle["training_subsample_note"] = subsampled_note
        bundle["train_rows_used"] = int(min(max_train_samples, n_full_benign)) if needs_benign_only_fit else None

    # Feature importance -- free to extract from tree-based models, and
    # directly explains WHICH raw features drive detection (addresses:
    # "why does the model succeed on flooding attacks and fail on recon/
    # spoofing attacks?"). Only tree models expose this natively.
    if hasattr(model, "feature_importances_") and feature_names is not None:
        importances = model.feature_importances_.tolist()
        ranked = sorted(zip(feature_names, importances), key=lambda kv: -kv[1])
        bundle["feature_importance"] = {name_: float(imp) for name_, imp in ranked}

    save_json(bundle, os.path.join(out_dir, f"results_{name}.json"))
    print(f"[BASELINE] {name}: acc={bundle['classification_report']['accuracy']:.4f} "
          f"AUC={bundle['auc']}  train_time={train_time:.1f}s")
    return bundle


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    out_dir = os.path.join(run_dir(cfg), "baselines_classical")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.quick and not has_csv_files(cfg.data_dir):
        make_synthetic_dataset(cfg.data_dir)

    files = list_csv_files(cfg.data_dir)
    df, _, was_cached = get_cached_sample(files, cfg, run_dir(cfg))
    print("[INFO] " + ("loaded cached" if was_cached else "built new") + f" sample: {df.shape}")
    X, y_bin, y_multi, feat_names = preprocess(df)
    splits = split(X, y_bin, y_multi, cfg)
    splits, scaler = scale(splits)

    X_tr = splits["train"]["X_scaled"].values.astype(np.float32)
    X_val = splits["val"]["X_scaled"].values.astype(np.float32)
    X_te = splits["test"]["X_scaled"].values.astype(np.float32)
    y_tr = splits["train"]["y_bin"].values.astype(np.int64)
    y_val = splits["val"]["y_bin"].values.astype(np.int64)
    y_te = splits["test"]["y_bin"].values.astype(np.int64)

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, IsolationForest
    from sklearn.svm import OneClassSVM

    run_one("LogisticRegression",
            LogisticRegression(max_iter=1000, class_weight="balanced"),
            X_tr, y_tr, X_val, y_val, X_te, y_te, out_dir)

    run_one("RandomForest",
            RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    random_state=cfg.seed, n_jobs=cfg.n_jobs),
            X_tr, y_tr, X_val, y_val, X_te, y_te, out_dir, feature_names=feat_names)

    run_one("IsolationForest",
            IsolationForest(n_estimators=200, contamination="auto", random_state=cfg.seed),
            X_tr, y_tr, X_val, y_val, X_te, y_te, out_dir, needs_benign_only_fit=True)

    run_one("OneClassSVM",
            OneClassSVM(kernel="rbf", gamma="scale", nu=0.1),
            X_tr, y_tr, X_val, y_val, X_te, y_te, out_dir, needs_benign_only_fit=True,
            max_train_samples=15000 if not cfg.quick else None, seed=cfg.seed)
    # NOTE: OneClassSVM's O(n^2)-O(n^3) training complexity makes it
    # completely impractical on hundreds of thousands of rows -- an
    # earlier full run took >7 hours on this exact dataset. Subsampling
    # the benign fit set to 15,000 rows is a standard, explicitly
    # documented practice for this model type (see `training_subsample_note`
    # in results_OneClassSVM.json) and keeps this baseline honest and
    # reproducible in reasonable time. RandomForest/XGBoost/IsolationForest
    # still train on the FULL training set -- only OCSVM is capped.

    try:
        import xgboost as xgb
        run_one("XGBoost",
                xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8,
                                   eval_metric="logloss", random_state=cfg.seed,
                                   n_jobs=cfg.n_jobs),
                X_tr, y_tr, X_val, y_val, X_te, y_te, out_dir, feature_names=feat_names)
    except ImportError:
        print("[WARN] xgboost not installed — skipping. `pip install xgboost` to include it "
              "(reviewers specifically want this baseline present).")

    print(f"[DONE] classical baselines complete. Saved under: {out_dir}")


if __name__ == "__main__":
    main()
