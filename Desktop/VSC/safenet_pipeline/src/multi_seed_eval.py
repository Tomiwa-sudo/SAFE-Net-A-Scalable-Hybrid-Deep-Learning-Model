"""
Runs SAFE-Net (fused + both ablation arms) and the two strongest baselines
(RandomForest, XGBoost) across multiple random seeds, each with an
INDEPENDENTLY resampled dataset (not just a different weight
initialization), and reports mean +/- std instead of single-run numbers.

Also runs paired statistical significance tests (paired t-test AND the
non-parametric Wilcoxon signed-rank test, since n_seeds is typically
small) comparing SAFE-Net-fused against each of: RandomForest, XGBoost,
raw-features-only, AE-latent-only. This directly answers "is the gap
to XGBoost/RandomForest real, or is it noise from a single run?"

Runtime warning: this repeats the most expensive parts of the pipeline
`cfg.n_seeds` times (default 3). Expect roughly 3x the combined runtime
of train_main.py's fused+ablation training plus RandomForest+XGBoost
fitting. Reduce with --n_seeds 2 if needed.
"""
import copy
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from data_utils import (
    list_csv_files, get_cached_sample, has_csv_files,
    preprocess, split, scale, make_synthetic_dataset,
)
from models import DenseAutoencoder, FusionClassifier, PlainMLPBaseline, count_params
from eval_utils import set_seed, get_device, full_evaluation, save_json
from train_main import train_autoencoder, train_classifier, calibrate_and_eval


def run_one_seed(seed, base_cfg, device, out_dir):
    cfg = copy.deepcopy(base_cfg)
    cfg.seed = seed
    set_seed(seed)

    files = list_csv_files(cfg.data_dir)
    df, _, was_cached = get_cached_sample(files, cfg, run_dir(base_cfg))
    print(f"[SEED {seed}] sample: {df.shape} ({'cached' if was_cached else 'built new'})")

    X, y_bin, y_multi, feat_names = preprocess(df)
    splits = split(X, y_bin, y_multi, cfg)
    splits, scaler = scale(splits)

    X_tr = splits["train"]["X_scaled"].values.astype(np.float32)
    X_val = splits["val"]["X_scaled"].values.astype(np.float32)
    X_te = splits["test"]["X_scaled"].values.astype(np.float32)
    y_tr = splits["train"]["y_bin"].values.astype(np.int64)
    y_val = splits["val"]["y_bin"].values.astype(np.int64)
    y_te = splits["test"]["y_bin"].values.astype(np.int64)
    input_dim = X_tr.shape[1]

    seed_dir = os.path.join(out_dir, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)
    results = {}

    # ---- Autoencoder + fused / ablation arms ----
    ae = DenseAutoencoder(input_dim, cfg.hidden, cfg.latent, cfg.dropout).to(device)
    ae = train_autoencoder(ae, X_tr, X_val, cfg, device, seed_dir)
    ae.eval()
    with torch.no_grad():
        z_tr = ae.encoder(torch.from_numpy(X_tr).to(device)).cpu().numpy()
        z_val = ae.encoder(torch.from_numpy(X_val).to(device)).cpu().numpy()
        z_te = ae.encoder(torch.from_numpy(X_te).to(device)).cpu().numpy()

    from sklearn.utils.class_weight import compute_class_weight
    cw = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr), dtype=torch.float32)

    fused_tr = np.hstack([z_tr, X_tr]).astype(np.float32)
    fused_val = np.hstack([z_val, X_val]).astype(np.float32)
    fused_te = np.hstack([z_te, X_te]).astype(np.float32)
    clf_fused = FusionClassifier(fused_tr.shape[1], 2, cfg.dropout).to(device)
    clf_fused = train_classifier(clf_fused, fused_tr, y_tr, fused_val, y_val, cfg, device, seed_dir, "fused", cw)
    r_fused, _ = calibrate_and_eval(clf_fused, fused_val, y_val, fused_te, y_te, device)
    results["SAFE-Net_fused"] = r_fused

    clf_raw = PlainMLPBaseline(X_tr.shape[1], 2, cfg.dropout).to(device)
    clf_raw = train_classifier(clf_raw, X_tr, y_tr, X_val, y_val, cfg, device, seed_dir, "rawonly", cw)
    r_raw, _ = calibrate_and_eval(clf_raw, X_val, y_val, X_te, y_te, device)
    results["raw_only"] = r_raw

    clf_z = FusionClassifier(z_tr.shape[1], 2, cfg.dropout).to(device)
    clf_z = train_classifier(clf_z, z_tr, y_tr, z_val, y_val, cfg, device, seed_dir, "aeonly", cw)
    r_z, _ = calibrate_and_eval(clf_z, z_val, y_val, z_te, y_te, device)
    results["ae_only"] = r_z

    # ---- RandomForest & XGBoost (the two baselines that beat SAFE-Net in the single-run result) ----
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import precision_recall_curve

    def fit_eval_sklearn(model, name):
        model.fit(X_tr, y_tr)
        val_scores = model.predict_proba(X_val)[:, 1]
        te_scores = model.predict_proba(X_te)[:, 1]
        prec, rec, thr = precision_recall_curve(y_val, val_scores)
        f1 = 2 * prec * rec / (prec + rec + 1e-12)
        threshold = float(thr[np.nanargmax(f1[:-1])]) if len(thr) else float(np.percentile(val_scores, 95))
        return full_evaluation(y_te, te_scores, threshold=threshold)

    results["RandomForest"] = fit_eval_sklearn(
        RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=seed, n_jobs=cfg.n_jobs),
        "RandomForest"
    )
    try:
        import xgboost as xgb
        results["XGBoost"] = fit_eval_sklearn(
            xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8,
                               colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=cfg.n_jobs),
            "XGBoost"
        )
    except ImportError:
        print("[WARN] xgboost not installed, skipping for this seed")

    return results


def paired_significance(values_a, values_b, name_a, name_b, metric):
    from scipy import stats
    a, b = np.array(values_a), np.array(values_b)
    out = {"metric": metric, "model_a": name_a, "model_b": name_b,
           "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(a.mean() - b.mean())}
    if len(a) < 2:
        out["note"] = "fewer than 2 seeds; significance test skipped"
        return out
    try:
        t_stat, t_p = stats.ttest_rel(a, b)
        out["paired_ttest_p"] = float(t_p)
    except Exception as e:
        out["paired_ttest_p"] = None
    try:
        if np.all(a == b):
            out["wilcoxon_p"] = 1.0
        else:
            w_stat, w_p = stats.wilcoxon(a, b)
            out["wilcoxon_p"] = float(w_p)
    except Exception:
        out["wilcoxon_p"] = None
    out["interpretation"] = (
        "no statistically significant difference (p >= 0.05)"
        if (out.get("paired_ttest_p") or 1.0) >= 0.05
        else "statistically significant difference (p < 0.05)"
    )
    return out


def main():
    cfg = build_config_from_cli()
    device = get_device(cfg.device)
    out_dir = os.path.join(run_dir(cfg), "multi_seed")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.quick and not has_csv_files(cfg.data_dir):
        make_synthetic_dataset(cfg.data_dir)

    print(f"[MULTI-SEED] running seeds: {cfg.seed_list}")
    t0 = time.time()
    per_seed_results = {}
    for seed in cfg.seed_list:
        per_seed_results[seed] = run_one_seed(seed, cfg, device, out_dir)
    print(f"[MULTI-SEED] all seeds complete in {time.time()-t0:.1f}s")

    # ---- Aggregate mean/std per model per metric ----
    model_names = list(next(iter(per_seed_results.values())).keys())
    aggregate = {}
    for model in model_names:
        accs = [per_seed_results[s][model]["classification_report"]["accuracy"] for s in cfg.seed_list if model in per_seed_results[s]]
        aucs = [per_seed_results[s][model]["auc"] for s in cfg.seed_list if model in per_seed_results[s]]
        f1s = [per_seed_results[s][model]["classification_report"]["1"]["f1-score"] for s in cfg.seed_list if model in per_seed_results[s]]
        aggregate[model] = {
            "accuracy_mean": float(np.mean(accs)), "accuracy_std": float(np.std(accs)),
            "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
            "attack_f1_mean": float(np.mean(f1s)), "attack_f1_std": float(np.std(f1s)),
            "n_seeds": len(accs), "per_seed_accuracy": accs, "per_seed_auc": aucs, "per_seed_attack_f1": f1s,
        }
    save_json(aggregate, os.path.join(out_dir, "multi_seed_aggregate.json"))
    print("[MULTI-SEED] aggregate:", {k: (v["accuracy_mean"], v["accuracy_std"]) for k, v in aggregate.items()})

    # ---- Significance tests: SAFE-Net-fused vs each comparator ----
    sig_tests = []
    fused_acc = aggregate["SAFE-Net_fused"]["per_seed_accuracy"]
    fused_auc = aggregate["SAFE-Net_fused"]["per_seed_auc"]
    for comparator in ["RandomForest", "XGBoost", "raw_only", "ae_only"]:
        if comparator not in aggregate:
            continue
        sig_tests.append(paired_significance(fused_acc, aggregate[comparator]["per_seed_accuracy"],
                                              "SAFE-Net_fused", comparator, "accuracy"))
        sig_tests.append(paired_significance(fused_auc, aggregate[comparator]["per_seed_auc"],
                                              "SAFE-Net_fused", comparator, "auc"))
    save_json(sig_tests, os.path.join(out_dir, "significance_tests.json"))
    for t in sig_tests:
        print(f"[SIG TEST] {t['model_a']} vs {t['model_b']} ({t['metric']}): "
              f"diff={t['mean_diff']:.4f}, p={t.get('paired_ttest_p')}, {t['interpretation']}")

    print(f"[DONE] multi-seed evaluation complete. Saved under: {out_dir}")


if __name__ == "__main__":
    main()
