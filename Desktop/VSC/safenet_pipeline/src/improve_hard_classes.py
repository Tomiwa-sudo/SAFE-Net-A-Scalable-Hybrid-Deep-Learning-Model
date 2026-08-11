"""
Turns the diagnostic finding ("the model is bad at detecting
reconnaissance/spoofing/injection attacks") into an actual fix, and
measures whether it worked.

Approach:
  1. Load the already-trained base fused model (from train_main.py).
  2. Compute per-attack-type recall on the VALIDATION set (not test --
     test stays untouched until final comparison, to avoid leakage).
  3. Any attack category below cfg.hard_class_recall_threshold is
     flagged "hard" -- this is fully data-driven, not a hardcoded list
     of label strings, so it works on any sample/run, not just this one.
  4. Add the engineered `flow_signature_frequency` feature (see
     data_utils.add_engineered_features).
  5. Retrain a new "improved" AE + fused classifier using per-sample
     oversampling weights that boost hard-class rows.
  6. Evaluate BOTH the original and improved model on the untouched
     TEST set, per overall metrics AND per-attack-type recall on the
     hard classes specifically, and save a direct before/after
     comparison.
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from data_utils import (
    list_csv_files, get_cached_sample, has_csv_files,
    preprocess, split, scale, make_synthetic_dataset, add_engineered_features,
)
from models import DenseAutoencoder, FusionClassifier, count_params
from eval_utils import set_seed, get_device, full_evaluation, save_json
from train_main import train_autoencoder, train_classifier, calibrate_and_eval


def per_attack_recall(y_multi, y_pred, y_true_bin):
    out = {}
    for label in sorted(set(y_multi)):
        mask = y_multi == label
        n = int(mask.sum())
        if n == 0:
            continue
        if label == "BENIGN":
            recall = float((y_pred[mask] == 0).sum() / n)
        else:
            recall = float((y_pred[mask] == 1).sum() / n)
        out[label] = {"support": n, "recall": recall}
    return out


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    main_dir = os.path.join(run_dir(cfg), "main")
    out_dir = os.path.join(run_dir(cfg), "hard_class_improvement")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.quick and not has_csv_files(cfg.data_dir):
        make_synthetic_dataset(cfg.data_dir)

    files = list_csv_files(cfg.data_dir)
    df, _, was_cached = get_cached_sample(files, cfg, run_dir(cfg))
    print("[INFO] " + ("loaded cached" if was_cached else "built new") + f" sample: {df.shape}")
    X, y_bin, y_multi, feat_names = preprocess(df)
    splits = split(X, y_bin, y_multi, cfg)
    splits, scaler = scale(splits)
    input_dim_base = splits["train"]["X_scaled"].shape[1]

    # ---- Step 1-2: load base model, compute per-attack recall on VALIDATION ----
    ae_base = DenseAutoencoder(input_dim_base, cfg.hidden, cfg.latent, cfg.dropout).to(device)
    clf_base = FusionClassifier(cfg.latent + input_dim_base, 2, cfg.dropout).to(device)
    ae_path, clf_path = os.path.join(main_dir, "best_ae.pt"), os.path.join(main_dir, "best_clf_fused.pt")
    if not (os.path.exists(ae_path) and os.path.exists(clf_path)):
        print("[ERROR] base model not found -- run train_main.py first for this --run_name")
        sys.exit(1)
    ae_base.load_state_dict(torch.load(ae_path, map_location=device))
    clf_base.load_state_dict(torch.load(clf_path, map_location=device))
    ae_base.eval(); clf_base.eval()

    # Use the SAME calibrated threshold the base model was actually
    # evaluated with in train_main.py (F1-optimal on validation), instead
    # of a hardcoded 0.5 -- using a different threshold here than the one
    # in the official results made the "before" numbers incomparable to
    # both the official per_attack_type_recall.json AND to the "after"
    # model's own freshly-calibrated threshold.
    base_results_path = os.path.join(main_dir, "results_fused.json")
    if os.path.exists(base_results_path):
        with open(base_results_path) as f:
            base_threshold = json.load(f)["threshold"]
        print(f"[HARD-CLASS] using base model's actual calibrated threshold: {base_threshold:.4f}")
    else:
        base_threshold = 0.5
        print("[WARN] results_fused.json not found; falling back to 0.5 threshold for base model")

    X_val_base = splits["val"]["X_scaled"].values.astype(np.float32)
    with torch.no_grad():
        z_val = ae_base.encoder(torch.from_numpy(X_val_base).to(device)).cpu().numpy()
        fused_val = np.hstack([z_val, X_val_base]).astype(np.float32)
        val_probs = torch.softmax(clf_base(torch.from_numpy(fused_val).to(device)), dim=1)[:, 1].cpu().numpy()
    val_pred = (val_probs >= base_threshold).astype(int)
    val_multi = splits["val"]["y_multi"].values
    val_recall = per_attack_recall(val_multi, val_pred, splits["val"]["y_bin"].values)

    hard_classes = sorted([lbl for lbl, r in val_recall.items()
                            if lbl != "BENIGN" and r["recall"] < cfg.hard_class_recall_threshold])
    save_json({"hard_class_recall_threshold": cfg.hard_class_recall_threshold,
               "validation_recall_by_class": val_recall,
               "hard_classes_detected": hard_classes},
              os.path.join(out_dir, "hard_class_detection.json"))
    print(f"[HARD-CLASS] {len(hard_classes)} hard classes detected on validation set "
          f"(recall < {cfg.hard_class_recall_threshold}): {hard_classes}")

    if not hard_classes:
        print("[HARD-CLASS] no hard classes found at this threshold -- nothing to improve. Exiting.")
        return

    # ---- Step 3-4: engineered feature + oversampling weights ----
    splits = add_engineered_features(splits)
    X_tr = splits["train"]["X_scaled"].values.astype(np.float32)
    X_val = splits["val"]["X_scaled"].values.astype(np.float32)
    X_te = splits["test"]["X_scaled"].values.astype(np.float32)
    y_tr = splits["train"]["y_bin"].values.astype(np.int64)
    y_val = splits["val"]["y_bin"].values.astype(np.int64)
    y_te = splits["test"]["y_bin"].values.astype(np.int64)
    input_dim = X_tr.shape[1]

    from sklearn.utils.class_weight import compute_class_weight
    base_cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr)
    sample_w = np.array([base_cw[label] for label in y_tr], dtype=np.float64)
    tr_multi = splits["train"]["y_multi"].values
    hard_mask = np.isin(tr_multi, hard_classes)
    sample_w[hard_mask] *= cfg.hard_class_oversample_weight
    print(f"[HARD-CLASS] boosting {hard_mask.sum():,} training rows ({hard_mask.mean()*100:.2f}%) "
          f"belonging to hard classes by {cfg.hard_class_oversample_weight}x")

    # ---- Step 5: retrain AE + fused classifier with new feature + weighted sampling ----
    ae_imp = DenseAutoencoder(input_dim, cfg.hidden, cfg.latent, cfg.dropout).to(device)
    ae_imp = train_autoencoder(ae_imp, X_tr, X_val, cfg, device, out_dir)
    ae_imp.eval()
    with torch.no_grad():
        z_tr = ae_imp.encoder(torch.from_numpy(X_tr).to(device)).cpu().numpy()
        z_val2 = ae_imp.encoder(torch.from_numpy(X_val).to(device)).cpu().numpy()
        z_te = ae_imp.encoder(torch.from_numpy(X_te).to(device)).cpu().numpy()

    fused_tr = np.hstack([z_tr, X_tr]).astype(np.float32)
    fused_val = np.hstack([z_val2, X_val]).astype(np.float32)
    fused_te = np.hstack([z_te, X_te]).astype(np.float32)

    clf_imp = FusionClassifier(fused_tr.shape[1], 2, cfg.dropout).to(device)
    clf_imp = train_classifier(clf_imp, fused_tr, y_tr, fused_val, y_val, cfg, device, out_dir,
                                "improved", class_weight=None, sample_weights=sample_w)
    results_imp, prob_imp = calibrate_and_eval(clf_imp, fused_val, y_val, fused_te, y_te, device)
    save_json(results_imp, os.path.join(out_dir, "results_improved.json"))

    # ---- Step 6: before/after comparison on the untouched TEST set ----
    # Evaluate the ORIGINAL base model on the test set (without the new
    # engineered feature, exactly as it was trained) for a fair before/after.
    X_te_base = splits["test"]["X_scaled"].drop(columns=["flow_signature_frequency"]).values.astype(np.float32)
    with torch.no_grad():
        z_te_base = ae_base.encoder(torch.from_numpy(X_te_base).to(device)).cpu().numpy()
        fused_te_base = np.hstack([z_te_base, X_te_base]).astype(np.float32)
        base_probs_te = torch.softmax(clf_base(torch.from_numpy(fused_te_base).to(device)), dim=1)[:, 1].cpu().numpy()
    base_pred_te = (base_probs_te >= base_threshold).astype(int)

    imp_pred_te = (prob_imp >= results_imp["threshold"]).astype(int)
    te_multi = splits["test"]["y_multi"].values

    base_recall_te = per_attack_recall(te_multi, base_pred_te, y_te)
    imp_recall_te = per_attack_recall(te_multi, imp_pred_te, y_te)

    comparison = {"hard_classes": hard_classes, "per_class_before_vs_after": {}}
    for cls in hard_classes:
        before = base_recall_te.get(cls, {"support": 0, "recall": None})
        after = imp_recall_te.get(cls, {"support": 0, "recall": None})
        comparison["per_class_before_vs_after"][cls] = {
            "support": before["support"],
            "recall_before": before["recall"],
            "recall_after": after["recall"],
            "improvement": (after["recall"] - before["recall"]) if (before["recall"] is not None and after["recall"] is not None) else None,
        }
    comparison["overall_before"] = {
        "accuracy": float((base_pred_te == y_te).mean()),
    }
    comparison["overall_after"] = {
        "accuracy": results_imp["classification_report"]["accuracy"],
        "auc": results_imp["auc"],
    }
    save_json(comparison, os.path.join(out_dir, "before_after_comparison.json"))

    # Free the base model + its intermediate arrays explicitly -- this
    # script keeps two full models (base + improved) in memory at once,
    # which is close to the ceiling on a 16GB machine already running
    # other processes. Not strictly required (Python will eventually
    # collect these), but cheap insurance against the transient
    # MemoryError seen on the first attempt at this stage.
    del ae_base, clf_base, z_te_base, fused_te_base, base_probs_te
    import gc
    gc.collect()

    print("\n[HARD-CLASS] Before -> After (test set recall on hard classes):")
    for cls, v in comparison["per_class_before_vs_after"].items():
        b = v["recall_before"] if v["recall_before"] is not None else float("nan")
        a = v["recall_after"] if v["recall_after"] is not None else float("nan")
        print(f"  {cls}: {b:.3f} -> {a:.3f}")
    print(f"\n[DONE] hard-class improvement complete. Saved under: {out_dir}")


if __name__ == "__main__":
    main()
