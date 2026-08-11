"""
Tests the Autoencoder for what autoencoders are actually good at:
flagging traffic that looks nothing like what it was trained on, with
ZERO attack labels used anywhere in training -- not even indirectly.

This is a genuinely different model from SAFE-Net's fused classifier:
it is trained ONLY on benign traffic, never sees a single attack
example of any kind, and detects attacks purely via reconstruction
error (traffic that doesn't look like normal benign traffic gets a
high reconstruction error and is flagged).

Because it never uses labels during training, it does not need
per-class retraining the way the supervised leave-one-attack-out test
in generalization_loao.py does -- ONE model, trained once, can be
evaluated directly against every attack category, including the exact
ones tested in the LOAO experiment, giving a direct, fair comparison:
does a purely unsupervised approach generalize to unseen/novel attacks
better than the supervised fusion classifier does?
"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from data_utils import (
    list_csv_files, get_cached_sample, has_csv_files,
    preprocess, split, scale, make_synthetic_dataset,
)
from models import DenseAutoencoder
from eval_utils import set_seed, get_device, full_evaluation, save_json


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = os.path.join(run_dir(cfg), "ae_anomaly_detector")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.quick and not has_csv_files(cfg.data_dir):
        make_synthetic_dataset(cfg.data_dir)

    files = list_csv_files(cfg.data_dir)
    df, _, was_cached = get_cached_sample(files, cfg, run_dir(cfg))
    print("[INFO] " + ("loaded cached" if was_cached else "built new") + f" sample: {df.shape}")
    X, y_bin, y_multi, feat_names = preprocess(df)
    splits = split(X, y_bin, y_multi, cfg)
    splits, scaler = scale(splits)
    input_dim = splits["train"]["X_scaled"].shape[1]

    X_tr_all = splits["train"]["X_scaled"].values.astype(np.float32)
    y_tr_bin = splits["train"]["y_bin"].values
    X_tr_benign = X_tr_all[y_tr_bin == 0]  # ONLY benign rows -- no attack examples at all
    print(f"[AE-ANOMALY] training on {X_tr_benign.shape[0]:,} benign-only rows "
          f"(0 attack examples used, unlike the supervised model)")

    X_val = splits["val"]["X_scaled"].values.astype(np.float32)
    X_te = splits["test"]["X_scaled"].values.astype(np.float32)
    y_val = splits["val"]["y_bin"].values.astype(np.int64)
    y_te = splits["test"]["y_bin"].values.astype(np.int64)

    ae = DenseAutoencoder(input_dim, cfg.hidden, cfg.latent, cfg.dropout).to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    crit = nn.MSELoss(reduction="none")

    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr_benign)), batch_size=cfg.batch_size, shuffle=True)
    val_t = torch.from_numpy(X_val).to(device)

    best_val, best_state, no_improve = np.inf, None, 0
    curve = {"train": [], "val": []}
    for ep in range(cfg.epochs_ae):
        ae.train()
        tr_loss = 0.0
        for (b,) in loader:
            b = b.to(device)
            opt.zero_grad()
            recon, _ = ae(b)
            loss = crit(recon, b).mean()
            loss.backward()
            opt.step()
            tr_loss += loss.item() * b.size(0)
        tr_loss /= len(loader.dataset)

        ae.eval()
        with torch.no_grad():
            recon, _ = ae(val_t)
            val_loss = crit(recon, val_t).mean().item()
        sched.step(val_loss)
        curve["train"].append(tr_loss)
        curve["val"].append(val_loss)
        print(f"[AE-ANOMALY] epoch {ep+1}/{cfg.epochs_ae} train={tr_loss:.6f} val={val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg.patience_ae:
                print(f"[AE-ANOMALY] early stop at epoch {ep+1}")
                break
    ae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(ae.state_dict(), os.path.join(out_dir, "best_ae_anomaly.pt"))
    save_json(curve, os.path.join(out_dir, "ae_anomaly_training_curve.json"))

    def recon_error(X):
        ae.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(X), 2048):
                xb = torch.from_numpy(X[i:i+2048]).to(device)
                recon, _ = ae(xb)
                e = ((recon - xb) ** 2).mean(dim=1).cpu().numpy()
                errs.append(e)
        return np.concatenate(errs)

    val_err = recon_error(X_val)
    te_err = recon_error(X_te)

    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, val_err)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[np.nanargmax(f1[:-1])]) if len(thr) else float(np.percentile(val_err, 95))

    results = full_evaluation(y_te, te_err, threshold=threshold)
    results["note"] = "Trained on benign traffic only; anomaly score = reconstruction error; no attack labels used in training."
    save_json(results, os.path.join(out_dir, "results_ae_anomaly.json"))
    print(f"[AE-ANOMALY] overall test acc={results['classification_report']['accuracy']:.4f} AUC={results['auc']}")

    # ---- Per-attack-type recall, and direct comparison against the supervised LOAO results ----
    te_pred = (te_err >= threshold).astype(int)
    te_multi = splits["test"]["y_multi"].values
    per_type = {}
    for label in sorted(set(te_multi)):
        mask = te_multi == label
        n = int(mask.sum())
        if n == 0:
            continue
        recall = float((te_pred[mask] == (0 if label == "BENIGN" else 1)).sum() / n)
        per_type[label] = {"support": n, "recall": recall}
    save_json(per_type, os.path.join(out_dir, "ae_anomaly_per_attack_recall.json"))

    loao_path = os.path.join(run_dir(cfg), "generalization_loao", "loao_results.json")
    comparison = {}
    if os.path.exists(loao_path):
        with open(loao_path) as f:
            loao = json.load(f)
        for attack_type, loao_r in loao.items():
            ae_r = per_type.get(attack_type)
            comparison[attack_type] = {
                "supervised_LOAO_recall_on_unseen": loao_r["recall_detected_as_attack_despite_never_trained_on_this_class"],
                "unsupervised_AE_anomaly_recall": ae_r["recall"] if ae_r else None,
                "note": "AE never used ANY attack labels (including this class) during training -- "
                        "its 'generalization' to this class is by construction, not by exclusion.",
            }
        save_json(comparison, os.path.join(out_dir, "ae_vs_supervised_loao_comparison.json"))
        print("\n[AE-ANOMALY] Unsupervised AE vs. Supervised LOAO recall on unseen attack types:")
        for k, v in comparison.items():
            print(f"  {k}: supervised={v['supervised_LOAO_recall_on_unseen']:.3f}  "
                  f"unsupervised_AE={v['unsupervised_AE_anomaly_recall']}")
    else:
        print("[AE-ANOMALY] no generalization_loao.py results found yet -- run that stage too "
              "for the full unsupervised-vs-supervised comparison.")

    print(f"\n[DONE] AE anomaly detector complete. Saved under: {out_dir}")


if __name__ == "__main__":
    main()
