"""

"""
import json
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from data_utils import (
    list_csv_files, get_cached_sample,
    preprocess, split, scale, make_synthetic_dataset, has_csv_files,
)
from models import DenseAutoencoder, FusionClassifier, PlainMLPBaseline, count_params
from eval_utils import set_seed, get_device, full_evaluation, save_json


def train_autoencoder(ae, X_tr, X_val, cfg, device, out_dir):
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    crit = nn.MSELoss()

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr)), batch_size=cfg.batch_size, shuffle=True)
    val_t = torch.from_numpy(X_val).to(device)

    best_val, best_state, no_improve = np.inf, None, 0
    curve = {"train": [], "val": []}

    for ep in range(cfg.epochs_ae):
        ae.train()
        tr_loss = 0.0
        for (batch,) in tr_loader:
            batch = batch.to(device)
            opt.zero_grad()
            recon, _ = ae(batch)
            loss = crit(recon, batch)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * batch.size(0)
        tr_loss /= len(tr_loader.dataset)

        ae.eval()
        with torch.no_grad():
            recon, _ = ae(val_t)
            val_loss = crit(recon, val_t).item()
        sched.step(val_loss)
        curve["train"].append(tr_loss)
        curve["val"].append(val_loss)
        print(f"[AE] epoch {ep+1}/{cfg.epochs_ae}  train={tr_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in ae.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience_ae:
                print(f"[AE] early stop at epoch {ep+1}")
                break

    ae.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(ae.state_dict(), os.path.join(out_dir, "best_ae.pt"))
    save_json(curve, os.path.join(out_dir, "ae_training_curve.json"))
    return ae


def train_classifier(clf, feat_tr, y_tr, feat_val, y_val, cfg, device, out_dir, tag,
                      class_weight=None, sample_weights=None):
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.lr_clf)
    crit = nn.CrossEntropyLoss(weight=class_weight.to(device) if class_weight is not None else None)

    dataset = TensorDataset(torch.from_numpy(feat_tr), torch.from_numpy(y_tr))
    if sample_weights is not None:
        # per-sample oversampling weights (e.g. boosting hard-to-detect
        # attack categories) instead of uniform shuffling.
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=len(sample_weights), replacement=True,
        )
        tr_loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=sampler)
    else:
        tr_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True)
    val_x = torch.from_numpy(feat_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)

    best_val, best_state, no_improve = np.inf, None, 0
    curve = {"train": [], "val": []}

    for ep in range(cfg.epochs_clf):
        clf.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = clf(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        clf.eval()
        with torch.no_grad():
            val_out = clf(val_x)
            val_loss = crit(val_out, val_y).item()
        curve["train"].append(tr_loss)
        curve["val"].append(val_loss)
        print(f"[CLF-{tag}] epoch {ep+1}/{cfg.epochs_clf}  train={tr_loss:.6f}  val={val_loss:.6f}")

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= cfg.patience_clf:
                print(f"[CLF-{tag}] early stop at epoch {ep+1}")
                break

    clf.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(clf.state_dict(), os.path.join(out_dir, f"best_clf_{tag}.pt"))
    save_json(curve, os.path.join(out_dir, f"clf_training_curve_{tag}.json"))
    return clf


def calibrate_and_eval(clf, feat_val, y_val, feat_te, y_te, device, mode="f1"):
    clf.eval()
    with torch.no_grad():
        val_prob = torch.softmax(clf(torch.from_numpy(feat_val).to(device)), dim=1)[:, 1].cpu().numpy()
        te_prob = torch.softmax(clf(torch.from_numpy(feat_te).to(device)), dim=1)[:, 1].cpu().numpy()

    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, val_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[np.nanargmax(f1[:-1])]) if len(thr) else float(np.percentile(val_prob, 95))
    # NOTE: threshold method is F1-optimal on the VALIDATION set, computed
    # here explicitly and logged -- this replaces the undocumented/
    # inconsistent percentile-vs-F1 mismatch between the two original
    # pipeline scripts. This exact value is saved into results.json below,
    # so the paper's Methodology section can cite it precisely.

    bundle = full_evaluation(y_te, te_prob, threshold=threshold)
    bundle["calibration_method"] = f"F1-optimal on validation set (mode={mode})"
    return bundle, te_prob


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = os.path.join(run_dir(cfg), "main")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] device={device}  run_dir={run_dir(cfg)}")

    if cfg.quick and not has_csv_files(cfg.data_dir):
        print("[INFO] --quick: generating tiny synthetic dataset for smoke test")
        make_synthetic_dataset(cfg.data_dir)

    files = list_csv_files(cfg.data_dir)
    print(f"[INFO] found {len(files)} csv files")

    t0 = time.time()
    df, sample_manifest, was_cached = get_cached_sample(files, cfg, run_dir(cfg))
    if was_cached:
        print(f"[INFO] loaded CACHED sample from a previous stage in this run: {df.shape[0]} rows "
              f"({time.time()-t0:.1f}s)")
    else:
        print(f"[INFO] built new sample (cached for later stages): {df.shape[0]} rows, "
              f"{df.shape[1]} cols ({time.time()-t0:.1f}s)")

    X, y_bin, y_multi, feat_names = preprocess(df)
    save_json(feat_names, os.path.join(out_dir, "feature_names.json"))

    splits = split(X, y_bin, y_multi, cfg)
    splits, scaler = scale(splits)
    with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    X_tr = splits["train"]["X_scaled"].values.astype(np.float32)
    X_val = splits["val"]["X_scaled"].values.astype(np.float32)
    X_te = splits["test"]["X_scaled"].values.astype(np.float32)
    y_tr = splits["train"]["y_bin"].values.astype(np.int64)
    y_val = splits["val"]["y_bin"].values.astype(np.int64)
    y_te = splits["test"]["y_bin"].values.astype(np.int64)

    class_balance = {
        "train": {"benign": int((y_tr == 0).sum()), "attack": int((y_tr == 1).sum())},
        "val": {"benign": int((y_val == 0).sum()), "attack": int((y_val == 1).sum())},
        "test": {"benign": int((y_te == 0).sum()), "attack": int((y_te == 1).sum())},
    }
    save_json(class_balance, os.path.join(out_dir, "class_balance.json"))
    print("[INFO] class balance:", class_balance)

    input_dim = X_tr.shape[1]

    # ---------------- Autoencoder ----------------
    ae = DenseAutoencoder(input_dim, cfg.hidden, cfg.latent, cfg.dropout).to(device)
    print(f"[INFO] AE params: {count_params(ae)}")
    ae = train_autoencoder(ae, X_tr, X_val, cfg, device, out_dir)

    ae.eval()
    with torch.no_grad():
        z_tr = ae.encoder(torch.from_numpy(X_tr).to(device)).cpu().numpy()
        z_val = ae.encoder(torch.from_numpy(X_val).to(device)).cpu().numpy()
        z_te = ae.encoder(torch.from_numpy(X_te).to(device)).cpu().numpy()

    from sklearn.utils.class_weight import compute_class_weight
    cw = torch.tensor(
        compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr), dtype=torch.float32
    )

    # ---------------- Fused model (main SAFE-Net v2) ----------------
    fused_tr = np.hstack([z_tr, X_tr]).astype(np.float32)
    fused_val = np.hstack([z_val, X_val]).astype(np.float32)
    fused_te = np.hstack([z_te, X_te]).astype(np.float32)

    clf_fused = FusionClassifier(fused_tr.shape[1], num_classes=2, dropout=cfg.dropout).to(device)
    clf_fused = train_classifier(clf_fused, fused_tr, y_tr, fused_val, y_val, cfg, device, out_dir, "fused", cw)
    results_fused, prob_fused = calibrate_and_eval(clf_fused, fused_val, y_val, fused_te, y_te, device)
    save_json(results_fused, os.path.join(out_dir, "results_fused.json"))
    print("[INFO] FUSED results:", json.dumps(results_fused["classification_report"]["accuracy"], indent=2))

    # ---------------- Ablation arm 1: raw features only (no AE) ----------------
    clf_raw = PlainMLPBaseline(X_tr.shape[1], num_classes=2, dropout=cfg.dropout).to(device)
    clf_raw = train_classifier(clf_raw, X_tr, y_tr, X_val, y_val, cfg, device, out_dir, "rawonly", cw)
    results_raw, _ = calibrate_and_eval(clf_raw, X_val, y_val, X_te, y_te, device)
    save_json(results_raw, os.path.join(out_dir, "results_rawonly.json"))

    # ---------------- Ablation arm 2: latent features only (AE, no raw fusion) ----------------
    clf_z = FusionClassifier(z_tr.shape[1], num_classes=2, dropout=cfg.dropout).to(device)
    clf_z = train_classifier(clf_z, z_tr, y_tr, z_val, y_val, cfg, device, out_dir, "aeonly", cw)
    results_z, _ = calibrate_and_eval(clf_z, z_val, y_val, z_te, y_te, device)
    save_json(results_z, os.path.join(out_dir, "results_aeonly.json"))

    ablation_summary = {
        "raw_only":  results_raw["classification_report"]["accuracy"],
        "ae_only":   results_z["classification_report"]["accuracy"],
        "fused":     results_fused["classification_report"]["accuracy"],
        "raw_only_attack_f1": results_raw["classification_report"]["1"]["f1-score"],
        "ae_only_attack_f1":  results_z["classification_report"]["1"]["f1-score"],
        "fused_attack_f1":    results_fused["classification_report"]["1"]["f1-score"],
        "raw_only_auc": results_raw["auc"],
        "ae_only_auc":  results_z["auc"],
        "fused_auc":    results_fused["auc"],
    }
    save_json(ablation_summary, os.path.join(out_dir, "ablation_summary.json"))
    print("[INFO] ablation summary:", ablation_summary)

    # ---------------- Per-attack-type recall (addresses "trivial binary" critique) ----------------
    y_multi_te = splits["test"]["y_multi"].values
    y_pred_fused = (prob_fused >= results_fused["threshold"]).astype(int)
    per_type = {}
    for label in sorted(set(y_multi_te)):
        mask = y_multi_te == label
        n = int(mask.sum())
        if n == 0:
            continue
        if label == "BENIGN":
            # for benign, "recall" = correctly kept as benign
            recall = float((y_pred_fused[mask] == 0).sum() / n)
        else:
            recall = float((y_pred_fused[mask] == 1).sum() / n)
        per_type[label] = {"support": n, "recall": recall}
    save_json(per_type, os.path.join(out_dir, "per_attack_type_recall.json"))

    # ---------------- Save a manifest tying everything together ----------------
    manifest = {
        "config": vars(cfg),
        "device": str(device),
        "input_dim": input_dim,
        "feature_names": feat_names,
        "total_runtime_sec": time.time() - t0,
    }
    save_json(manifest, os.path.join(out_dir, "run_manifest.json"))
    print(f"[DONE] main training complete. Everything saved under: {out_dir}")


if __name__ == "__main__":
    main()
