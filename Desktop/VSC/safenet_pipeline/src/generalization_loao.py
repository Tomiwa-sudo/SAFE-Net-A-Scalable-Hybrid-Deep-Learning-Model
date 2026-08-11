"""

"""
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from data_utils import (
    has_csv_files,
    list_csv_files, get_cached_sample,
    preprocess, split, scale, make_synthetic_dataset,
)
from models import DenseAutoencoder, FusionClassifier
from eval_utils import set_seed, get_device, save_json


def quick_train_ae(ae, X_tr, cfg, device, epochs):
    opt = torch.optim.Adam(ae.parameters(), lr=cfg.lr_ae)
    crit = nn.MSELoss()
    loader = DataLoader(TensorDataset(torch.from_numpy(X_tr)), batch_size=cfg.batch_size, shuffle=True)
    for ep in range(epochs):
        ae.train()
        for (b,) in loader:
            b = b.to(device)
            opt.zero_grad()
            recon, _ = ae(b)
            loss = crit(recon, b)
            loss.backward()
            opt.step()
    return ae


def quick_train_clf(clf, feat_tr, y_tr, cfg, device, epochs, class_weight):
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.lr_clf)
    crit = nn.CrossEntropyLoss(weight=class_weight.to(device))
    loader = DataLoader(TensorDataset(torch.from_numpy(feat_tr), torch.from_numpy(y_tr)),
                         batch_size=cfg.batch_size, shuffle=True)
    for ep in range(epochs):
        clf.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(clf(xb), yb)
            loss.backward()
            opt.step()
    return clf


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = os.path.join(run_dir(cfg), "generalization_loao")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.quick and not has_csv_files(cfg.data_dir):
        make_synthetic_dataset(cfg.data_dir)

    files = list_csv_files(cfg.data_dir)
    df, _, was_cached = get_cached_sample(files, cfg, run_dir(cfg))
    print("[INFO] " + ("loaded cached" if was_cached else "built new") + f" sample: {df.shape}")
    X, y_bin, y_multi, feat_names = preprocess(df)
    splits = split(X, y_bin, y_multi, cfg)
    splits, scaler = scale(splits)
    input_dim = X.shape[1]

    epochs_ae = 5 if not cfg.quick else 2
    epochs_clf = 5 if not cfg.quick else 2

    results = {}
    available_attacks = set(splits["train"]["y_multi"].unique())
    tested = [a for a in cfg.loao_attack_subset if a in available_attacks]
    if not tested:
        tested = [a for a in available_attacks if a != "BENIGN"][:2]
    print(f"[LOAO] testing generalization to unseen: {tested}")

    for held_out in tested:
        print(f"\n[LOAO] === holding out {held_out} ===")
        tr_mask = splits["train"]["y_multi"].values != held_out
        val_mask = splits["val"]["y_multi"].values != held_out
        te_mask_heldout_only = splits["test"]["y_multi"].values == held_out

        if te_mask_heldout_only.sum() == 0:
            print(f"[LOAO] skip {held_out}: no test rows available")
            continue

        X_tr = splits["train"]["X_scaled"].values[tr_mask].astype(np.float32)
        y_tr = splits["train"]["y_bin"].values[tr_mask].astype(np.int64)
        X_val = splits["val"]["X_scaled"].values[val_mask].astype(np.float32)
        X_te_held = splits["test"]["X_scaled"].values[te_mask_heldout_only].astype(np.float32)

        from sklearn.utils.class_weight import compute_class_weight
        cw = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr),
                           dtype=torch.float32)

        ae = DenseAutoencoder(input_dim, cfg.hidden, cfg.latent, cfg.dropout).to(device)
        ae = quick_train_ae(ae, X_tr, cfg, device, epochs_ae)

        ae.eval()
        with torch.no_grad():
            z_tr = ae.encoder(torch.from_numpy(X_tr).to(device)).cpu().numpy()
            z_te_held = ae.encoder(torch.from_numpy(X_te_held).to(device)).cpu().numpy()

        fused_tr = np.hstack([z_tr, X_tr]).astype(np.float32)
        fused_te_held = np.hstack([z_te_held, X_te_held]).astype(np.float32)

        clf = FusionClassifier(fused_tr.shape[1], num_classes=2, dropout=cfg.dropout).to(device)
        clf = quick_train_clf(clf, fused_tr, y_tr, cfg, device, epochs_clf, cw)

        clf.eval()
        with torch.no_grad():
            probs = torch.softmax(clf(torch.from_numpy(fused_te_held).to(device)), dim=1)[:, 1].cpu().numpy()
        preds = (probs >= 0.5).astype(int)  # unseen-attack rows: any positive threshold is informative;
                                             # 0.5 used for simplicity/consistency across held-out classes
        recall_on_unseen = float((preds == 1).mean())

        results[held_out] = {
            "n_test_rows": int(te_mask_heldout_only.sum()),
            "recall_detected_as_attack_despite_never_trained_on_this_class": recall_on_unseen,
            "mean_predicted_attack_probability": float(probs.mean()),
        }
        print(f"[LOAO] {held_out}: detected {recall_on_unseen*100:.1f}% of never-before-seen samples as Attack")

    save_json(results, os.path.join(out_dir, "loao_results.json"))
    print(f"\n[DONE] LOAO generalization test complete. Saved under: {out_dir}")


if __name__ == "__main__":
    main()
