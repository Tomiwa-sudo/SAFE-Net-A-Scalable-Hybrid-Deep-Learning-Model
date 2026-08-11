"""
Addresses Reviewer #1: "the authors choose to collapse this rich dataset
into a trivial binary classification task." We add a genuine multiclass
head trained on the SAME fused features, but grouped into ~8 coarse
CICIoT2023 categories (documented mapping below) so the resulting
confusion matrix is actually READABLE at print size — a raw 33x33 heatmap
is unreadable in any journal column width and would violate the user's
explicit "no enlargement needed" requirement. The full 33-class recall
table (plain numbers, not a heatmap) is still produced separately by
train_main.py's per_attack_type_recall.json — nothing about resolution
is lost, only the confusion-matrix VISUALIZATION is coarsened.
"""
import os
import re
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
from models import DenseAutoencoder, FusionClassifier, count_params
from eval_utils import set_seed, get_device, save_json
from sklearn.metrics import classification_report, confusion_matrix


CATEGORY_RULES = [
    (r"^BENIGN$", "Benign"),
    (r"^DDOS-", "DDoS"),
    (r"^DOS-", "DoS"),
    (r"^MIRAI-", "Mirai/Botnet"),
    (r"^RECON-", "Reconnaissance"),
    (r"SPOOF", "Spoofing"),
    (r"MITM", "MITM"),
    (r"BRUTE", "Brute Force"),
    (r"WEB|XSS|SQL|UPLOAD", "Web-based"),
    (r"VULNERABILITYSCAN", "Reconnaissance"),
]


def to_category(label: str) -> str:
    for pattern, cat in CATEGORY_RULES:
        if re.search(pattern, label):
            return cat
    return "Other"


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    main_dir = os.path.join(run_dir(cfg), "main")
    out_dir = os.path.join(run_dir(cfg), "multiclass")
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

    cat_tr = splits["train"]["y_multi"].apply(to_category)
    cat_val = splits["val"]["y_multi"].apply(to_category)
    cat_te = splits["test"]["y_multi"].apply(to_category)
    categories = sorted(set(cat_tr) | set(cat_val) | set(cat_te))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    save_json(cat_to_idx, os.path.join(out_dir, "category_index_map.json"))

    y_tr = cat_tr.map(cat_to_idx).values.astype(np.int64)
    y_val = cat_val.map(cat_to_idx).values.astype(np.int64)
    y_te = cat_te.map(cat_to_idx).values.astype(np.int64)

    X_tr = splits["train"]["X_scaled"].values.astype(np.float32)
    X_val = splits["val"]["X_scaled"].values.astype(np.float32)
    X_te = splits["test"]["X_scaled"].values.astype(np.float32)

    ae = DenseAutoencoder(input_dim, cfg.hidden, cfg.latent, cfg.dropout).to(device)
    ae_path = os.path.join(main_dir, "best_ae.pt")
    if os.path.exists(ae_path):
        ae.load_state_dict(torch.load(ae_path, map_location=device))
        print("[MULTICLASS] loaded pretrained AE from main run (frozen encoder)")
    ae.eval()
    with torch.no_grad():
        z_tr = ae.encoder(torch.from_numpy(X_tr).to(device)).cpu().numpy()
        z_val = ae.encoder(torch.from_numpy(X_val).to(device)).cpu().numpy()
        z_te = ae.encoder(torch.from_numpy(X_te).to(device)).cpu().numpy()

    fused_tr = np.hstack([z_tr, X_tr]).astype(np.float32)
    fused_val = np.hstack([z_val, X_val]).astype(np.float32)
    fused_te = np.hstack([z_te, X_te]).astype(np.float32)

    n_classes = len(categories)
    clf = FusionClassifier(fused_tr.shape[1], num_classes=n_classes, dropout=cfg.dropout).to(device)

    from sklearn.utils.class_weight import compute_class_weight
    cw = torch.tensor(
        compute_class_weight("balanced", classes=np.arange(n_classes), y=y_tr), dtype=torch.float32
    ).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=cfg.lr_clf)
    crit = nn.CrossEntropyLoss(weight=cw)
    loader = DataLoader(TensorDataset(torch.from_numpy(fused_tr), torch.from_numpy(y_tr)),
                         batch_size=cfg.batch_size, shuffle=True)

    epochs = cfg.epochs_clf if not cfg.quick else 3
    best_val, best_state, no_improve = np.inf, None, 0
    val_x = torch.from_numpy(fused_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    curve = {"train": [], "val": []}
    for ep in range(epochs):
        clf.train()
        tr_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(clf(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(loader.dataset)
        clf.eval()
        with torch.no_grad():
            val_loss = crit(clf(val_x), val_y).item()
        curve["train"].append(tr_loss)
        curve["val"].append(val_loss)
        print(f"[MULTICLASS] epoch {ep+1}/{epochs} train={tr_loss:.4f} val={val_loss:.4f}")
        if val_loss < best_val - 1e-6:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in clf.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg.patience_clf:
                break
    clf.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(clf.state_dict(), os.path.join(out_dir, "best_clf_multiclass.pt"))
    save_json(curve, os.path.join(out_dir, "multiclass_training_curve.json"))

    clf.eval()
    with torch.no_grad():
        te_logits = clf(torch.from_numpy(fused_te).to(device))
        te_pred = torch.argmax(te_logits, dim=1).cpu().numpy()

    report = classification_report(y_te, te_pred, target_names=categories,
                                    output_dict=True, digits=4, zero_division=0)
    cm = confusion_matrix(y_te, te_pred).tolist()

    bundle = {
        "categories": categories,
        "classification_report": report,
        "confusion_matrix": cm,
        "category_rules_used": [{"pattern": p, "category": c} for p, c in CATEGORY_RULES],
    }
    save_json(bundle, os.path.join(out_dir, "results_multiclass.json"))
    print(f"[DONE] multiclass evaluation complete. Saved under: {out_dir}")


if __name__ == "__main__":
    main()
