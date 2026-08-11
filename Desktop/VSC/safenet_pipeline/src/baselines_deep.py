"""
Deep learning baselines requested by Reviewer #2: recent-style hybrid
(CNN+BiLSTM+Attention) and a Transformer-based IDS, trained/evaluated
on the IDENTICAL split and scaler as the main model and the classical
baselines. All operate on single flow rows (see models.py docstring for
why feature-vector-as-sequence is used instead of any row-order claim).
"""
import os
import sys
import time

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
from models import CNN1DBaseline, CNNBiLSTMAttention, TinyTransformer, count_params
from eval_utils import set_seed, get_device, full_evaluation, save_json


def batched_eval_loss(model, X, y, crit, device, batch_size=512):
    """Computes validation/test loss in batches instead of one giant
    forward pass -- avoids the multi-GB intermediate-activation spike
    that CNN/BiLSTM/Transformer layers produce when fed 200k+ rows at
    once (this was the exact cause of the CPU OOM crash)."""
    model.eval()
    total_loss, n = 0.0, X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            yb = torch.from_numpy(y[i:i + batch_size]).to(device)
            loss = crit(model(xb), yb)
            total_loss += loss.item() * xb.size(0)
    return total_loss / n


def batched_predict_proba(model, X, device, batch_size=512):
    """Same fix, for the final probability computation used to build
    the ROC curve / confusion matrix / threshold calibration."""
    model.eval()
    outs = []
    n = X.shape[0]
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.from_numpy(X[i:i + batch_size]).to(device)
            prob = torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy()
            outs.append(prob)
    return np.concatenate(outs)


def train_and_eval(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te, cfg, device, out_dir):
    from sklearn.utils.class_weight import compute_class_weight
    cw = torch.tensor(compute_class_weight("balanced", classes=np.array([0, 1]), y=y_tr),
                       dtype=torch.float32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_clf)
    crit = nn.CrossEntropyLoss(weight=cw)

    tr_loader = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                            batch_size=cfg.batch_size, shuffle=True)

    best_val, best_state, no_improve = np.inf, None, 0
    curve = {"train": [], "val": []}
    t0 = time.time()
    for ep in range(cfg.epochs_clf):
        model.train()
        tr_loss = 0.0
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(tr_loader.dataset)

        val_loss = batched_eval_loss(model, X_val, y_val, crit, device, batch_size=cfg.batch_size)
        curve["train"].append(tr_loss)
        curve["val"].append(val_loss)
        print(f"[{name}] epoch {ep+1}/{cfg.epochs_clf} train={tr_loss:.4f} val={val_loss:.4f}")

        if val_loss < best_val - 1e-6:
            best_val, no_improve = val_loss, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve += 1
            if no_improve >= cfg.patience_clf:
                print(f"[{name}] early stop at epoch {ep+1}")
                break
    train_time = time.time() - t0

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save(model.state_dict(), os.path.join(out_dir, f"best_{name}.pt"))
    save_json(curve, os.path.join(out_dir, f"training_curve_{name}.json"))

    val_prob = batched_predict_proba(model, X_val, device, batch_size=cfg.batch_size)
    t1 = time.time()
    te_prob = batched_predict_proba(model, X_te, device, batch_size=cfg.batch_size)
    infer_time = time.time() - t1

    from sklearn.metrics import precision_recall_curve
    prec, rec, thr = precision_recall_curve(y_val, val_prob)
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    threshold = float(thr[np.nanargmax(f1[:-1])]) if len(thr) else float(np.percentile(val_prob, 95))

    bundle = full_evaluation(y_te, te_prob, threshold=threshold)
    bundle["model_name"] = name
    bundle["param_count"] = count_params(model)
    bundle["train_time_sec"] = train_time
    bundle["inference_time_ms_per_sample"] = (infer_time / len(y_te)) * 1000
    save_json(bundle, os.path.join(out_dir, f"results_{name}.json"))
    print(f"[{name}] DONE acc={bundle['classification_report']['accuracy']:.4f} AUC={bundle['auc']}")
    return bundle


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    device = get_device(cfg.device)
    out_dir = os.path.join(run_dir(cfg), "baselines_deep")
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
    input_dim = X_tr.shape[1]

    models = {
        "CNN1D": CNN1DBaseline(input_dim),
        "CNN_BiLSTM_Attention": CNNBiLSTMAttention(input_dim),
        "TinyTransformer": TinyTransformer(input_dim),
    }
    for name, model in models.items():
        model = model.to(device)
        train_and_eval(name, model, X_tr, y_tr, X_val, y_val, X_te, y_te, cfg, device, out_dir)

    print(f"[DONE] deep baselines complete. Saved under: {out_dir}")


if __name__ == "__main__":
    main()
