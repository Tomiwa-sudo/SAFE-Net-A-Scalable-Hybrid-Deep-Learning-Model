"""
Shared evaluation + serialization utilities.

CRITICAL DESIGN RULE (this is the fix for the reviewer's #1 complaint —
narrative text, table, and confusion-matrix figure contradicting each
other): every stage in this pipeline computes its metrics ONCE, writes
them to a single `results.json`, and every downstream consumer (plots,
tables, the auto-generated results_summary.md you paste into the paper)
reads from that same file. Numbers are never re-typed or re-derived by
hand anywhere in this codebase.
"""
import json
import os
import random

import numpy as np
import torch
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, roc_auc_score, auc as auc_fn
)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(pref: str = "auto"):
    if pref == "cpu":
        return torch.device("cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def full_evaluation(y_true, y_score, y_pred=None, threshold=None, positive_label=1):
    """
    Single canonical evaluation function. Computes everything the paper
    needs (classification report dict, confusion matrix, AUC, ROC points,
    chosen threshold) and returns it as one plain-dict bundle that is
    JSON-serializable as-is. y_score = probability of the positive class.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if y_pred is None:
        assert threshold is not None, "Provide either y_pred or a threshold"
        y_pred = (y_score >= threshold).astype(int)

    report = classification_report(y_true, y_pred, output_dict=True, digits=4, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    try:
        auc_val = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_points = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    except ValueError:
        auc_val = None
        roc_points = None

    bundle = {
        "threshold": float(threshold) if threshold is not None else None,
        "classification_report": report,
        "confusion_matrix": cm,
        "confusion_matrix_labels": ["Benign(0)", "Attack(1)"],
        "auc": auc_val,
        "roc_points": roc_points,
        "n_samples": int(len(y_true)),
    }
    return bundle


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    return str(o)


def load_json(path):
    with open(path) as f:
        return json.load(f)
