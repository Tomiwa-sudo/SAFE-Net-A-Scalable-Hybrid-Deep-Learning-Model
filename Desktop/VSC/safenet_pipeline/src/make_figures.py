"""
Reads every results_*.json produced by the other stages and generates:
  - Publication-ready PNG figures (300 DPI, large fonts, tight layout,
    sized for direct insertion into a two-column IEEE paper with NO
    resizing/enlargement needed).
  - results_summary.md: every number the paper needs, auto-formatted as
    markdown tables and copy-paste-ready prose, generated directly from
    the same JSON files the figures read. Because the paper's text,
    Table II, and confusion-matrix figure will all be transcribed from
    THIS ONE FILE, they cannot contradict each other the way the
    original submission's did.

Run this LAST, after main/baselines_classical/baselines_deep/latency/
generalization_loao/multiclass have all completed.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from eval_utils import load_json

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.autolayout": False,
})


def safe_load(path):
    if os.path.exists(path):
        return load_json(path)
    print(f"[WARN] missing: {path} (skipping figures/sections that need it)")
    return None


def savefig(fig, path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[FIGURE] saved {path}")


def fig_training_curves(main_dir, out_dir):
    ae_curve = safe_load(os.path.join(main_dir, "ae_training_curve.json"))
    clf_curve = safe_load(os.path.join(main_dir, "clf_training_curve_fused.json"))
    if not ae_curve or not clf_curve:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(ae_curve["train"], label="Train", linewidth=2)
    axes[0].plot(ae_curve["val"], label="Validation", linewidth=2)
    axes[0].set_title("(a) Autoencoder Reconstruction Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("MSE Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(clf_curve["train"], label="Train", linewidth=2)
    axes[1].plot(clf_curve["val"], label="Validation", linewidth=2)
    axes[1].set_title("(b) Fusion Classifier Loss")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Cross-Entropy Loss")
    axes[1].legend(); axes[1].grid(alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_training_curves.png"))


def fig_confusion_matrix_grid(results_map, out_dir, fname, ncols=3):
    entries = [(name, r) for name, r in results_map.items() if r is not None]
    if not entries:
        return
    n = len(entries)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.0 * nrows))
    axes = np.array(axes).reshape(-1)
    for i, (name, r) in enumerate(entries):
        cm = np.array(r["confusion_matrix"])
        ax = axes[i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(name, fontsize=12)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "Attack"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "Attack"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        vmax = cm.max()
        for r_i in range(cm.shape[0]):
            for c_i in range(cm.shape[1]):
                color = "white" if cm[r_i, c_i] > vmax * 0.5 else "black"
                ax.text(c_i, r_i, f"{cm[r_i, c_i]:,}", ha="center", va="center",
                         color=color, fontsize=11, fontweight="bold")
    for j in range(len(entries), len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, fname))


def fig_roc_comparison(results_map, out_dir):
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, r in results_map.items():
        if r is None or r.get("roc_points") is None:
            continue
        fpr = r["roc_points"]["fpr"]; tpr = r["roc_points"]["tpr"]
        auc_val = r.get("auc")
        label = f"{name} (AUC={auc_val:.4f})" if auc_val is not None else name
        ax.plot(fpr, tpr, linewidth=2, label=label)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison: SAFE-Net v2 vs. Baselines")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_roc_comparison.png"))


def fig_bar_comparison(results_map, out_dir):
    names, acc, prec, rec, f1, auc_vals = [], [], [], [], [], []
    for name, r in results_map.items():
        if r is None:
            continue
        cr = r["classification_report"]
        names.append(name)
        acc.append(cr["accuracy"])
        prec.append(cr["1"]["precision"])
        rec.append(cr["1"]["recall"])
        f1.append(cr["1"]["f1-score"])
        auc_vals.append(r.get("auc") or 0)
    if not names:
        return
    metrics = {"Accuracy": acc, "Attack Precision": prec, "Attack Recall": rec,
               "Attack F1": f1, "AUC": auc_vals}
    x = np.arange(len(names))
    width = 0.15
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 1.3), 6))
    for i, (mname, vals) in enumerate(metrics.items()):
        ax.bar(x + i * width, vals, width, label=mname)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("SAFE-Net v2 vs. All Baselines")
    ax.legend(loc="lower right", ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_baseline_bar_comparison.png"))


def fig_ablation(ablation, out_dir):
    if ablation is None:
        return
    arms = ["raw_only", "ae_only", "fused"]
    labels = ["Raw Features\nOnly (no AE)", "AE Latent\nOnly (no fusion)", "Fused\n(SAFE-Net v2)"]
    acc = [ablation[f"{a}"] for a in arms]
    f1 = [ablation[f"{a}_attack_f1"] for a in arms]
    auc_vals = [ablation[f"{a}_auc"] or 0 for a in arms]

    x = np.arange(len(arms))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.bar(x - width, acc, width, label="Accuracy")
    ax.bar(x, f1, width, label="Attack F1")
    ax.bar(x + width, auc_vals, width, label="AUC")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_title("Ablation Study: Contribution of Each Component")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_ablation.png"))


def fig_latency_scaling(latency, out_dir):
    if latency is None:
        return
    curve = latency["scaling_curve"]
    bs = [c["batch_size"] for c in curve]
    per_sample = [c["per_sample_ms_mean"] for c in curve]
    per_sample_std = [c["per_sample_ms_std"] for c in curve]
    throughput = [c["throughput_samples_per_sec_mean"] for c in curve]

    fig, ax1 = plt.subplots(figsize=(8, 6))
    color1 = "tab:blue"
    ax1.set_xlabel("Concurrent Batch Size (packets processed together)")
    ax1.set_ylabel("Per-Sample Latency (ms)", color=color1)
    ax1.errorbar(bs, per_sample, yerr=per_sample_std, marker="o", color=color1, linewidth=2, capsize=3)
    ax1.set_xscale("log", base=2)
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    color2 = "tab:red"
    ax2.set_ylabel("Throughput (samples/sec)", color=color2)
    ax2.plot(bs, throughput, marker="s", color=color2, linewidth=2, linestyle="--")
    ax2.tick_params(axis="y", labelcolor=color2)

    fig.suptitle("Inference Latency and Throughput vs. Concurrent Load")
    savefig(fig, os.path.join(out_dir, "fig_latency_scaling.png"))


def fig_per_attack_recall(per_type, out_dir):
    if per_type is None:
        return
    items = sorted(per_type.items(), key=lambda kv: kv[1]["recall"])
    labels = [k for k, _ in items]
    vals = [v["recall"] for _, v in items]
    colors = ["#2ca02c" if l == "BENIGN" else "#d62728" if v < 0.9 else "#1f77b4"
              for l, v in zip(labels, vals)]

    fig, ax = plt.subplots(figsize=(9, max(5, len(labels) * 0.32)))
    ax.barh(labels, vals, color=colors)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Detection Recall")
    ax.set_title("Per-Attack-Type Detection Recall (Full 33-Class Breakdown)")
    for i, v in enumerate(vals):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_per_attack_recall.png"))


def fig_multiclass_confusion(mc, out_dir):
    if mc is None:
        return
    cats = mc["categories"]
    cm = np.array(mc["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(max(6, len(cats) * 0.9), max(5, len(cats) * 0.8)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(cats))); ax.set_xticklabels(cats, rotation=35, ha="right")
    ax.set_yticks(range(len(cats))); ax.set_yticklabels(cats)
    ax.set_xlabel("Predicted Category"); ax.set_ylabel("Actual Category")
    ax.set_title("Multiclass Confusion Matrix (Coarse-Grouped Categories)")
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax * 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig(fig, os.path.join(out_dir, "fig_multiclass_confusion.png"))


def fig_generalization(loao, out_dir):
    if loao is None or len(loao) == 0:
        return
    names = list(loao.keys())
    vals = [loao[n]["recall_detected_as_attack_despite_never_trained_on_this_class"] for n in names]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 1.4), 5.5))
    bars = ax.bar(names, vals, color="#9467bd")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Detection Rate on Unseen Attack Type")
    ax.set_title("Generalization to Previously Unseen Attack Types (Leave-One-Attack-Out)")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=20, ha="right")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_generalization_loao.png"))


def write_summary_md(all_data, out_path):
    lines = ["# SAFE-Net v2 — Auto-Generated Results Summary",
             "",
             "**Every number below is read directly from the saved `results_*.json` files.**",
             "Copy these numbers verbatim into the paper's text, Table, and figure captions —",
             "do not hand-retype them, so text/table/figure can never contradict each other again.",
             ""]

    fused = all_data.get("main_fused")
    if fused:
        cr = fused["classification_report"]
        cm = fused["confusion_matrix"]
        lines += [
            "## Main Model (SAFE-Net v2, fused) — Table II material",
            "",
            "| | Precision | Recall | F1-score | Support |",
            "|---|---|---|---|---|",
            f"| Benign (0) | {cr['0']['precision']:.4f} | {cr['0']['recall']:.4f} | {cr['0']['f1-score']:.4f} | {int(cr['0']['support'])} |",
            f"| Attack (1) | {cr['1']['precision']:.4f} | {cr['1']['recall']:.4f} | {cr['1']['f1-score']:.4f} | {int(cr['1']['support'])} |",
            "",
            f"- **Accuracy:** {cr['accuracy']:.4f}",
            f"- **AUC:** {fused['auc']}",
            f"- **Decision threshold used (F1-optimal on validation set):** {fused['threshold']:.6f}",
            "",
            "Confusion matrix (row=actual, col=predicted, [Benign, Attack]):",
            f"```\n{np.array(cm)}\n```",
            "",
            f"Prose sentence to paste directly into Results: "
            f"\"Of {cr['0']['support']:.0f} benign test samples, {cm[0][0]:,} were correctly classified "
            f"and {cm[0][1]:,} were misclassified as attacks. Of {cr['1']['support']:.0f} attack test "
            f"samples, {cm[1][1]:,} were correctly detected and {cm[1][0]:,} were missed.\"",
            "",
        ]

    ablation = all_data.get("ablation")
    if ablation:
        lines += [
            "## Ablation Study",
            "",
            "| Configuration | Accuracy | Attack F1 | AUC |",
            "|---|---|---|---|",
            f"| Raw features only (no AE) | {ablation['raw_only']:.4f} | {ablation['raw_only_attack_f1']:.4f} | {ablation['raw_only_auc']} |",
            f"| AE latent only (no raw fusion) | {ablation['ae_only']:.4f} | {ablation['ae_only_attack_f1']:.4f} | {ablation['ae_only_auc']} |",
            f"| **Fused (SAFE-Net v2)** | **{ablation['fused']:.4f}** | **{ablation['fused_attack_f1']:.4f}** | **{ablation['fused_auc']}** |",
            "",
        ]

    baselines = all_data.get("all_baselines")
    if baselines:
        lines += ["## Full Baseline Comparison Table", "",
                   "| Model | Accuracy | Attack Precision | Attack Recall | Attack F1 | AUC |",
                   "|---|---|---|---|---|---|"]
        for name, r in baselines.items():
            if r is None:
                continue
            cr = r["classification_report"]
            lines.append(
                f"| {name} | {cr['accuracy']:.4f} | {cr['1']['precision']:.4f} | "
                f"{cr['1']['recall']:.4f} | {cr['1']['f1-score']:.4f} | {r.get('auc')} |"
            )
        lines.append("")
        lines.append("**Report every row above in the paper, including any baseline that matches or "
                      "exceeds SAFE-Net v2 — do not omit inconvenient results.**")
        lines.append("")

    latency = all_data.get("latency")
    if latency:
        lines += [
            "## Latency",
            "",
            f"- **Single-sample (batch=1) latency:** {latency['headline_single_sample_latency_ms']:.4f} ms "
            f"± {latency['headline_single_sample_latency_ms_std']:.4f} ms (CPU, {latency['cpu_threads_used']} threads, "
            f"real test-set rows, mean over repeated trials after warm-up)",
            "",
            "| Batch Size | Per-Sample Latency (ms) | Throughput (samples/sec) |",
            "|---|---|---|",
        ]
        for c in latency["scaling_curve"]:
            lines.append(f"| {c['batch_size']} | {c['per_sample_ms_mean']:.4f} ± {c['per_sample_ms_std']:.4f} | "
                          f"{c['throughput_samples_per_sec_mean']:.1f} |")
        lines.append("")

    loao = all_data.get("loao")
    if loao:
        lines += ["## Generalization to Unseen Attack Types (Leave-One-Attack-Out)", "",
                   "| Held-Out Attack Type | Test Samples | Detected as Attack (never trained on this label) |",
                   "|---|---|---|"]
        for name, r in loao.items():
            lines.append(f"| {name} | {r['n_test_rows']} | "
                          f"{r['recall_detected_as_attack_despite_never_trained_on_this_class']*100:.1f}% |")
        lines.append("")

    per_type = all_data.get("per_attack")
    if per_type:
        lines += ["## Full Per-Attack-Type Recall (all classes present in sample)", "",
                   "| Attack Type | Support | Recall |", "|---|---|---|"]
        for name, r in sorted(per_type.items(), key=lambda kv: -kv[1]["recall"]):
            lines.append(f"| {name} | {r['support']} | {r['recall']:.4f} |")
        lines.append("")

    ms_aggregate = all_data.get("ms_aggregate")
    ms_sig = all_data.get("ms_sig")
    if ms_aggregate:
        n_seeds = next(iter(ms_aggregate.values()))["n_seeds"]
        lines += [f"## Multi-Seed Evaluation (n={n_seeds} independent seeds, mean ± std)", "",
                   "| Model | Accuracy | AUC | Attack F1 |", "|---|---|---|---|"]
        for name, v in ms_aggregate.items():
            lines.append(f"| {name} | {v['accuracy_mean']:.4f} ± {v['accuracy_std']:.4f} | "
                          f"{v['auc_mean']:.4f} ± {v['auc_std']:.4f} | "
                          f"{v['attack_f1_mean']:.4f} ± {v['attack_f1_std']:.4f} |")
        lines.append("")
        if ms_sig:
            lines += ["### Statistical Significance (paired t-test, SAFE-Net_fused vs. each comparator)", "",
                       "| Comparison | Metric | Mean Diff | p-value | Interpretation |",
                       "|---|---|---|---|---|"]
            for t in ms_sig:
                lines.append(f"| {t['model_a']} vs {t['model_b']} | {t['metric']} | "
                              f"{t['mean_diff']:.4f} | {t.get('paired_ttest_p')} | {t['interpretation']} |")
            lines.append("")

    hc_comparison = all_data.get("hc_comparison")
    if hc_comparison:
        lines += ["## Targeted Improvement: Hard-to-Detect Attack Categories", "",
                   f"Hard classes (validation recall < threshold): {', '.join(hc_comparison.get('hard_classes', []))}",
                   "",
                   "| Attack Type | Support | Recall Before | Recall After | Improvement |",
                   "|---|---|---|---|---|"]
        for cls, v in hc_comparison.get("per_class_before_vs_after", {}).items():
            b = v["recall_before"]; a = v["recall_after"]; imp = v["improvement"]
            lines.append(f"| {cls} | {v['support']} | {b if b is None else f'{b:.4f}'} | "
                          f"{a if a is None else f'{a:.4f}'} | {imp if imp is None else f'{imp:+.4f}'} |")
        lines.append("")

    ae_vs_loao = all_data.get("ae_vs_loao")
    ae_anomaly_results = all_data.get("ae_anomaly_results")
    if ae_anomaly_results:
        cr = ae_anomaly_results["classification_report"]
        lines += ["## Unsupervised Autoencoder as a Standalone Zero-Day Detector",
                   "", f"Trained on benign traffic ONLY -- zero attack labels used at any point.",
                   f"Overall test accuracy: {cr['accuracy']:.4f}, AUC: {ae_anomaly_results['auc']}",
                   ""]
        if ae_vs_loao:
            lines += ["| Unseen Attack Type | Supervised Model (LOAO) Recall | Unsupervised AE Recall |",
                       "|---|---|---|"]
            for k, v in ae_vs_loao.items():
                lines.append(f"| {k} | {v['supervised_LOAO_recall_on_unseen']:.4f} | "
                              f"{v['unsupervised_AE_anomaly_recall']} |")
            lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"[SUMMARY] wrote {out_path}")


def fig_multi_seed_significance(aggregate, sig_tests, out_dir):
    if aggregate is None:
        return
    models = list(aggregate.keys())
    means = [aggregate[m]["accuracy_mean"] for m in models]
    stds = [aggregate[m]["accuracy_std"] for m in models]
    fig, ax = plt.subplots(figsize=(max(7, len(models) * 1.3), 6))
    bars = ax.bar(models, means, yerr=stds, capsize=5, color="#4c72b0")
    ax.set_ylabel("Accuracy (mean ± std across seeds)")
    n = aggregate[models[0]]["n_seeds"]
    ax.set_title(f"Multi-Seed Comparison (n={n} seeds) with Significance Testing")
    ax.set_xticks(range(len(models))); ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    if sig_tests:
        sig_map = {(t["model_a"], t["model_b"]): t for t in sig_tests if t["metric"] == "accuracy"}
        base_idx = models.index("SAFE-Net_fused") if "SAFE-Net_fused" in models else 0
        for i, m in enumerate(models):
            key = ("SAFE-Net_fused", m)
            if key in sig_map:
                p = sig_map[key].get("paired_ttest_p")
                label = f"p={p:.3f}" if p is not None else "n/a"
                ax.text(i, means[i] + stds[i] + 0.015, label, ha="center", fontsize=8)
    savefig(fig, os.path.join(out_dir, "fig_multi_seed_significance.png"))


def fig_feature_importance(results_map, out_dir, top_n=15):
    entries = [(name, r["feature_importance"]) for name, r in results_map.items()
               if r is not None and "feature_importance" in r]
    if not entries:
        return
    n = len(entries)
    fig, axes = plt.subplots(1, n, figsize=(6.5 * n, 6))
    if n == 1:
        axes = [axes]
    for ax, (name, fi) in zip(axes, entries):
        items = list(fi.items())[:top_n][::-1]
        labels = [k for k, _ in items]
        vals = [v for _, v in items]
        ax.barh(labels, vals, color="#55a868")
        ax.set_title(f"{name} — Top {top_n} Feature Importances")
        ax.set_xlabel("Importance")
        ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    savefig(fig, os.path.join(out_dir, "fig_feature_importance.png"))


def fig_hard_class_before_after(comparison, out_dir):
    if comparison is None:
        return
    items = comparison.get("per_class_before_vs_after", {})
    if not items:
        return
    labels = list(items.keys())
    before = [items[k]["recall_before"] or 0 for k in labels]
    after = [items[k]["recall_after"] or 0 for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.9), 6))
    ax.bar(x - width/2, before, width, label="Before (base model)", color="#c44e52")
    ax.bar(x + width/2, after, width, label="After (hard-class-improved model)", color="#55a868")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Detection Recall")
    ax.set_ylim(0, 1.05)
    ax.set_title("Targeted Improvement: Hard-to-Detect Attack Categories, Before vs. After")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_hard_class_before_after.png"))


def fig_ae_anomaly_vs_supervised(comparison, out_dir):
    if comparison is None or len(comparison) == 0:
        return
    labels = list(comparison.keys())
    sup = [comparison[k]["supervised_LOAO_recall_on_unseen"] for k in labels]
    unsup = [comparison[k]["unsupervised_AE_anomaly_recall"] or 0 for k in labels]
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 6))
    ax.bar(x - width/2, sup, width, label="Supervised Fusion Classifier (LOAO)", color="#4c72b0")
    ax.bar(x + width/2, unsup, width, label="Unsupervised AE (benign-only, no labels)", color="#dd8452")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Detection Recall on Unseen Attack Type")
    ax.set_ylim(0, 1.05)
    ax.set_title("Unsupervised Anomaly Detection vs. Supervised Model on Unseen Attacks")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    savefig(fig, os.path.join(out_dir, "fig_ae_anomaly_vs_supervised.png"))


def main():
    cfg = build_config_from_cli()
    rd = run_dir(cfg)
    main_dir = os.path.join(rd, "main")
    base_c_dir = os.path.join(rd, "baselines_classical")
    base_d_dir = os.path.join(rd, "baselines_deep")
    lat_dir = os.path.join(rd, "latency")
    loao_dir = os.path.join(rd, "generalization_loao")
    mc_dir = os.path.join(rd, "multiclass")
    ms_dir = os.path.join(rd, "multi_seed")
    hc_dir = os.path.join(rd, "hard_class_improvement")
    ae_dir = os.path.join(rd, "ae_anomaly_detector")
    fig_dir = os.path.join(rd, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    fused = safe_load(os.path.join(main_dir, "results_fused.json"))
    rawonly = safe_load(os.path.join(main_dir, "results_rawonly.json"))
    aeonly = safe_load(os.path.join(main_dir, "results_aeonly.json"))
    ablation = safe_load(os.path.join(main_dir, "ablation_summary.json"))
    per_attack = safe_load(os.path.join(main_dir, "per_attack_type_recall.json"))
    latency = safe_load(os.path.join(lat_dir, "latency_results.json"))
    loao = safe_load(os.path.join(loao_dir, "loao_results.json"))
    mc = safe_load(os.path.join(mc_dir, "results_multiclass.json"))
    ms_aggregate = safe_load(os.path.join(ms_dir, "multi_seed_aggregate.json"))
    ms_sig = safe_load(os.path.join(ms_dir, "significance_tests.json"))
    hc_comparison = safe_load(os.path.join(hc_dir, "before_after_comparison.json"))
    ae_vs_loao = safe_load(os.path.join(ae_dir, "ae_vs_supervised_loao_comparison.json"))
    ae_anomaly_results = safe_load(os.path.join(ae_dir, "results_ae_anomaly.json"))

    all_baselines = {"SAFE-Net v2 (Fused)": fused}
    for name in ["LogisticRegression", "RandomForest", "IsolationForest", "OneClassSVM", "XGBoost"]:
        all_baselines[name] = safe_load(os.path.join(base_c_dir, f"results_{name}.json"))
    for name in ["CNN1D", "CNN_BiLSTM_Attention", "TinyTransformer"]:
        all_baselines[name] = safe_load(os.path.join(base_d_dir, f"results_{name}.json"))

    # ---- Figures ----
    fig_training_curves(main_dir, fig_dir)
    fig_confusion_matrix_grid(
        {"SAFE-Net v2": fused, "RandomForest": all_baselines.get("RandomForest"),
         "XGBoost": all_baselines.get("XGBoost"), "CNN_BiLSTM_Attention": all_baselines.get("CNN_BiLSTM_Attention")},
        fig_dir, "fig_confusion_matrices.png"
    )
    fig_roc_comparison(all_baselines, fig_dir)
    fig_bar_comparison(all_baselines, fig_dir)
    fig_ablation(ablation, fig_dir)
    fig_latency_scaling(latency, fig_dir)
    fig_per_attack_recall(per_attack, fig_dir)
    fig_multiclass_confusion(mc, fig_dir)
    fig_generalization(loao, fig_dir)
    fig_multi_seed_significance(ms_aggregate, ms_sig, fig_dir)
    fig_feature_importance(
        {"RandomForest": all_baselines.get("RandomForest"), "XGBoost": all_baselines.get("XGBoost")},
        fig_dir
    )
    fig_hard_class_before_after(hc_comparison, fig_dir)
    fig_ae_anomaly_vs_supervised(ae_vs_loao, fig_dir)

    all_data = {
        "main_fused": fused, "ablation": ablation, "all_baselines": all_baselines,
        "latency": latency, "loao": loao, "per_attack": per_attack,
        "ms_aggregate": ms_aggregate, "ms_sig": ms_sig,
        "hc_comparison": hc_comparison, "ae_vs_loao": ae_vs_loao,
        "ae_anomaly_results": ae_anomaly_results,
    }
    write_summary_md(all_data, os.path.join(rd, "results_summary.md"))
    print(f"\n[DONE] all figures + results_summary.md written under: {rd}")


if __name__ == "__main__":
    main()
