"""
Latency benchmark — fixes the original latency.py bug where "per packet"
latency was actually a batch time divided by (batch_size * seq_len), an
amortized number dressed up as real-time single-packet latency.

This version:
  1. Uses REAL held-out test rows (cycled to fill each batch size), not
     torch.randn dummy data.
  2. Reports genuine single-sample latency (batch_size=1) as the headline
     "ms per packet" number.
  3. ALSO produces a throughput-SCALING curve across increasing concurrent
     batch sizes (1..1024), directly answering Reviewer #1's request for
     "a graph showing how inference latency scales as the number of
     concurrent packets per second scales exponentially."
  4. Runs on CPU by default (matching the paper's stated experimental
     setup) with a fixed thread count so the number is reproducible.
  5. Repeats every measurement many times after a warm-up period and
     reports mean +/- std, not a single lucky timing.
"""
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from config import build_config_from_cli, run_dir
from data_utils import (
    has_csv_files,
    list_csv_files, get_cached_sample,
    preprocess, split, scale, make_synthetic_dataset,
)
from models import DenseAutoencoder, FusionClassifier
from eval_utils import set_seed, save_json


BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def benchmark_pipeline(ae, clf, X_pool, batch_sizes, n_repeats=50, warmup=10, device="cpu"):
    ae.eval()
    clf.eval()
    results = []
    n_pool = X_pool.shape[0]

    for bs in batch_sizes:
        idx = np.resize(np.arange(n_pool), bs * (n_repeats + warmup))
        np.random.shuffle(idx)
        batches = [torch.from_numpy(X_pool[idx[i * bs:(i + 1) * bs]]).to(device)
                   for i in range(n_repeats + warmup)]

        # warm-up (not timed) -- lets caches/threading settle
        with torch.no_grad():
            for b in batches[:warmup]:
                z = ae.encoder(b)
                fused = torch.cat([z, b], dim=1)
                _ = clf(fused)

        times = []
        with torch.no_grad():
            for b in batches[warmup:]:
                t0 = time.perf_counter()
                z = ae.encoder(b)
                fused = torch.cat([z, b], dim=1)
                _ = clf(fused)
                t1 = time.perf_counter()
                times.append(t1 - t0)

        times = np.array(times)
        per_batch_ms = times * 1000
        per_sample_ms = per_batch_ms / bs
        throughput = bs / times  # samples/sec

        results.append({
            "batch_size": bs,
            "per_batch_ms_mean": float(per_batch_ms.mean()),
            "per_batch_ms_std": float(per_batch_ms.std()),
            "per_sample_ms_mean": float(per_sample_ms.mean()),
            "per_sample_ms_std": float(per_sample_ms.std()),
            "throughput_samples_per_sec_mean": float(throughput.mean()),
            "throughput_samples_per_sec_std": float(throughput.std()),
            "n_repeats": n_repeats,
        })
        print(f"[LATENCY] batch={bs:>4}  per-sample={per_sample_ms.mean():.4f}±{per_sample_ms.std():.4f} ms  "
              f"throughput={throughput.mean():.1f} samples/sec")
    return results


def main():
    cfg = build_config_from_cli()
    set_seed(cfg.seed)
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    device = torch.device("cpu")  # deliberate: matches the paper's stated hardware, and
                                   # keeps the number honest/reproducible across machines
    main_dir = os.path.join(run_dir(cfg), "main")
    out_dir = os.path.join(run_dir(cfg), "latency")
    os.makedirs(out_dir, exist_ok=True)

    if cfg.quick and not has_csv_files(cfg.data_dir):
        make_synthetic_dataset(cfg.data_dir)

    files = list_csv_files(cfg.data_dir)
    df, _, was_cached = get_cached_sample(files, cfg, run_dir(cfg))
    print("[INFO] " + ("loaded cached" if was_cached else "built new") + f" sample: {df.shape}")
    X, y_bin, y_multi, feat_names = preprocess(df)
    splits = split(X, y_bin, y_multi, cfg)
    splits, scaler = scale(splits)
    X_te = splits["test"]["X_scaled"].values.astype(np.float32)
    input_dim = X_te.shape[1]

    ae = DenseAutoencoder(input_dim, cfg.hidden, cfg.latent, cfg.dropout).to(device)
    ae.load_state_dict(torch.load(os.path.join(main_dir, "best_ae.pt"), map_location=device))
    clf = FusionClassifier(cfg.latent + input_dim, num_classes=2, dropout=cfg.dropout).to(device)
    clf.load_state_dict(torch.load(os.path.join(main_dir, "best_clf_fused.pt"), map_location=device))

    batch_sizes = BATCH_SIZES if not cfg.quick else [1, 2, 4, 8]
    n_repeats = 50 if not cfg.quick else 5
    results = benchmark_pipeline(ae, clf, X_te, batch_sizes, n_repeats=n_repeats, device=device)

    headline = results[0]  # batch_size == 1
    summary = {
        "headline_single_sample_latency_ms": headline["per_sample_ms_mean"],
        "headline_single_sample_latency_ms_std": headline["per_sample_ms_std"],
        "note": (
            "headline number is a genuine single-sample (batch_size=1) forward pass "
            "through AE-encoder + fusion classifier, timed on CPU with real test rows, "
            "mean over repeated trials after warm-up. This replaces the old batch-time / "
            "(batch_size * seq_len) computation."
        ),
        "cpu_threads_used": torch.get_num_threads(),
        "scaling_curve": results,
    }
    save_json(summary, os.path.join(out_dir, "latency_results.json"))
    print(f"[DONE] latency benchmark complete. Saved under: {out_dir}")
    print(f"[HEADLINE] single-sample latency: {headline['per_sample_ms_mean']:.4f} ms "
          f"± {headline['per_sample_ms_std']:.4f} ms")


if __name__ == "__main__":
    main()
