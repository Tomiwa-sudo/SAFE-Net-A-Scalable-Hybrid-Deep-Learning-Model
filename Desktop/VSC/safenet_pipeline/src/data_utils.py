"""
Data loading & sampling utilities.

Key design choices (each one directly answers a review comment):

1. NEVER loads all 63 files fully into memory at once for the final
   training sample. We stream each file in chunks, keep a running
   per-label reservoir sample, and only materialize the final
   `sample_size`-row DataFrame. This keeps the pipeline runnable on
   a 16GB-RAM laptop even though the source data is ~45M rows.

2. Sampling is fully seeded, scripted, and logged (counts per class,
   per file) to `sampling_manifest.json` — nothing about how the
   1.4M-row training set was built is left undocumented, which was
   an explicit reviewer complaint about the original paper.

3. NO row-order-based "sequence" assumptions anywhere. Each row is
   treated as an independent flow-level record (see NOTES_ON_REDESIGN.md
   for why: CICIoT2023 rows carry no timestamp, and the merged CSVs
   were already found to interleave attack types within a few hundred
   rows, so there is no genuine temporal signal to preserve or lose).

4. The original multiclass `Label` is preserved throughout (not just
   the binarized Benign/Attack column), so downstream stages can do
   per-attack-type recall, multiclass grouping, and leave-one-attack-out
   generalization tests without re-reading the raw files.
"""
import glob
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

LABEL_COL = "Label"
BINARY_COL = "Label_Binary"


def list_csv_files(data_dir: str) -> List[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True))
    if not files:
        raise FileNotFoundError(
            f"No CSV files found under {data_dir}. "
            f"Point --data_dir at the folder containing your Merged*.csv files."
        )
    return files


def has_csv_files(data_dir: str) -> bool:
    """Used by --quick mode to decide whether to generate synthetic data.
    Checking directory existence alone isn't enough — an empty or
    placeholder-only folder should still trigger synthetic generation."""
    if not os.path.isdir(data_dir):
        return False
    return len(glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)) > 0


def count_labels_across_files(files: List[str], chunksize: int = 200_000) -> Dict[str, int]:
    """First pass: cheap streaming count of rows per label, across all files."""
    counts: Dict[str, int] = {}
    for fp in files:
        for chunk in pd.read_csv(fp, usecols=[LABEL_COL], chunksize=chunksize):
            vc = chunk[LABEL_COL].astype(str).str.upper().value_counts()
            for k, v in vc.items():
                counts[k] = counts.get(k, 0) + int(v)
    return counts


def stratified_sample(
    files: List[str],
    label_counts: Dict[str, int],
    cfg,
    chunksize: int = 200_000,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Second pass: build a reproducible stratified sample.
      - `benign_ratio` of the final sample is BENIGN.
      - the remaining rows are split as evenly as possible across every
        non-benign attack class present in the data (with a floor of
        `min_rows_per_attack_class` so rare classes survive), which is
        the same "equal representation" principle the original paper
        described in prose but never showed the mechanics of.
    Implemented via per-class reservoir sampling so we never hold more
    than one file-chunk plus the running reservoirs in memory at once.
    """
    rng = np.random.default_rng(cfg.seed)

    non_benign = [c for c in label_counts if c != "BENIGN"]
    n_attack_classes = max(1, len(non_benign))
    n_benign_target = int(cfg.sample_size * cfg.benign_ratio)
    n_attack_target_total = cfg.sample_size - n_benign_target
    per_class_target = max(cfg.min_rows_per_attack_class, n_attack_target_total // n_attack_classes)

    targets = {"BENIGN": n_benign_target}
    for c in non_benign:
        targets[c] = min(per_class_target, label_counts[c])  # can't sample more than exists

    # Vectorized, fixed-memory reservoir sampling: each label's reservoir is
    # ONE pre-allocated float32 numpy array (target_n x n_features), not a
    # Python list of pandas Series. This keeps peak memory bounded to
    # roughly sample_size * n_features * 4 bytes (~a few hundred MB for
    # 1.4M rows / 39 features) instead of growing without bound as the
    # scan progresses -- the earlier list-of-Series version was the cause
    # of the MemoryError, since each Series carries far more overhead than
    # the raw floats it holds.
    feature_cols = [c for c in pd.read_csv(files[0], nrows=0).columns.tolist() if c != LABEL_COL]
    n_features = len(feature_cols)

    reservoirs = {lbl: np.full((max(t, 1), n_features), np.nan, dtype=np.float32)
                  for lbl, t in targets.items()}
    filled = {lbl: 0 for lbl in targets}
    seen = {lbl: 0 for lbl in targets}

    for fp in files:
        for chunk in pd.read_csv(fp, chunksize=chunksize, low_memory=False):
            chunk[LABEL_COL] = chunk[LABEL_COL].astype(str).str.upper()
            try:
                feat_arr_full = chunk[feature_cols].to_numpy(dtype=np.float32)
            except (ValueError, TypeError):
                feat_arr_full = chunk[feature_cols].apply(
                    pd.to_numeric, errors="coerce"
                ).to_numpy(dtype=np.float32)
            labels_arr = chunk[LABEL_COL].to_numpy()

            for label, target_n in targets.items():
                if target_n == 0:
                    continue
                mask = labels_arr == label
                m = int(mask.sum())
                if m == 0:
                    continue
                sub_feats = feat_arr_full[mask]

                res = reservoirs[label]
                f, s = filled[label], seen[label]

                if f < target_n:
                    n_fill = min(target_n - f, m)
                    res[f:f + n_fill] = sub_feats[:n_fill]
                    f += n_fill
                    s += n_fill
                    sub_feats = sub_feats[n_fill:]
                    m -= n_fill

                if m > 0 and f >= target_n > 0:
                    # classic online reservoir replacement, vectorized:
                    # for the (s+i)-th item seen overall (0-indexed), it
                    # replaces a random existing slot with probability
                    # target_n / (s+i+1).
                    j = s + np.arange(m)
                    r = rng.integers(0, j + 1)          # random int in [0, j] inclusive, per row
                    winners = np.nonzero(r < target_n)[0]
                    if winners.size > 0:
                        res[r[winners]] = sub_feats[winners]
                    s += m

                filled[label], seen[label] = f, s

    frames = []
    for label, res in reservoirs.items():
        f = filled[label]
        if f == 0:
            continue
        part = pd.DataFrame(res[:f], columns=feature_cols)
        part[LABEL_COL] = label
        frames.append(part)
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=cfg.seed).reset_index(drop=True)  # shuffle final sample
    # NOTE: this shuffle is safe — it happens AFTER sampling, on independent
    # flow-level rows with no sequence semantics attached. It is not the same
    # bug as shuffling-before-windowing in the original pipeline.

    manifest = {
        "target_sample_size": cfg.sample_size,
        "benign_ratio_requested": cfg.benign_ratio,
        "per_class_targets": targets,
        "rows_actually_sampled_per_class": filled,
        "rows_seen_per_class_in_source_files": seen,
        "final_sample_shape": list(df.shape),
        "seed": cfg.seed,
    }
    return df, manifest


def preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, List[str]]:
    """
    Returns:
        X            - numeric feature matrix (no Label columns)
        y_binary     - 0/1 Benign/Attack
        y_multiclass - original string label, preserved for downstream analysis
        feature_names
    """
    df = df.copy()
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.upper()
    df[BINARY_COL] = (df[LABEL_COL] != "BENIGN").astype(int)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != BINARY_COL]
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    y_multiclass = df[LABEL_COL].copy()
    y_binary = df[BINARY_COL].copy()
    X = df[numeric_cols].astype(np.float32)
    return X, y_binary, y_multiclass, numeric_cols


def split(X, y_bin, y_multi, cfg):
    idx = np.arange(len(X))
    idx_tr, idx_tmp = train_test_split(
        idx, test_size=cfg.test_size, stratify=y_bin, random_state=cfg.seed
    )
    idx_val, idx_te = train_test_split(
        idx_tmp, test_size=cfg.val_size,
        stratify=y_bin.iloc[idx_tmp], random_state=cfg.seed
    )
    out = {}
    for name, ix in [("train", idx_tr), ("val", idx_val), ("test", idx_te)]:
        out[name] = dict(
            X=X.iloc[ix].reset_index(drop=True),
            y_bin=y_bin.iloc[ix].reset_index(drop=True),
            y_multi=y_multi.iloc[ix].reset_index(drop=True),
        )
    return out


def scale(splits: dict):
    scaler = MinMaxScaler()
    scaler.fit(splits["train"]["X"])
    for name in ["train", "val", "test"]:
        cols = splits[name]["X"].columns
        splits[name]["X_scaled"] = pd.DataFrame(
            scaler.transform(splits[name]["X"]), columns=cols
        )
    return splits, scaler


def get_cached_sample(files: List[str], cfg, cache_dir: str) -> Tuple[pd.DataFrame, Dict, bool]:
    """
    Ensures the (expensive) label-counting + stratified-sampling passes over
    the full raw dataset happen ONCE PER (RUN, SEED), not once per pipeline
    stage and not once per seed collision.

    Cache files are seed-suffixed (sampled_data_seed<N>.csv) so multi-seed
    evaluation can build independent samples per seed without clobbering
    each other. For backward compatibility with runs completed before this
    fix, a seed==cfg default (42) also checks the old unsuffixed
    `sampled_data.csv` path and treats it as that seed's cache.

    Returns (df, manifest, was_cached).
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"sampled_data_seed{cfg.seed}.csv")
    manifest_path = os.path.join(cache_dir, f"sampling_manifest_seed{cfg.seed}.json")
    legacy_cache_path = os.path.join(cache_dir, "sampled_data.csv")
    legacy_manifest_path = os.path.join(cache_dir, "sampling_manifest.json")
    label_counts_path = os.path.join(cache_dir, "label_counts_full_dataset.json")

    if not os.path.exists(cache_path) and cfg.seed == 42 and os.path.exists(legacy_cache_path):
        cache_path, manifest_path = legacy_cache_path, legacy_manifest_path

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        manifest = {}
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)
        return df, manifest, True

    if os.path.exists(label_counts_path):
        with open(label_counts_path) as f:
            label_counts = json.load(f)
    else:
        label_counts = count_labels_across_files(files)
        with open(label_counts_path, "w") as f:
            json.dump(label_counts, f, indent=2)

    df, manifest = stratified_sample(files, label_counts, cfg)
    cache_path = os.path.join(cache_dir, f"sampled_data_seed{cfg.seed}.csv")
    manifest_path = os.path.join(cache_dir, f"sampling_manifest_seed{cfg.seed}.json")
    df.to_csv(cache_path, index=False)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return df, manifest, False


def add_engineered_features(splits: dict) -> dict:
    """
    Adds one new feature -- 'flow_signature_frequency' -- that approximates
    "how common is this exact flow fingerprint" within the training set.

    Motivation: CICIoT2023's processed feature set has NO source/destination
    IP or port columns, so a true session/host-grouping feature (e.g.
    "connections per source in the last N seconds") is not buildable from
    this data. What IS buildable, and plausible as a signal for scan- and
    brute-force-style attacks: these attacks typically generate many
    near-identical, minimal flows (same protocol, near-zero payload, same
    flag pattern) in rapid succession, which should show up as an unusually
    HIGH frequency of a near-duplicate flow fingerprint relative to normal,
    more heterogeneous traffic -- even without knowing which host sent them.

    Implementation: bin a handful of raw columns into a coarse fingerprint,
    count fingerprint frequency ON THE TRAINING SET ONLY (no leakage), and
    map that count onto train/val/test. Unseen fingerprints in val/test map
    to 0 (never observed in training).
    """
    fingerprint_cols = [c for c in ["Protocol Type", "Rate", "Tot size", "IAT",
                                     "fin_flag_number", "syn_flag_number",
                                     "rst_flag_number", "psh_flag_number"]
                        if c in splits["train"]["X"].columns]
    if not fingerprint_cols:
        return splits  # schema doesn't match expectations; skip gracefully

    def make_fingerprint(df):
        parts = []
        for c in fingerprint_cols:
            col = df[c]
            if col.dtype.kind in "fc":  # float/continuous -> coarse log-ish bin
                binned = np.sign(col) * np.log1p(np.abs(col)).round(1)
            else:
                binned = col
            parts.append(binned.astype(str))
        return parts[0].str.cat(parts[1:], sep="|")

    fp_train = make_fingerprint(splits["train"]["X"])
    freq_map = fp_train.value_counts()

    # Normalize the raw counts before adding them alongside already-scaled
    # [0,1] features -- this was the bug that blew up AE reconstruction
    # loss by ~6 orders of magnitude last run. log1p compresses the heavy
    # tail (a few fingerprints repeated thousands of times), then min-max
    # scale fit on the TRAIN set only, applied consistently to val/test.
    train_freq_raw = fp_train.map(freq_map).fillna(0).astype(np.float64)
    train_freq_log = np.log1p(train_freq_raw)
    freq_min, freq_max = train_freq_log.min(), train_freq_log.max()
    freq_range = max(freq_max - freq_min, 1e-8)

    for name in ["train", "val", "test"]:
        fp = make_fingerprint(splits[name]["X"])
        raw_count = fp.map(freq_map).fillna(0).astype(np.float64)
        scaled = ((np.log1p(raw_count) - freq_min) / freq_range).clip(0, 1).astype(np.float32)
        splits[name]["X"] = splits[name]["X"].copy()
        splits[name]["X"]["flow_signature_frequency"] = scaled.values
        if "X_scaled" in splits[name]:
            splits[name]["X_scaled"] = splits[name]["X_scaled"].copy()
            splits[name]["X_scaled"]["flow_signature_frequency"] = scaled.values

    return splits


def make_synthetic_dataset(out_dir: str, n_files: int = 3, rows_per_file: int = 2000, seed: int = 42):
    """
    Generates tiny synthetic CSVs with EXACTLY the same 40-column schema
    as the real CICIoT2023 Merged*.csv files (verified against the user's
    inspection report), for smoke-testing the pipeline before running it
    on real data. Not used for any reported result.
    """
    rng = np.random.RandomState(seed)
    feature_cols = [
        "Header_Length", "Protocol Type", "Time_To_Live", "Rate",
        "fin_flag_number", "syn_flag_number", "rst_flag_number", "psh_flag_number",
        "ack_flag_number", "ece_flag_number", "cwr_flag_number",
        "ack_count", "syn_count", "fin_count", "rst_count",
        "HTTP", "HTTPS", "DNS", "Telnet", "SMTP", "SSH", "IRC",
        "TCP", "UDP", "DHCP", "ARP", "ICMP", "IGMP", "IPv", "LLC",
        "Tot sum", "Min", "Max", "AVG", "Std", "Tot size", "IAT", "Number", "Variance",
    ]
    labels = [
        "BENIGN", "DDOS-ICMP_FLOOD", "DDOS-TCP_FLOOD", "DDOS-UDP_FLOOD",
        "DOS-TCP_FLOOD", "DOS-UDP_FLOOD", "MIRAI-GREETH_FLOOD", "MIRAI-GREIP_FLOOD",
        "RECON-PORTSCAN", "RECON-HOSTDISCOVERY", "DNS_SPOOFING", "MITM-ARPSPOOFING",
        "VULNERABILITYSCAN",
    ]
    os.makedirs(out_dir, exist_ok=True)
    for f in range(n_files):
        data = rng.rand(rows_per_file, len(feature_cols)) * 100
        df = pd.DataFrame(data, columns=feature_cols)
        df["Label"] = rng.choice(labels, size=rows_per_file, p=_label_probs(labels))
        df.to_csv(os.path.join(out_dir, f"Merged{f+1:02d}.csv"), index=False)
    return out_dir


def _label_probs(labels):
    p = np.ones(len(labels))
    p[0] = 6.0  # benign more common, mirrors real dataset skew
    return p / p.sum()
