"""
Central configuration for the SAFE-Net v2 pipeline.
NOTHING in this file is a hardcoded personal path — everything is
either a sensible relative default or overridable via CLI flags.

"""
import argparse
import os
from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---- Paths (all relative by default) ----
    data_dir: str = "./data/merged_csv"       # folder containing Merged*.csv (or any *.csv)
    artifacts_dir: str = "./artifacts"         # everything gets written under here
    run_name: str = ""                         # auto-generated if empty

    # ---- Sampling (mirrors the original paper's 3:1 benign:attack design,
    #      but now fully scripted, seeded, and logged so it's reproducible
    #      and auditable — addresses reviewer complaint about opaque sampling) ----
    sample_size: int = 1_400_000
    benign_ratio: float = 0.75                 # 3:1 benign:attack, same as original paper
    min_rows_per_attack_class: int = 500        # floor so rare attack types aren't erased

    # ---- Splits ----
    test_size: float = 0.30
    val_size: float = 0.50                      # fraction of the 30% temp split
    seed: int = 42

    # ---- Model (Option B: no LSTM, no windowing — see NOTES_ON_REDESIGN.md) ----
    hidden: int = 128
    latent: int = 32
    epochs_ae: int = 50
    epochs_clf: int = 30
    patience_ae: int = 7
    patience_clf: int = 7
    lr_ae: float = 1e-3
    lr_clf: float = 1e-3
    batch_size: int = 256
    dropout: float = 0.3

    # ---- Multi-seed evaluation & statistical significance ----
    n_seeds: int = 3
    seed_list: List[int] = field(default_factory=lambda: [42, 43, 44])

    # ---- Hard-class targeted improvement ----
    hard_class_recall_threshold: float = 0.70   # classes below this (on val set) are "hard"
    hard_class_oversample_weight: float = 3.0   # loss/sampling weight multiplier for hard-class rows

    # ---- Generalization (leave-one-attack-out) ----
    loao_attack_subset: List[str] = field(default_factory=lambda: [
        "DDoS-ICMP_FLOOD", "DOS-TCP_FLOOD", "MIRAI-GREETH_FLOOD",
        "RECON-PORTSCAN", "DNS_SPOOFING",
    ])

    # ---- Multiclass grouping (kept small so confusion matrix stays legible) ----
    # Maps raw CICIoT2023 labels -> coarse category, used only for the
    # multiclass figure/table. Full 33-class recall is still reported
    # as a plain table (fig_per_attack_recall.png), just not as a heatmap.
    category_map: dict = field(default_factory=lambda: {
        "BENIGN": "Benign",
    })  # extended programmatically at runtime for any label containing keywords

    # ---- Misc ----
    device: str = "auto"                        # "auto" | "cpu" | "cuda"
    quick: bool = False                          # smoke-test mode: tiny synthetic run
    n_jobs: int = -1                             # for sklearn baselines


def build_config_from_cli() -> Config:
    p = argparse.ArgumentParser(description="SAFE-Net v2 pipeline configuration")
    cfg = Config()
    p.add_argument("--data_dir", type=str, default=cfg.data_dir)
    p.add_argument("--artifacts_dir", type=str, default=cfg.artifacts_dir)
    p.add_argument("--run_name", type=str, default=cfg.run_name)
    p.add_argument("--sample_size", type=int, default=cfg.sample_size)
    p.add_argument("--benign_ratio", type=float, default=cfg.benign_ratio)
    p.add_argument("--test_size", type=float, default=cfg.test_size)
    p.add_argument("--val_size", type=float, default=cfg.val_size)
    p.add_argument("--seed", type=int, default=cfg.seed)
    p.add_argument("--hidden", type=int, default=cfg.hidden)
    p.add_argument("--latent", type=int, default=cfg.latent)
    p.add_argument("--epochs_ae", type=int, default=cfg.epochs_ae)
    p.add_argument("--epochs_clf", type=int, default=cfg.epochs_clf)
    p.add_argument("--batch_size", type=int, default=cfg.batch_size)
    p.add_argument("--device", type=str, default=cfg.device)
    p.add_argument("--n_seeds", type=int, default=cfg.n_seeds)
    p.add_argument("--hard_class_recall_threshold", type=float, default=cfg.hard_class_recall_threshold)
    p.add_argument("--hard_class_oversample_weight", type=float, default=cfg.hard_class_oversample_weight)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    for k, v in vars(args).items():
        setattr(cfg, k, v)

    if cfg.quick:
        # Tiny synthetic smoke-test settings — used to validate the pipeline
        # runs end-to-end before committing hours of compute on the real data.
        cfg.sample_size = 4000
        cfg.epochs_ae = 3
        cfg.epochs_clf = 3
        cfg.batch_size = 64
        cfg.loao_attack_subset = cfg.loao_attack_subset[:2]
        cfg.n_seeds = 2

    cfg.seed_list = [42 + i for i in range(cfg.n_seeds)]

    if not cfg.run_name:
        import datetime
        cfg.run_name = "run_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(os.path.join(cfg.artifacts_dir, cfg.run_name), exist_ok=True)
    return cfg


def run_dir(cfg: Config) -> str:
    return os.path.join(cfg.artifacts_dir, cfg.run_name)
