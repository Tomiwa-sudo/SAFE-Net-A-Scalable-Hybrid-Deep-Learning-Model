"""
Model architectures.

MAIN MODEL (Option B redesign): a plain dense Autoencoder + a fusion MLP
classifier that operates on ONE FLOW ROW AT A TIME. No LSTM, no sliding
windows, no seq_len. This directly fixes:
  - the shuffle-before-windowing bug (there's no windowing left to shuffle)
  - the "sequences aren't temporal" methodology flaw (CICIoT2023 rows carry
    no timestamp; treating rows as i.i.d. flow records is now the honest
    framing, not an accidental one)
  - the latency-measurement confusion (one row in -> one prediction out,
    so "per packet" now means exactly what it says)

DEEP BASELINES: CNN1D, CNN+BiLSTM+Attention, and a small Transformer
encoder. These treat the 39 features of a single row as an ORDERED
FEATURE VECTOR (a position in feature-space, not a position in time) —
a standard, defensible technique in tabular deep learning, and clearly
different from claiming genuine temporal structure across rows.
"""
import torch
import torch.nn as nn


# ============================================================
# Main SAFE-Net v2 components
# ============================================================
class DenseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 128, latent: int = 32, dropout: float = 0.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class FusionClassifier(nn.Module):
    """Takes [latent_z ; raw_features] -> class logits. num_classes=2 for
    binary Benign/Attack, or set higher for the multiclass head."""
    def __init__(self, in_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# Deep learning baselines (row-level, feature-vector-as-sequence)
# ============================================================
class CNN1DBaseline(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(32),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(64),
            nn.AdaptiveAvgPool1d(8),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Linear(64 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, features) -- features treated as spatial axis
        return self.head(self.conv(x))


class CNNBiLSTMAttention(nn.Module):
    """A 'recent-style' hybrid baseline requested by reviewers: CNN feature
    extraction over the feature vector, a BiLSTM over the resulting CNN
    feature map (treated as a short generic sequence of learned feature
    groups, NOT as a claim about temporal order of packets), and additive
    attention pooling before classification."""
    def __init__(self, input_dim: int, num_classes: int = 2, cnn_channels: int = 32, lstm_hidden: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1), nn.ReLU(), nn.BatchNorm1d(cnn_channels),
        )
        self.lstm = nn.LSTM(cnn_channels, lstm_hidden, batch_first=True, bidirectional=True)
        self.attn = nn.Linear(lstm_hidden * 2, 1)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)              # (B, 1, F)
        c = self.conv(x)                # (B, C, F)
        c = c.permute(0, 2, 1)          # (B, F, C) -- F positions treated as sequence steps
        out, _ = self.lstm(c)           # (B, F, 2H)
        w = torch.softmax(self.attn(out), dim=1)   # attention weights over positions
        pooled = (w * out).sum(dim=1)   # (B, 2H)
        return self.head(pooled)


class TinyTransformer(nn.Module):
    """A compact Transformer encoder baseline (the reviewer's 'transformer-
    based IDS' suggestion), operating over the feature vector with each
    scalar feature embedded as a token."""
    def __init__(self, input_dim: int, num_classes: int = 2, d_model: int = 32, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_dim = input_dim
        self.token_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, input_dim, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                                            batch_first=True, dropout=0.1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        tok = self.token_embed(x.unsqueeze(-1))  # (B, F, d_model)
        tok = tok + self.pos_embed
        enc = self.encoder(tok)                  # (B, F, d_model)
        pooled = enc.mean(dim=1)
        return self.head(pooled)


class PlainMLPBaseline(nn.Module):
    """Standalone supervised MLP with no autoencoder/unsupervised component
    at all -- used in the ablation study as the 'raw features only' arm."""
    def __init__(self, input_dim: int, num_classes: int = 2, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)
