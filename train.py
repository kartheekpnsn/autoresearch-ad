"""
Tabular binary classification training script.

Usage:
    uv run train.py

The script trains a PyTorch deep neural network on data/master.csv, where the
target column is "num". For binary classification, num == 0 maps to class 0 and
any non-zero value maps to class 1.
"""

from __future__ import annotations

import csv
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = Path("data/master.csv")
TARGET_COLUMN = "num"
SEED = 42


@dataclass
class Config:
    hidden_dims: tuple[int, ...] = (128, 64, 32)
    dropout: float = 0.20
    batch_size: int = 64
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    val_fraction: float = 0.20
    beta: float = 2.0


class TabularBinaryClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], dropout: float):
        super().__init__()
        layers: list[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def fbeta_score(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    beta: float = 2.0,
    eps: float = 1e-12,
) -> float:
    y_true = y_true.bool()
    y_pred = y_pred.bool()
    tp = (y_true & y_pred).sum().float()
    fp = (~y_true & y_pred).sum().float()
    fn = (y_true & ~y_pred).sum().float()
    beta_sq = beta * beta
    numerator = (1 + beta_sq) * tp
    denominator = numerator + beta_sq * fn + fp
    return (numerator / denominator.clamp_min(eps)).item()


def load_tabular_data(path: Path, target_column: str) -> tuple[pd.DataFrame, pd.Series]:
    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path}")

    df = pd.read_csv(path)
    df.columns = [col.strip() for col in df.columns]
    if target_column not in df.columns:
        raise ValueError(f"Target column {target_column!r} not found in {path}")

    target = pd.to_numeric(df.pop(target_column), errors="coerce")
    if target.isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values")

    features = df.copy()
    for column in features.columns:
        numeric = pd.to_numeric(features[column], errors="coerce")
        if numeric.notna().mean() >= 0.95:
            features[column] = numeric.fillna(numeric.median())
        else:
            features[column] = features[column].astype("string").fillna("missing")

    features = pd.get_dummies(features, dummy_na=False)
    labels = (target.astype(float) > 0).astype(np.float32)
    return features.astype(np.float32), labels


def stratified_split(
    y: np.ndarray,
    val_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for klass in np.unique(y):
        klass_indices = np.flatnonzero(y == klass)
        rng.shuffle(klass_indices)
        val_count = max(1, int(round(len(klass_indices) * val_fraction)))
        val_indices.extend(klass_indices[:val_count].tolist())
        train_indices.extend(klass_indices[val_count:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return np.array(train_indices), np.array(val_indices)


def standardize(
    x_train: np.ndarray,
    x_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (x_train - mean) / std, (x_val - mean) / std


def make_loaders(config: Config, device: torch.device) -> tuple[DataLoader, TensorDataset, int, float]:
    features, labels = load_tabular_data(DATA_PATH, TARGET_COLUMN)
    train_idx, val_idx = stratified_split(labels.to_numpy(), config.val_fraction, SEED)

    x = features.to_numpy(dtype=np.float32)
    y = labels.to_numpy(dtype=np.float32)
    x_train, x_val = standardize(x[train_idx], x[val_idx])
    y_train, y_val = y[train_idx], y[val_idx]

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    val_dataset = TensorDataset(
        torch.tensor(x_val, dtype=torch.float32, device=device),
        torch.tensor(y_val, dtype=torch.float32, device=device),
    )
    loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    positive_rate = float(y_train.mean())
    return loader, val_dataset, x.shape[1], positive_rate


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: TensorDataset,
    beta: float,
) -> tuple[float, float, float]:
    model.eval()
    x_val, y_val = dataset.tensors
    logits = model(x_val)
    loss = F.binary_cross_entropy_with_logits(logits, y_val).item()
    probs = torch.sigmoid(logits)

    best_score = -1.0
    best_threshold = 0.5
    for threshold in torch.linspace(0.05, 0.95, 19, device=probs.device):
        score = fbeta_score(y_val, probs >= threshold, beta=beta)
        if score > best_score:
            best_score = score
            best_threshold = threshold.item()

    return best_score, best_threshold, loss


def train() -> dict[str, float | int | str]:
    config = Config()
    set_seed(SEED)
    device = pick_device()

    train_loader, val_dataset, input_dim, positive_rate = make_loaders(config, device)
    model = TabularBinaryClassifier(input_dim, config.hidden_dims, config.dropout).to(device)

    pos_weight_value = (1.0 - positive_rate) / max(positive_rate, 1e-6)
    pos_weight = torch.tensor(pos_weight_value, dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    phase1_epochs = 40
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase1_epochs)
    phase2_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1, total_iters=config.epochs)

    best_fbeta = -1.0
    best_threshold = 0.5
    best_epoch = 0
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        seen = 0

        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            lam = float(np.random.beta(0.4, 0.4))
            idx = torch.randperm(x_batch.size(0), device=device)
            x_batch = lam * x_batch + (1 - lam) * x_batch[idx]
            y_batch = lam * y_batch + (1 - lam) * y_batch[idx]
            logits = model(x_batch)
            loss = F.binary_cross_entropy_with_logits(
                logits,
                y_batch,
                pos_weight=pos_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = x_batch.size(0)
            train_loss_sum += loss.item() * batch_size
            seen += batch_size

        if epoch <= phase1_epochs:
            scheduler.step()
        else:
            phase2_scheduler.step()

        val_fbeta, threshold, val_loss = evaluate(model, val_dataset, beta=config.beta)
        train_loss = train_loss_sum / max(seen, 1)
        if val_fbeta > best_fbeta:
            best_fbeta = val_fbeta
            best_threshold = threshold
            best_epoch = epoch

        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(
                f"epoch {epoch:03d} | train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | val_fbeta: {val_fbeta:.4f} | "
                f"threshold: {threshold:.2f}",
                flush=True,
            )

    elapsed = time.time() - started
    num_params = sum(p.numel() for p in model.parameters())
    return {
        "data_path": str(DATA_PATH),
        "target": TARGET_COLUMN,
        "device": device.type,
        "fbeta_beta": config.beta,
        "val_fbeta": round(best_fbeta, 6),
        "best_threshold": round(best_threshold, 4),
        "best_epoch": best_epoch,
        "training_seconds": round(elapsed, 3),
        "num_features": input_dim,
        "num_params": num_params,
        **{f"config_{key}": value for key, value in asdict(config).items()},
    }


def print_csv_summary(metrics: dict[str, float | int | str]) -> None:
    print("---")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(metrics), lineterminator="\n")
    writer.writeheader()
    writer.writerow(metrics)


if __name__ == "__main__":
    summary = train()
    print_csv_summary(summary)
