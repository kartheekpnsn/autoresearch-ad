"""Train a variational autoencoder for multivariate anomaly detection.

Usage:
    conda run -n persistent_env python train.py
    conda run -n persistent_env python train.py --evaluate-test

The VAE learns from clean causal windows. Validation labels select an anomaly
threshold by F2 score; test labels are used only when explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "data/gen_data_29aug2026.csv"
TARGET_COLUMN = "is_injected"
TIME_COLUMN = "timestamp"
TRAIN_END = 550
VAL_END = 825
SEED = 42
EXIT_SUCCESS = 0


@dataclass(frozen=True)
class Config:
    """VAE architecture, optimization, and temporal feature configuration."""

    hidden_dims: tuple[int, ...] = (96, 48)
    latent_dim: int = 8
    dropout: float = 0.05
    sequence_length: int = 24
    batch_size: int = 64
    epochs: int = 120
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    kl_weight: float = 0.01
    patience: int = 20
    beta: float = 2.0


@dataclass(frozen=True)
class WindowedData:
    """Causal windows and metadata aligned to each window endpoint."""

    values: np.ndarray
    labels: np.ndarray
    history_has_anomaly: np.ndarray
    endpoint_indices: np.ndarray


@dataclass(frozen=True)
class DataSplits:
    """Scaled train, validation, and test arrays."""

    train: np.ndarray
    val: np.ndarray
    val_labels: np.ndarray
    test: np.ndarray
    test_labels: np.ndarray
    input_dim: int
    signal_count: int


@dataclass(frozen=True)
class DetectionMetrics:
    """Pointwise and event-level anomaly detection metrics."""

    fbeta: float
    precision: float
    recall: float
    false_positives: int
    captured_points: int
    anomaly_points: int
    captured_events: int
    anomaly_events: int
    event_recall: float


class VariationalAutoencoder(nn.Module):
    """Fully connected VAE for flattened multivariate time windows."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...],
        latent_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        encoder_layers: list[nn.Module] = []
        previous_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            previous_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.mean = nn.Linear(previous_dim, latent_dim)
        self.log_variance = nn.Linear(previous_dim, latent_dim)

        decoder_layers: list[nn.Module] = []
        previous_dim = latent_dim
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [nn.Linear(previous_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            )
            previous_dim = hidden_dim
        decoder_layers.append(nn.Linear(previous_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reconstruct values through a sampled latent representation."""
        encoded = self.encoder(values)
        mean = self.mean(encoded)
        log_variance = self.log_variance(encoded).clamp(-12.0, 12.0)
        standard_deviation = torch.exp(0.5 * log_variance)
        latent = mean + torch.randn_like(standard_deviation) * standard_deviation
        return self.decoder(latent), mean, log_variance

    def reconstruct(self, values: torch.Tensor) -> torch.Tensor:
        """Reconstruct values deterministically from the latent mean."""
        return self.decoder(self.mean(self.encoder(values)))


def create_parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="evaluate the locked validation threshold on the test split",
    )
    return parser


def pick_device() -> torch.device:
    """Select MPS, CUDA, or CPU in preference order."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _parse_binary_labels(values: pd.Series) -> np.ndarray:
    """Parse strict boolean or zero/one labels."""
    normalized = values.astype("string").str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False}
    labels = normalized.map(mapping)
    if labels.isna().any():
        invalid = sorted(normalized[labels.isna()].dropna().unique().tolist())
        raise ValueError(
            f"Target column {TARGET_COLUMN!r} must contain only boolean or 0/1 "
            f"values; found {invalid}"
        )
    return labels.to_numpy(dtype=bool)


def load_time_series(path: Path) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Load and validate a timestamped numeric anomaly-detection dataset."""
    if not path.is_file():
        raise FileNotFoundError(f"Could not find data file: {path}")

    frame = pd.read_csv(path)
    frame.columns = [column.strip() for column in frame.columns]
    required = {TIME_COLUMN, TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    timestamps = pd.to_datetime(frame[TIME_COLUMN], errors="coerce")
    if timestamps.isna().any():
        raise ValueError(f"Column {TIME_COLUMN!r} contains invalid timestamps")
    if timestamps.duplicated().any():
        raise ValueError(f"Column {TIME_COLUMN!r} contains duplicate timestamps")

    frame = frame.assign(**{TIME_COLUMN: timestamps}).sort_values(TIME_COLUMN)
    labels = _parse_binary_labels(frame[TARGET_COLUMN])
    candidate_columns = [
        column for column in frame.columns if column not in {TIME_COLUMN, TARGET_COLUMN}
    ]
    numeric = frame[candidate_columns].apply(pd.to_numeric, errors="coerce")
    signal_columns = tuple(
        column for column in numeric.columns if numeric[column].notna().any()
    )
    if not signal_columns:
        raise ValueError("Dataset must contain at least one numeric signal column")

    values = numeric.loc[:, signal_columns].to_numpy(dtype=np.float32, copy=True)
    values[~np.isfinite(values)] = np.nan
    return values, labels, signal_columns


def build_causal_windows(
    values: np.ndarray, labels: np.ndarray, sequence_length: int
) -> WindowedData:
    """Flatten trailing windows and preserve endpoint and history labels."""
    if sequence_length < 2:
        raise ValueError("sequence_length must be at least 2")
    if len(values) != len(labels):
        raise ValueError("values and labels must contain the same number of rows")
    if len(values) < sequence_length:
        raise ValueError("dataset is shorter than sequence_length")

    windows = np.lib.stride_tricks.sliding_window_view(
        values, window_shape=sequence_length, axis=0
    ).transpose(0, 2, 1)
    flattened = windows.reshape(len(windows), -1).astype(np.float32, copy=True)
    label_windows = np.lib.stride_tricks.sliding_window_view(labels, sequence_length)
    endpoint_indices = np.arange(sequence_length - 1, len(values))
    return WindowedData(
        values=flattened,
        labels=labels[endpoint_indices],
        history_has_anomaly=label_windows.any(axis=1),
        endpoint_indices=endpoint_indices,
    )


def _assert_event_safe_boundaries(labels: np.ndarray, boundaries: tuple[int, ...]) -> None:
    """Reject split boundaries that divide a contiguous labeled event."""
    for boundary in boundaries:
        if boundary <= 0 or boundary >= len(labels):
            raise ValueError(f"Split boundary {boundary} is outside the dataset")
        if labels[boundary - 1] and labels[boundary]:
            raise ValueError(f"Split boundary {boundary} divides an anomaly event")


def prepare_splits(values: np.ndarray, labels: np.ndarray, config: Config) -> DataSplits:
    """Create chronological splits with train-only imputation and scaling."""
    if len(values) <= VAL_END:
        raise ValueError(f"Dataset must contain more than {VAL_END} rows")
    _assert_event_safe_boundaries(labels, (TRAIN_END, VAL_END))
    windowed = build_causal_windows(values, labels, config.sequence_length)

    train_mask = (windowed.endpoint_indices < TRAIN_END) & ~windowed.history_has_anomaly
    val_mask = (windowed.endpoint_indices >= TRAIN_END) & (
        windowed.endpoint_indices < VAL_END
    )
    test_mask = windowed.endpoint_indices >= VAL_END
    train = windowed.values[train_mask]
    val = windowed.values[val_mask]
    test = windowed.values[test_mask]
    if not len(train) or not len(val) or not len(test):
        raise ValueError("Chronological split produced an empty partition")

    fill_values = np.nanmedian(train, axis=0)
    if np.isnan(fill_values).any():
        raise ValueError("At least one feature is missing in every clean training window")
    train = np.where(np.isnan(train), fill_values, train)
    val = np.where(np.isnan(val), fill_values, val)
    test = np.where(np.isnan(test), fill_values, test)
    mean = train.mean(axis=0, keepdims=True)
    standard_deviation = train.std(axis=0, keepdims=True)
    standard_deviation = np.where(standard_deviation < 1e-6, 1.0, standard_deviation)

    return DataSplits(
        train=((train - mean) / standard_deviation).astype(np.float32),
        val=((val - mean) / standard_deviation).astype(np.float32),
        val_labels=windowed.labels[val_mask],
        test=((test - mean) / standard_deviation).astype(np.float32),
        test_labels=windowed.labels[test_mask],
        input_dim=windowed.values.shape[1],
        signal_count=values.shape[1],
    )


def vae_loss(
    reconstructed: torch.Tensor,
    values: torch.Tensor,
    mean: torch.Tensor,
    log_variance: torch.Tensor,
    kl_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, reconstruction, and KL losses."""
    reconstruction_loss = F.mse_loss(reconstructed, values)
    kl_loss = -0.5 * torch.mean(
        1 + log_variance - mean.square() - log_variance.exp()
    )
    return reconstruction_loss + kl_weight * kl_loss, reconstruction_loss, kl_loss


@torch.no_grad()
def reconstruction_scores(
    model: VariationalAutoencoder, values: np.ndarray, device: torch.device
) -> np.ndarray:
    """Calculate deterministic per-window reconstruction errors."""
    model.eval()
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    reconstructed = model.reconstruct(tensor)
    return torch.mean((tensor - reconstructed).square(), dim=1).cpu().numpy()


def detection_metrics(
    labels: np.ndarray, predictions: np.ndarray, beta: float
) -> DetectionMetrics:
    """Calculate pointwise F-beta and contiguous-event capture metrics."""
    labels = np.asarray(labels, dtype=bool)
    predictions = np.asarray(predictions, dtype=bool)
    true_positives = int(np.sum(labels & predictions))
    false_positives = int(np.sum(~labels & predictions))
    false_negatives = int(np.sum(labels & ~predictions))
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    beta_squared = beta * beta
    denominator = beta_squared * precision + recall
    fbeta = (
        (1 + beta_squared) * precision * recall / denominator
        if denominator
        else 0.0
    )

    event_starts = np.flatnonzero(labels & ~np.r_[False, labels[:-1]])
    event_ends = np.flatnonzero(labels & ~np.r_[labels[1:], False]) + 1
    captured_events = sum(
        bool(predictions[start:end].any())
        for start, end in zip(event_starts, event_ends, strict=True)
    )
    anomaly_events = len(event_starts)
    return DetectionMetrics(
        fbeta=fbeta,
        precision=precision,
        recall=recall,
        false_positives=false_positives,
        captured_points=true_positives,
        anomaly_points=int(labels.sum()),
        captured_events=captured_events,
        anomaly_events=anomaly_events,
        event_recall=captured_events / max(anomaly_events, 1),
    )


def select_threshold(
    scores: np.ndarray, labels: np.ndarray, beta: float
) -> tuple[float, DetectionMetrics]:
    """Select the validation threshold maximizing F-beta and anomaly recall."""
    if not len(scores) or len(scores) != len(labels):
        raise ValueError("scores and labels must be non-empty and equally sized")
    if not np.asarray(labels, dtype=bool).any():
        raise ValueError("validation labels must contain at least one anomaly")

    best_threshold = float(np.max(scores))
    best_metrics = detection_metrics(labels, scores >= best_threshold, beta)
    best_key = (
        best_metrics.fbeta,
        best_metrics.recall,
        -best_metrics.false_positives,
        best_threshold,
    )
    for threshold in np.unique(scores):
        metrics = detection_metrics(labels, scores >= threshold, beta)
        key = (metrics.fbeta, metrics.recall, -metrics.false_positives, float(threshold))
        if key > best_key:
            best_threshold = float(threshold)
            best_metrics = metrics
            best_key = key
    return best_threshold, best_metrics


def train_model(
    model: VariationalAutoencoder,
    splits: DataSplits,
    config: Config,
    device: torch.device,
) -> tuple[int, float]:
    """Train on clean windows and restore the best normal-validation checkpoint."""
    train_dataset = TensorDataset(torch.tensor(splits.train, dtype=torch.float32))
    generator = torch.Generator().manual_seed(SEED)
    loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    normal_val = splits.val[~splits.val_labels]
    normal_val_tensor = torch.tensor(normal_val, dtype=torch.float32, device=device)
    best_state = deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        seen = 0
        for (batch,) in loader:
            batch = batch.to(device)
            reconstructed, mean, log_variance = model(batch)
            loss, _, _ = vae_loss(
                reconstructed, batch, mean, log_variance, config.kl_weight
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(batch)
            seen += len(batch)

        model.eval()
        with torch.no_grad():
            val_reconstructed = model.reconstruct(normal_val_tensor)
            val_loss = F.mse_loss(val_reconstructed, normal_val_tensor).item()
        train_loss = train_loss_sum / max(seen, 1)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 20 == 0 or epoch == config.epochs:
            print(
                f"epoch {epoch:03d} | train_loss: {train_loss:.4f} | "
                f"normal_val_reconstruction: {val_loss:.4f}",
                flush=True,
            )
        if epochs_without_improvement >= config.patience:
            break

    model.load_state_dict(best_state)
    return best_epoch, best_val_loss


def _metric_fields(prefix: str, metrics: DetectionMetrics) -> dict[str, float | int]:
    """Convert detection metrics to prefixed CSV fields."""
    return {
        f"{prefix}_fbeta": round(metrics.fbeta, 6),
        f"{prefix}_precision": round(metrics.precision, 6),
        f"{prefix}_recall": round(metrics.recall, 6),
        f"{prefix}_false_positives": metrics.false_positives,
        f"{prefix}_captured_points": metrics.captured_points,
        f"{prefix}_anomaly_points": metrics.anomaly_points,
        f"{prefix}_captured_events": metrics.captured_events,
        f"{prefix}_anomaly_events": metrics.anomaly_events,
        f"{prefix}_event_recall": round(metrics.event_recall, 6),
    }


def train(*, evaluate_test: bool = False) -> dict[str, float | int | str]:
    """Train the VAE and return a stable experiment summary."""
    config = Config()
    set_seed(SEED)
    device = pick_device()
    values, labels, signal_columns = load_time_series(DATA_PATH)
    splits = prepare_splits(values, labels, config)
    model = VariationalAutoencoder(
        splits.input_dim, config.hidden_dims, config.latent_dim, config.dropout
    ).to(device)

    started = time.time()
    best_epoch, best_val_loss = train_model(model, splits, config, device)
    val_scores = reconstruction_scores(model, splits.val, device)
    threshold, val_metrics = select_threshold(val_scores, splits.val_labels, config.beta)
    elapsed = time.time() - started

    summary: dict[str, float | int | str] = {
        "data_path": "data/gen_data_29aug2026.csv",
        "target": TARGET_COLUMN,
        "device": device.type,
        "fbeta_beta": config.beta,
        **_metric_fields("val", val_metrics),
        "best_threshold": round(threshold, 6),
        "best_epoch": best_epoch,
        "best_normal_val_loss": round(best_val_loss, 6),
        "training_seconds": round(elapsed, 3),
        "num_rows": len(values),
        "num_signals": len(signal_columns),
        "num_features": splits.input_dim,
        "num_train_windows": len(splits.train),
        "num_params": sum(parameter.numel() for parameter in model.parameters()),
        **{f"config_{key}": value for key, value in asdict(config).items()},
    }
    test_fields = [
        "test_fbeta",
        "test_precision",
        "test_recall",
        "test_false_positives",
        "test_captured_points",
        "test_anomaly_points",
        "test_captured_events",
        "test_anomaly_events",
        "test_event_recall",
    ]
    summary.update({field: "" for field in test_fields})
    if evaluate_test:
        test_scores = reconstruction_scores(model, splits.test, device)
        test_metrics = detection_metrics(
            splits.test_labels, test_scores >= threshold, config.beta
        )
        summary.update(_metric_fields("test", test_metrics))
    return summary


def print_csv_summary(metrics: dict[str, float | int | str]) -> None:
    """Print metrics as a two-line CSV block for experiment automation."""
    print("---")
    writer = csv.DictWriter(sys.stdout, fieldnames=list(metrics), lineterminator="\n")
    writer.writeheader()
    writer.writerow(metrics)


def main() -> int:
    """Run training and preserve tracebacks for experiment failures."""
    args = create_parser().parse_args()
    try:
        print_csv_summary(train(evaluate_test=args.evaluate_test))
    except KeyboardInterrupt:
        print("Interrupted by user", file=sys.stderr)
        return 130
    return EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())