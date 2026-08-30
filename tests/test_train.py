"""Tests for VAE anomaly-detection preparation and evaluation."""

import numpy as np
import pytest
import torch

from train import (
    Config,
    VariationalAutoencoder,
    build_causal_windows,
    detection_metrics,
    prepare_splits,
    select_threshold,
)


def test_given_multivariate_rows_when_windowed_then_order_is_causal():
    # Arrange
    values = np.arange(10, dtype=np.float32).reshape(5, 2)
    labels = np.array([False, True, False, False, False])

    # Act
    windowed = build_causal_windows(values, labels, sequence_length=3)

    # Assert
    assert windowed.values[0].tolist() == [0, 1, 2, 3, 4, 5]
    assert windowed.endpoint_indices.tolist() == [2, 3, 4]


def test_given_anomaly_in_window_history_when_windowed_then_history_is_flagged():
    # Arrange
    values = np.arange(6, dtype=np.float32).reshape(6, 1)
    labels = np.array([False, True, False, False, False, False])

    # Act
    windowed = build_causal_windows(values, labels, sequence_length=3)

    # Assert
    assert windowed.history_has_anomaly.tolist() == [True, True, False, False]


def test_given_chronological_data_when_prepared_then_uses_fixed_splits_and_clean_train():
    # Arrange
    rows = 900
    values = np.column_stack(
        [np.arange(rows, dtype=np.float32), np.arange(rows, dtype=np.float32) * 2]
    )
    labels = np.zeros(rows, dtype=bool)
    labels[100] = True
    labels[560:590] = True
    labels[850:900] = True
    config = Config(sequence_length=3)

    # Act
    splits = prepare_splits(values, labels, config)

    # Assert
    assert len(splits.train) == 545
    assert len(splits.val) == 275
    assert len(splits.test) == 75
    assert splits.val_labels.sum() == 30
    assert splits.test_labels.sum() == 50
    assert splits.train.mean(axis=0) == pytest.approx(0.0, abs=1e-5)


def test_given_split_boundary_inside_event_when_prepared_then_raises():
    # Arrange
    values = np.ones((900, 1), dtype=np.float32)
    labels = np.zeros(900, dtype=bool)
    labels[549:551] = True

    # Act & Assert
    with pytest.raises(ValueError, match="divides an anomaly event"):
        prepare_splits(values, labels, Config(sequence_length=3))


def test_given_scores_when_threshold_selected_then_f2_is_maximized():
    # Arrange
    scores = np.array([0.1, 0.9, 0.8, 0.2], dtype=np.float32)
    labels = np.array([False, True, True, False])

    # Act
    threshold, metrics = select_threshold(scores, labels, beta=2.0)

    # Assert
    assert threshold == pytest.approx(0.8)
    assert metrics.fbeta == 1.0
    assert metrics.recall == 1.0


def test_given_contiguous_anomalies_when_scored_then_events_are_counted_once():
    # Arrange
    labels = np.array([False, True, True, False, True, True, True, False])
    predictions = np.array([False, False, True, False, False, False, False, True])

    # Act
    metrics = detection_metrics(labels, predictions, beta=2.0)

    # Assert
    assert metrics.captured_points == 1
    assert metrics.captured_events == 1
    assert metrics.anomaly_events == 2
    assert metrics.event_recall == 0.5


def test_given_batch_when_vae_runs_then_outputs_have_expected_shapes():
    # Arrange
    model = VariationalAutoencoder(
        input_dim=12, hidden_dims=(8, 4), latent_dim=3, dropout=0.0
    )
    values = torch.randn(5, 12)

    # Act
    reconstructed, mean, log_variance = model(values)
    deterministic = model.reconstruct(values)

    # Assert
    assert reconstructed.shape == values.shape
    assert deterministic.shape == values.shape
    assert mean.shape == (5, 3)
    assert log_variance.shape == (5, 3)