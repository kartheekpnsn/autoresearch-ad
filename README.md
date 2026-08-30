---
title: Autoresearch for VAE Anomaly Detection
description: Autonomous VAE experiments for multivariate time-series anomaly detection
---

## Overview

This repository supports autonomous PyTorch VAE experiments on a labeled,
multivariate time series. The VAE learns normal temporal behavior without using
labels in its loss. Validation labels calibrate a reconstruction-error
threshold and rank experiments by F2, which emphasizes anomaly recall.

## Task

* Data: `data/gen_data.csv`
* Timestamp: `timestamp`
* Anomaly label: `is_injected`
* Signals: six numeric value columns in the current dataset
* Model: variational autoencoder
* Anomaly score: deterministic reconstruction mean squared error
* Primary metric: validation F2 with `beta=2`
* Runtime: Conda environment `persistent_env`
* Device order: MPS, CUDA, then CPU

The script resolves the data path relative to `train.py`, so it works from any
current working directory.

## Data Preparation

The baseline converts each endpoint into a flattened, causal 24-hour window
across all numeric signals. It uses fixed chronological endpoint boundaries:

| Split | Source-row endpoints | Use |
|-------|----------------------|-----|
| Train | `< 550` | VAE fitting on clean windows |
| Validation | `550` through `824` | Checkpoint and threshold selection |
| Test | `>= 825` | Locked baseline and final evaluation |

A training window is removed when any timestamp in its history has
`is_injected == True`. Missing-value imputation and standardization are fitted
only on the remaining training windows.

The current source file contains 20,000 rows. The test split intentionally
retains the full chronological tail.

## Metrics

Validation reconstruction scores select the threshold that maximizes pointwise
F2. Ties favor higher anomaly recall, fewer false positives, and then the
higher threshold.

Each run reports:

* Pointwise F2, precision, and anomaly recall
* False-positive count
* Captured and total anomaly points
* Captured and total contiguous anomaly events
* Event recall

Test fields use the same CSV schema on every run. They remain blank unless
`--evaluate-test` is supplied. The selected validation threshold is applied to
test scores without retuning.

## Environment

Verify the existing environment:

```bash
conda run -n persistent_env python -c "import numpy, pandas, torch"
```

Install the repository requirements into that environment only when needed:

```bash
conda run -n persistent_env python -m pip install -r requirements.txt
```

## Run The Baseline

Use test evaluation for the initial baseline and retained final winner:

```bash
conda run -n persistent_env python train.py --evaluate-test
```

Use validation-only execution for intermediate autoresearch experiments:

```bash
conda run -n persistent_env python train.py
```

The final output is a CSV header and one result row. The summary includes
`val_fbeta`, `val_recall`, `val_event_recall`, `best_threshold`, model size,
runtime, configuration fields, and the optional test metrics.

## Autonomous Experiments

[program.md](program.md) defines the full research protocol. During the loop,
the agent modifies only `train.py`, commits one focused hypothesis, runs it,
and keeps the commit only when validation F2 strictly improves.

The fixed data path, labels, split boundaries, seed, F2 definition, and locked
test policy are part of the evaluation contract. Labels cannot be features or
training targets, and test metrics cannot influence experiment selection.

## Validate Changes

```bash
conda run -n persistent_env pytest -q tests/test_train.py
conda run -n persistent_env ruff check train.py tests/test_train.py
```

## Project Structure

```text
train.py          VAE training, scoring, and CSV experiment output
program.md        Autonomous research instructions and constraints
requirements.txt  Python dependency inventory
pytest.ini        Test import configuration
tests/            Focused preprocessing, metric, and model tests
```

## License

MIT