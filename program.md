---
title: VAE Anomaly Detection Autoresearch
description: Autonomous experiment protocol for tabular time-series anomaly detection
---

## Objective

Autonomously improve a PyTorch variational autoencoder (VAE) that detects
anomalies in multivariate time-series data. Optimize validation F2 while
preserving a locked chronological test holdout.

The immutable task contract is:

* Data file: `data/gen_data_29aug2026.csv`
* Timestamp column: `timestamp`
* Label column: `is_injected`
* Positive class: `is_injected == True`
* Model family: PyTorch VAE using reconstruction error as the anomaly score
* Primary selection metric: pointwise validation F2 (`beta=2`)
* Supporting metrics: anomaly recall, precision, false positives, captured
  anomaly points, and contiguous-event recall
* Runtime: Conda environment `persistent_env`
* Platform: macOS with MPS preferred and CPU fallback

Labels may exclude contaminated training windows and calibrate the validation
threshold. Labels must never be model inputs, feature values, reconstruction
targets, or part of the VAE loss.

## Setup

Before starting a new research run:

1. Propose a date-based run tag, such as `aug30`.
2. Confirm that `autoresearch/<tag>` does not already exist.
3. Create `autoresearch/<tag>` from the current default branch.
4. Read `README.md`, `train.py`, `program.md`, and `requirements.txt`.
5. Verify that the data file exists and contains `timestamp` and `is_injected`.
6. Verify the environment with
   `conda run -n persistent_env python -c "import numpy, pandas, torch"`.
7. Create `results.csv` with the header shown in the Results Log section.
8. Confirm setup with the user, then begin the experiment loop.

Do not install packages unless an existing dependency from `requirements.txt`
is missing in `persistent_env`.

## Data And Evaluation Contract

The baseline uses flattened causal windows containing the current timestamp and
the previous 23 hours across all numeric signals.

Window endpoints are split by original source-row index:

| Split | Endpoint rows | Purpose |
|-------|---------------|---------|
| Train | `< 550` | Learn normal behavior |
| Validation | `550` through `824` | Select checkpoints and anomaly threshold |
| Test | `>= 825` | Locked final evaluation |

These boundaries do not divide a contiguous injected-anomaly event. Keep them
fixed across all experiments so scores remain comparable.

The training pipeline must satisfy these rules:

* Exclude a training window when any timestamp in that window is labeled as an
  injected anomaly
* Fit imputation and scaling statistics on retained training windows only
* Fit VAE weights on retained training windows only
* Use only normal validation windows for reconstruction-based checkpoint
  selection
* Select the anomaly threshold on the full validation split by maximum F2
* Apply the locked validation threshold to test scores without retuning
* Treat contiguous `True` labels within a split as one anomaly event

The data currently contains 20,000 rows. The long test tail is intentional and
must not be truncated or sampled to improve results.

## Allowed Experiments

Modify only `train.py` during the autonomous experiment loop. Reasonable
experiments include:

* Encoder and decoder depth or width
* Latent dimension
* Activation functions, normalization, and dropout
* Causal temporal representation and sequence length
* Reconstruction loss formulation
* KL weighting or annealing
* Optimizer, learning rate, weight decay, batch size, and scheduler
* Early stopping and checkpoint-selection details
* Unsupervised signal transformations fitted on training data only

Keep the implementation a VAE whose anomaly score is derived from
reconstruction behavior. Prefer a simpler experiment when scores are equal.

## Forbidden Changes

Do not:

* Change the data path, timestamp column, target column, split boundaries,
  random seed, F2 definition, or primary selection metric
* Add a supervised prediction head or optimize model weights against labels
* Include labels or label-derived values in model inputs
* Fit preprocessing on validation or test data
* Tune thresholds, architecture, or hyperparameters against test metrics
* Truncate, resample, regenerate, or edit the source dataset
* Add dependencies or modify files other than `train.py`

## Baseline

The first run must establish the untouched baseline and may evaluate test once:

```bash
conda run -n persistent_env python train.py --evaluate-test > run.log 2>&1
tail -n 1 run.log
```

Record both validation and test metrics for the baseline. During model
selection, run without `--evaluate-test`; the stable CSV schema leaves test
fields blank.

## Experiment Loop

Repeat until manually stopped:

1. Inspect the current branch, commit, and working tree.
2. Form one testable hypothesis based on the current best result.
3. Make one focused change to `train.py`.
4. Commit the experiment.
5. Run the experiment:

   ```bash
   conda run -n persistent_env python train.py > run.log 2>&1
   ```

6. Read the final CSV row:

   ```bash
   tail -n 1 run.log
   ```

7. If the CSV row is missing or malformed, inspect the failure:

   ```bash
   tail -n 50 run.log
   ```

8. Append the experiment to `results.csv`.
9. Keep the commit only when `val_fbeta` strictly improves.
10. Reset the experiment commit when `val_fbeta` is equal or worse.

Never use test fields in the keep or discard decision. Do not run
`--evaluate-test` during intermediate experiments.

## Results Log

Keep `results.csv` uncommitted with these columns:

```csv
commit,val_fbeta,status,description,notes
```

Use a seven-character commit hash, `keep`, `discard`, or `crash`, a concise
experiment description, and useful validation details:

```csv
commit,val_fbeta,status,description,notes
a1b2c3d,0.812345,keep,baseline,"threshold=1.42 recall=0.84 events=2/2"
b2c3d4e,0.826100,keep,lower KL weight,"threshold=1.37 recall=0.87 events=2/2"
c3d4e5f,0.801000,discard,wider decoder,"threshold=1.51 recall=0.79 events=2/2"
d4e5f6g,0.000000,crash,higher learning rate,"loss became non-finite"
```

The script prints progress followed by a header and one CSV result row. Extract
the score from the `val_fbeta` column, not from a hard-coded column position.

## Final Evaluation

When experimentation is manually stopped, evaluate the retained winner once:

```bash
conda run -n persistent_env python train.py --evaluate-test > final.log 2>&1
tail -n 1 final.log
```

Report the best validation F2, validation recall, validation event capture,
locked test F2, test recall, test event capture, experiment count, and the main
change responsible for the improvement.

## Runtime And Failure Rules

* Allow at most 10 minutes per experiment
* Kill and record runs exceeding the limit as crashes
* Repair obvious implementation errors and rerun the same experiment
* Abandon fundamentally broken ideas after a few focused repair attempts
* Keep model size and memory practical for a Mac
* Continue autonomously until manually stopped; do not ask whether to continue