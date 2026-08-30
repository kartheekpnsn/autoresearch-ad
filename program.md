# autoresearch

This is an experiment to have the LLM do its own tabular deep-learning research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist; this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md`: repository context.
   - `train.py`: the file you modify. Model architecture, optimizer, training loop, and metric.
   - `data/master.csv`: the structured tabular dataset.
4. **Verify data exists**: Check that `data/master.csv` exists and contains the target column `num`.
5. **Initialize results.csv**: Create `results.csv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment trains a PyTorch deep neural network for binary classification on structured tabular data. The data lives in `data/master.csv`, and the target column is `num`. Treat `num == 0` as class 0 and `num > 0` as class 1.

The computer is Mac OS. Prefer Apple GPU acceleration through PyTorch MPS when available, with CPU as a fallback. You launch an experiment simply as:

```bash
uv run train.py
```

**What you CAN do:**
- Modify `train.py`. Everything is fair game inside that file: model architecture, optimizer, learning rate, batch size, dropout, class weighting, feature handling, threshold selection, validation split, training loop, etc.
- Change the architecture or hyperparameters and do a simple run for each experiment.

**What you CANNOT do:**
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Change the task, data path, target column, or metric definition.

**The goal is simple: maximize validation FBeta score with beta=2.** Everything is fair game inside the training script as long as it is still a PyTorch deep neural network for tabular binary classification on `data/master.csv`.

**Memory** is a soft constraint. Prefer simple models that fit comfortably on a Mac using MPS. Some increase is acceptable for meaningful FBeta gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Output format

Once the script finishes it prints epoch logs and then a CSV summary like this:

```csv
---
data_path,target,device,fbeta_beta,val_fbeta,best_threshold,best_epoch,training_seconds,num_features,num_params,config_hidden_dims,config_dropout,config_batch_size,config_epochs,config_learning_rate,config_weight_decay,config_val_fraction,config_beta
data/master.csv,num,mps,2.0,0.812345,0.35,80,4.2,10,12033,"(128, 64, 32)",0.2,64,120,0.001,0.0001,0.2,2.0
```

You can extract the final metric from the log file:

```bash
tail -n 1 run.log
```

## Logging results

When an experiment is done, log it to `results.csv`.

The CSV has a header row and 5 columns:

```csv
commit,val_fbeta,status,description,notes
```

1. git commit hash (short, 7 chars)
2. validation FBeta score with beta=2 (e.g. 0.812345); use 0.000000 for crashes
3. status: `keep`, `discard`, or `crash`
4. short text description of what this experiment tried
5. optional notes, such as best threshold, epoch, or failure reason

Example:

```csv
commit,val_fbeta,status,description,notes
a1b2c3d,0.812345,keep,baseline,"threshold=0.35 epoch=80"
b2c3d4e,0.830100,keep,wider hidden layers,"threshold=0.40 epoch=62"
c3d4e5f,0.801000,discard,higher dropout,"threshold=0.25 epoch=90"
d4e5f6g,0.000000,crash,too high learning rate,"loss became NaN"
```

## The Experiment Loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-mps`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on.
2. Tune `train.py` with an experimental idea by directly hacking the code.
3. git commit.
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything; do not use tee or let output flood your context).
5. Read out the result CSV row: `tail -n 1 run.log`.
6. If the CSV row is missing or malformed, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in `results.csv` (do not commit `results.csv`; leave it untracked by git).
8. If `val_fbeta` improved (higher), you "advance" the branch, keeping the git commit.
9. If `val_fbeta` is equal or worse, you git reset back to where you started.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. You are advancing the branch so that you can iterate.

**Timeout**: Each experiment should be a simple run and complete quickly on Mac OS. If a run exceeds 10 minutes, kill it and treat it as a failure.

**Crashes**: If a run crashes, use your judgment. If it's something dumb and easy to fix, fix it and re-run. If the idea itself is fundamentally broken, log `crash` and move on.

**NEVER STOP**: Once the experiment loop has begun, do not pause to ask the human if you should continue. Do not ask "should I keep going?" or "is this a good stopping point?". The human expects you to continue working until manually stopped.
