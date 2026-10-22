# c2 time lane: commands, environment, budget, runs

Design and predictions: [QUESTION.md](QUESTION.md). Observations: [NOTE.md](NOTE.md). Primary figure: [figures/time_main.png](figures/time_main.png), with `.svg` and the CSVs beside it.

## Files

| file | what it is |
|---|---|
| `plan.json`, `registry.jsonl` | every planned run, and append-only status events from `research/regimes/common/run_jobs.py` |
| `time_analyze.py` | `publish` copies the small run files and checkpoints listed with `--commit-checkpoints` into `runs/`, with a SHA-256 manifest of everything local. `check` covers on-host continuation, the join at 10k, and recipe against source, written to `checks.json`. `measure` writes the CSVs in `figures/`. `figure` draws `time_main` from them. |
| `test_time.py` | threshold, censoring and error-compatibility counts on hand-made inputs; neighbour label sets against `c2task` |
| `checks.json` | output of `check` |
| `runs/<ID>/` | `config.json`, `metrics.jsonl`, `runtime.json`, `stdout.log` and `MANIFEST.json`, plus committed checkpoints |

Shared code is `../common/` at pin `b3e8132`, unedited.

## Commands (exact)

```bash
export PATH=~/.local/bin:$PATH   # uv 0.12.23
uv sync --locked --group analysis
uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py    # own process
uv run --locked python -m unittest -v research/composition-c2/time/test_time.py    # own process
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/time/plan.json --root runs/c2 --registry research/composition-c2/time/registry.jsonl --session-label c2-time-cloud-1"
$R --workers 2 --ids c2-time-verify-tt-inter-s51200-a1 c2-time-verify-tu-inter-s51200-a1 c2-time-verify-tt-final-s51200-a1
$R --workers 2 --ids c2-time-tu-inter-s51200-a1 c2-time-tt-final-s51200-a1 c2-time-tu-inter-s51201-a1 c2-time-tt-final-s51201-a1 \
    c2-time-tu-inter-s51202-a1 c2-time-tt-final-s51202-a1 c2-time-tt-inter-s51200-a1 c2-time-tt-inter-s51201-a1 c2-time-tt-inter-s51202-a1
A="uv run --locked python research/composition-c2/time/time_analyze.py"
$A publish --root runs/c2 --ids <IDs> --commit-checkpoints <ID>/ckpt-20000.npz ... <ID>/ckpt-50000.npz
$A check && $A measure && $A figure
```

## Environment and calibration

- **Host.** `vm`, a cloud container: 4 visible CPUs, Intel Xeon @ 2.10 GHz, CPython 3.12, NumPy 2.3.5, one BLAS thread per job (`run_jobs.py` sets the thread variables and the malloc thresholds). Each run's `config.json` → `provenance` holds the full inventory.
- **Calibration and on-host continuation check (M1).** `c2-time-verify-{tt-inter,tu-inter,tt-final}-s51200-a1` resume the recovered `ckpt-1000` of seed 51200 to 3,000 updates with `--diag`, two jobs at a time.
  - Every r1 key matches the published record at all 41 logged steps (`checks.json`).
  - An update took 4.06, 4.09 and 3.72 ms. Peak RSS was 43–44 MiB.
  - These three runs used c2train's defaults for the recipe. Their `config.json` records show that the defaults equal the source runs' recipes (`checks.json` → `recipe_differs` is empty).
- **Timeouts.** 600 s for each 40,000-update continuation; the calibration predicts about 170 s.
- **Recipe flags.** After the reviewer's pin concern C1 (`--resume` does not check the recipe flags), every continuation entry in `plan.json` spells out `--task-id`, `--transition`, `--readout`, `--step-weights`, `--hidden`, `--lr`, `--wd` and `--eval-every`. `check` compares each run's recorded recipe with its source run's `config.json`.

## Budget tally (elapsed wall time with at least one job running)

| block | runs | wall |
|---|---|---|
| calibration and on-host continuation check (2 concurrent) | 3 | 16 s |
| nine continuations, 10k→50k (2 concurrent), 04:11:36–04:23:33Z | 9 | 717 s |
| **total so far** | 12 | **about 12.2 of 40 min** |

The nine took 135–158 s each, with peak RSS 44 MiB, against the 600 s timeout.

## Run manifest

None failed or was capped (`registry.jsonl`). Each `runs/<ID>/` holds `config.json`, `metrics.jsonl`, `runtime.json`, `stdout.log` and `MANIFEST.json`, which gives the SHA-256 and size of every local file and whether it is committed.
- **Checkpoints.** `ckpt-20000`, `ckpt-30000`, `ckpt-40000` and `ckpt-50000` of all nine continuations are committed: 36 files, about 17 MB in all. Each holds weights, both Adam moments and the step count (`nets.save`). The 10k states are the recovered r1 files in `../recovered/`.
- **Comparing states.** As D2 says, compare arrays, not file hashes.

| run ID | role | resumes (SHA-256 in `../recovered/RECOVERY.json`) | status | held-out final 10k → 50k; threshold |
|---|---|---|---|---|
| c2-time-verify-tt-inter-s51200-a1 | calibration and continuation check, 1k→3k | r1-comp-rnn-inter-s51200-a1/ckpt-1000 | complete, identical | |
| c2-time-verify-tu-inter-s51200-a1 | calibration and continuation check, 1k→3k | r1-rev-comp-rnn-untied-s51200-a1/ckpt-1000 | complete, identical | |
| c2-time-verify-tt-final-s51200-a1 | calibration and continuation check, 1k→3k | r1-comp-rnn-final-s51200-a1/ckpt-1000 | complete, identical | |
| c2-time-tt-inter-s51200-a1 | continuation 10k→50k | r1-comp-rnn-inter-s51200-a1/ckpt-10000 | complete, 144 s | 1.000 → 1.000; threshold 7,700 |
| c2-time-tt-inter-s51201-a1 | continuation 10k→50k | r1-comp-rnn-inter-s51201-a1/ckpt-10000 | complete, 143 s | 0.961 → 1.000; threshold 9,150 |
| c2-time-tt-inter-s51202-a1 | continuation 10k→50k | r1-comp-rnn-inter-s51202-a1/ckpt-10000 | complete, 142 s | 1.000 → 1.000; threshold 8,200 |
| c2-time-tu-inter-s51200-a1 | continuation 10k→50k | r1-rev-comp-rnn-untied-s51200-a1/ckpt-10000 | complete, 158 s | 0.633 → 1.000; threshold 24,900 |
| c2-time-tu-inter-s51201-a1 | continuation 10k→50k | r1-rev-comp-rnn-untied-s51201-a1/ckpt-10000 | complete, 152 s | 0.523 → 1.000; threshold 32,300 |
| c2-time-tu-inter-s51202-a1 | continuation 10k→50k | r1-rev-comp-rnn-untied-s51202-a1/ckpt-10000 | complete, 148 s | 0.516 → 1.000; threshold 33,550 |
| c2-time-tt-final-s51200-a1 | continuation 10k→50k | r1-comp-rnn-final-s51200-a1/ckpt-10000 | complete, 144 s | 0.273 → 0.562; threshold censored (> 50,000) |
| c2-time-tt-final-s51201-a1 | continuation 10k→50k | r1-comp-rnn-final-s51201-a1/ckpt-10000 | complete, 137 s | 0.312 → 0.695; threshold censored (> 50,000) |
| c2-time-tt-final-s51202-a1 | continuation 10k→50k | r1-comp-rnn-final-s51202-a1/ckpt-10000 | complete, 135 s | 0.320 → 0.656; threshold censored (> 50,000) |

Threshold: the first of five consecutive logged evaluations with held-out final accuracy ≥ 0.95, at 50-update resolution. A crossing before 10,000 is read from r1's record.

## Figure

`figures/time_main.png` and `.svg` are drawn from `figures/time_main.csv` by `time_analyze.py measure && time_analyze.py figure`. The supporting CSVs are:
- `time_threshold.csv`: crossings, censoring, 10k and 50k values;
- `time_error_compatibility.csv`: neighbour compatibility of wrong predictions, every 1,000 updates;
- `time_checkpoints.csv`: parameter norm, distance between checkpoints, and the estimated path length.
