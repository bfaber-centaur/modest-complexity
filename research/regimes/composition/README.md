# R1 composing small programs: commands, environment, budget, runs

Design and predictions: [QUESTION.md](QUESTION.md). Observations: [NOTE.md](NOTE.md). Primary figure: [figures/heldout.png](figures/heldout.png) (+ `.svg`, CSVs beside it). Supporting: [figures/decoding.png](figures/decoding.png) (probes and interchange). Adaptive follow-up, kept separate: [figures/followup_rnn.png](figures/followup_rnn.png).

## Files

| file | what it is |
|---|---|
| `task.py` | ops, Latin-square split, rows/encoding, equivalence/coverage checks and non-trained baselines (`python task.py` prints them) |
| `test_task.py` | independent checks: labels against permutation-matrix products, split counts by array sums, encoding round trip, no row leakage, a train.py smoke run (bitwise-identical init across conditions) |
| `train.py` | one run: conditions `final`, `inter`, `shuf`, `full` (P1), `interp` (P2); `--model rnn` (follow-up: shared transition and readout, own backpropagation, finite-difference checked in `test_task.py`); writes `config.json`, `metrics.jsonl`, `runtime.json`, checkpoints |
| `analyze.py` | `publish` (small run files + SHA-256 manifest into `runs/`), `probe` (linear probes and interchange interventions from local checkpoints, each checked against its published SHA-256), `figure` (all three figures and their CSVs from `runs/`) |
| `plan.json`, `registry.jsonl` | every planned run; append-only status events from `common/run_jobs.py` |

## Commands (exact)

```bash
export PATH=~/.local/bin:$PATH   # uv 0.12.23, PLAN §3
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m unittest -v research/regimes/common/test_nets.py research/regimes/composition/test_task.py
uv run --locked python research/regimes/composition/task.py
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/regimes/composition/plan.json --root runs/regimes --registry research/regimes/composition/registry.jsonl --session-label r1-comp-cloud-1"
$R --workers 1 --ids r1-comp-calib-inter-s900000-a1
$R --workers 2 --ids r1-comp-full-s51200-a1 r1-comp-interp-s51200-a1
$R --workers 2 --ids r1-comp-final-s51200-a1 r1-comp-inter-s51200-a1 r1-comp-shuf-s51200-a1 \
    r1-comp-final-s51201-a1 r1-comp-inter-s51201-a1 r1-comp-shuf-s51201-a1 \
    r1-comp-final-s51202-a1 r1-comp-inter-s51202-a1 r1-comp-shuf-s51202-a1
$R --workers 1 --ids r1-comp-probe-a1     # analyze.py probe: needs the local ckpt-0/ckpt-10000 files of the nine runs
$R --workers 1 --ids r1-comp-calib-rnn-inter-s900001-a1
$R --workers 2 --ids r1-comp-rnn-final-s51200-a1 r1-comp-rnn-inter-s51200-a1 r1-comp-rnn-final-s51201-a1 \
    r1-comp-rnn-inter-s51201-a1 r1-comp-rnn-final-s51202-a1 r1-comp-rnn-inter-s51202-a1
uv run --locked python research/regimes/composition/analyze.py publish --root runs/regimes --ids <all IDs above> \
    --commit-checkpoints r1-comp-final-s51200-a1/ckpt-10000.npz r1-comp-inter-s51200-a1/ckpt-10000.npz r1-comp-shuf-s51200-a1/ckpt-10000.npz
uv run --locked python research/regimes/composition/analyze.py figure
```

## Environment and calibration

Host `vm` (cloud container), 4 visible CPUs, Intel Xeon @ 2.10 GHz, CPython 3.12.14, NumPy 2.3.5, one BLAS thread per job (`run_jobs.py` sets the thread variables and the malloc thresholds). Full inventory (NumPy config, cgroup limits, git commit, source hashes) is in every run's `config.json` → `provenance`.

Calibration `r1-comp-calib-inter-s900000-a1` (seed 900000, not a research run): 1,000 `inter` updates on 384 rows in 2.1 s (about 480 updates/s), peak RSS 39 MiB. Timeouts were set from it: 90 s for 10,000 updates on 384 rows (expected about 25 s), 120 s for the 512-row P1. Observed with two concurrent jobs: `final` 16 s, `inter`/`shuf` 21–22 s, `full` 27 s, `interp` 4 s; peak RSS 37–40 MiB per process. That is a short-run measurement, not a forecast for longer runs.

## Budget tally (elapsed wall time with at least one job running)

| block | runs | wall |
|---|---|---|
| calibration | 1 | 2 s |
| positive controls P1, P2 (2 concurrent) | 2 | 27 s |
| initial comparison (2 concurrent) | 9 | 98 s |
| decoding and interventions (`r1-comp-probe-a1`) | 1 | 35 s |
| RNN calibration | 1 | 4 s |
| follow-up RNN (2 concurrent) | 6 | 121 s |
| **total** | 20 | **about 4.8 of 25 min** |

## Run manifest

All 20 planned runs completed; none failed or was capped (`registry.jsonl`). `r1-comp-probe-a1` publishes `probes.csv` and `interventions.csv`; it writes no `config.json`, its inputs being the checkpoints it verified against the published manifests, and its code being `analyze.py probe` as of c493823 (unchanged since). Each `runs/<ID>/` holds `config.json`, `metrics.jsonl`, `runtime.json`, `stdout.log` and `MANIFEST.json` (SHA-256 and size of every file in the local run directory, with `committed` true or false). Checkpoints `ckpt-0`, `ckpt-1000` and `ckpt-10000` stay local except the three `ckpt-10000.npz` of init seed 51200 (`final`, `inter`, `shuf`), about 1.5 MB; every run is deterministic and takes under 50 s to regenerate.

| run ID | role | held-out final accuracy at 10,000 updates |
|---|---|---|
| r1-comp-calib-inter-s900000-a1 | calibration (1,000 updates) | not a research result |
| r1-comp-full-s51200-a1 | P1, all 512 rows trained | 1.000 (trained rows) |
| r1-comp-interp-s51200-a1 | P2, trained transitions iterated | 1.000 |
| r1-comp-final-s5120{0,1,2}-a1 | initial comparison | 0.164, 0.172, 0.141 |
| r1-comp-inter-s5120{0,1,2}-a1 | initial comparison | 0.242, 0.203, 0.250 |
| r1-comp-shuf-s5120{0,1,2}-a1 | initial comparison | 0.125, 0.109, 0.102 |
| r1-comp-probe-a1 | decoding and interchange on the nine runs' checkpoints 0 and 10,000 | (analysis) |
| r1-comp-calib-rnn-inter-s900001-a1 | RNN calibration (1,000 updates, 4.2 s) | not a research result |
| r1-comp-rnn-final-s5120{0,1,2}-a1 | follow-up | 0.273, 0.312, 0.320 |
| r1-comp-rnn-inter-s5120{0,1,2}-a1 | follow-up | 1.000, 0.961, 1.000 |
