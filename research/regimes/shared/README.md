# R2 shared subcomputations: commands, environment, budget, runs

Design and predictions: [QUESTION.md](QUESTION.md). Findings: [NOTE.md](NOTE.md).

## Files

| file | what |
|---|---|
| `task.py` | the 1,024 inputs, label functions (y1, y2, part, unrel, follow-up split), probe targets, split, coverage |
| `test_task.py` | independent oracle and split checks (itertools/boolean recomputation, relevant-bit enumeration, disjoint/nested splits) |
| `train.py` | one run = one arm x one replicate (transfer with frozen readout, or joint) |
| `analyze.py` | `publish` (run files + `runs/MANIFEST.sha256`), `figures`, `mechanism` |
| `plan.json`, `registry.jsonl` | run plan (pushed before launch) and the runner's append-only log |
| `figures/` | `transfer` (primary) and `final.csv` / `transfer_trajectories.csv`; `joint`; `flip` + `flip_sensitivity.csv`; follow-up `followup` (split arm + layer-2 probes); `mechanism.csv` (probes, steering) |
| `runs/<ID>/` | config, summary, runtime, gzipped metrics, and published checkpoints |

## Commands

```bash
export PATH=~/.local/bin:$PATH   # uv 0.12.23, PLAN.md section 3
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m unittest -v research/regimes/common/test_nets.py research/regimes/shared/test_task.py
# runs (IDs from plan.json, in plan order)
uv run --locked python research/regimes/common/run_jobs.py --plan research/regimes/shared/plan.json \
    --root runs/regimes --registry research/regimes/shared/registry.jsonl --session-label r1-share-cloud-1 --workers 1 --ids ...
# publish: every run with its phase-1 checkpoint, then fresh/rel/part/unrel again adding final_n512.npz,
# then the same-task arm with no checkpoints (kept local to stay near the 2 MB checkpoint limit)
uv run --locked python research/regimes/shared/analyze.py publish --root runs/regimes --ids <all IDs> --checkpoints pretrained.npz
uv run --locked python research/regimes/shared/analyze.py publish --root runs/regimes \
    --ids <r1-share-{fresh,rel,part,unrel}-s5220{0,1,2}-a1> --checkpoints pretrained.npz final_n512.npz
uv run --locked python research/regimes/shared/analyze.py publish --root runs/regimes --ids <r1-share-same-s5220{0,1,2}-a1> --checkpoints
uv run --locked python research/regimes/shared/analyze.py mechanism   # probes and steering on published checkpoints -> figures/mechanism.csv
uv run --locked python research/regimes/shared/analyze.py figures     # reads only runs/, registry.jsonl and mechanism.csv
```

Published checkpoints: about 2.2 MB in total. `pretrained.npz` is published for rel, part, unrel and split, and `final_n512.npz` for fresh, rel, part and unrel. `final_n64.npz`, the other `final_n512.npz`, the same-arm `pretrained.npz` and raw `metrics.jsonl` stay local; their SHA-256 are in the manifest marked "(local only)". Each run is deterministic given its seeds, so they can be regenerated (about 10 s per run).

## Environment

- **Host.** Cloud container `vm`, Intel Xeon @ 2.10 GHz, 4 visible CPUs (an earlier note said 1: that was `nproc` honoring `OMP_NUM_THREADS=1`).
- **Software.** CPython 3.12.14, NumPy 2.3.5, OpenBLAS 0.3.30, all BLAS threads 1, locked uv environment. The runner sets the glibc malloc thresholds.
- **Workers.** The initial runs used `--workers 1`; the three follow-up runs used `--workers 2`. Two concurrent runs took 10.1 and 10.3 s, the same as serial.
- **Memory.** Peak RSS was about 40 MiB per run.
- **Per-run record.** `config.json` → `provenance` (git commit, source and lock hashes, cgroup).

## Calibration

`r1-share-calib-rel-s952200-a1` ran the actual script on the `rel` arm with calibration seeds 952200/952100. It is not pooled.

| phase | updates | wall time |
|---|---|---|
| pretraining | 2,000 updates on 512 rows | 2.1 s |
| phase 2, seven n | 7 × 2,000 | 6.1 s |
| frozen readouts | 7 × 2,000 | 1.6 s |
| **whole run, including startup** | | **10.1 s** |

Timeouts were set at 30 s (3×). This is a 10 s measurement, not a forecast for other sizes.

## Budget tally (elapsed numerical time, 25 min cap)

| step | runs | wall s (runner, incl. startup) |
|---|---|---|
| smoke | 1 | 0.6 |
| calibration | 1 | 10.1 |
| initial comparison (`--workers 1`) | 24 | 212.8 |
| follow-up `split` (`--workers 2`; 30.5 s of job time, 20 s elapsed) | 3 | 20 |
| **total elapsed** | 29 | **244 (4.1 min)** |

Analysis scripts each take under 10 s and are not counted.

## Run manifest summary

All 29 runs completed. None failed or was capped, and no ID has more than one attempt. Initial comparison: arms `fresh`, `rel`, `part`, `unrel`, `same`, `joint-rel`, `joint-unrel`, `alone-y1` × replicates s52200/52201/52202 (data seeds 52100/52101/52102). The calibration and smoke runs (seeds 952xxx) are recorded but never pooled. Follow-up: `split` × the same three replicates, planned and pushed at 1d9d240 before launch.
