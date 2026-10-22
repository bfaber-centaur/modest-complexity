# c2 sharing lane: commands, calibration, budget, runs

Question, predictions and measures: [QUESTION.md](QUESTION.md) (committed at f84dbc3, before any c2-share measurement). Adaptive choices: [ADAPTIVE.md](ADAPTIVE.md). Primary figure: [figures/sharing_grid.png](figures/sharing_grid.png) (+ `.svg`; sources `trajectories.csv`, `endpoints.csv`). Decoder table: [figures/decode_table.png](figures/decode_table.png) (+ `.svg`, `decode_table.csv`). Observations and their limits: [NOTE.md](NOTE.md) (initial comparison; follow-up pending).

## Files

| file | what it is |
|---|---|
| `share_runs.py` | where the 12 runs of the 2×2 live; r1-style metric rows with N/¬N for each (computed from logged `held_final_pred` for the r1 runs) |
| `share_checks.py` | oracle checks, no training: init identity of all four schemes, gradient aggregation, recovered `ckpt-0` = `c2rnn.init`, N/¬N against the reviewer's numbers → `out/checks.txt` |
| `share_replays.py` | replays against the original TT/TU runs: r1 metric keys, `held_subset`, and every array of `ckpt-1000` and `ckpt-10000` → `out/replays.txt` |
| `share_decode.py` | decoder-across-position table (D5, `c2probe.decode_table` unchanged, probe seed 64300) and readout-across-position table → `out/decode_table.csv`, `out/readout_table.csv` |
| `share_groups.py` | per-group parameter norm, Adam-direction and decay step norms, their cosine, fraction of √v̂ below eps, and copy-to-copy cosine, from checkpoints → `out/groups.csv` |
| `share_figure.py` | the two figures and their CSVs |
| `share_publish.py` | copies run records into `runs/<ID>/` with a SHA-256 `MANIFEST.json` |
| `plan.json`, `registry.jsonl` | every planned run; append-only status events from `research/regimes/common/run_jobs.py` |

## Commands (exact)

```bash
export PATH=~/.local/bin:$PATH   # uv 0.12.23 installed per research/regimes/PLAN.md §3 (the container's uv 0.11.32 cannot fetch CPython 3.12.14)
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py      # own process: 10 tests OK, 15 s
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/sharing/plan.json --root runs/c2 --registry research/composition-c2/sharing/registry.jsonl --session-label c2-share-cloud-1"
$R --workers 2 --ids c2-share-calib-tt-s900100-a1 c2-share-calib-tu-s900100-a1 c2-share-calib-ut-s900100-a1 c2-share-calib-uu-s900100-a1
$R --workers 2 --ids c2-share-ut-s51200-a1 c2-share-uu-s51200-a1 c2-share-ut-s51201-a1 c2-share-uu-s51201-a1 c2-share-ut-s51202-a1 c2-share-uu-s51202-a1 \
    c2-share-replay-tt-s51200-a1 c2-share-replay-tu-s51200-a1 c2-share-replay-tt-s51201-a1 c2-share-replay-tu-s51201-a1 c2-share-replay-tt-s51202-a1 c2-share-replay-tu-s51202-a1
P="uv run --locked python research/composition-c2/sharing"
$P/share_publish.py --ids <the 16 IDs above> --commit-checkpoints c2-share-{ut,uu}-s5120{0,1,2}-a1/ckpt-10000.npz
$P/share_checks.py && $P/share_replays.py
$P/share_decode.py --steps 0 10000     # 24 tables, about 60 s
$P/share_groups.py
$P/share_figure.py
```

The figure, decoder and group scripts need the local checkpoints (`runs/c2/<ID>/`) for steps other than 10,000 of UT/UU; the recovered originals cover TT/TU at 0, 1,000 and 10,000. Every run is deterministic and takes about 75 s to regenerate.

## Environment and calibration

Host `vm` (cloud container), 4 visible CPUs, Intel Xeon @ 2.80 GHz, CPython 3.12.14, NumPy 2.3.5, one BLAS thread per job. Each run's `config.json` → `provenance` holds the full inventory.

Calibration `c2-share-calib-{tt,tu,ut,uu}-s900100-a1`: 2,000 updates, two concurrent jobs, the same configuration as the coordinator's `c2-co-calib-*`. Not research runs.
- ms per update: TT 6.11, TU 6.30, UT 7.23, UU 7.12. Peak RSS 44–45 MiB.
- Every `metrics.jsonl` line, `--diag` included, is identical to the coordinator host's.

In the research runs (two concurrent jobs, 10,000 updates, `--diag`, six checkpoints), ms per update was:

| cell | ms per update | run wall |
|---|---|---|
| TT replays | 6.28–6.49 | 64–66 s |
| TU replays | 6.35–6.74 | 65–69 s |
| UT | 7.12–7.42 | 73–76 s |
| UU | 7.29–7.49 | 75–77 s |

The untied-transition cells cost about 12–17% more per update. That is elapsed execution on this host, not a property of the schemes elsewhere; r1's host ran TT at 3.9 ms per update.

## Budget tally (elapsed wall time with at least one runner job, cap 20 min)

| block | runs | wall |
|---|---|---|
| calibration (2 concurrent) | 4 | 27 s |
| initial comparison: 6 new + 6 replays (2 concurrent) | 12 | 426 s |
| **total so far** | 16 | **7.6 of 20 min** |

Analysis outside the runner, on finished checkpoints: decoder tables about 60 s, checks, replays, groups and figures under 10 s each.

## Run manifest

All 16 planned runs completed; none failed or was capped (`registry.jsonl`). Each `runs/<ID>/` holds `config.json`, `metrics.jsonl`, `runtime.json`, `stdout.log` and `MANIFEST.json` (SHA-256 and size of every file of the local run directory, `committed` true/false). The six `ckpt-10000.npz` of UT and UU are committed (about 7.7 MB). The other checkpoints (0, 1k, 2k, 4k, 7k, and the replays' 10k) are local only.

| run ID | role | held-out final at 10k | ¬N at 10k |
|---|---|---|---|
| `c2-share-calib-{tt,tu,ut,uu}-s900100-a1` | calibration, 2,000 updates | not a research result | |
| `r1-comp-rnn-inter-s5120{0,1,2}-a1` | TT (reused original, recovered checkpoints) | 1.000, 0.961, 1.000 | 1.000, 0.881, 1.000 |
| `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1` | TU (reused, reviewer's r1 arm) | 0.633, 0.523, 0.516 | 0.238, 0.071, 0.048 |
| `c2-share-ut-s5120{0,1,2}-a1` | UT (new) | 0.836, 0.680, 0.625 | 0.595, 0.333, 0.190 |
| `c2-share-uu-s5120{0,1,2}-a1` | UU (new) | 0.766, 0.609, 0.703 | 0.381, 0.190, 0.262 |
| `c2-share-replay-tt-s5120{0,1,2}-a1` | replay of TT, not a new seed | identical to the original | |
| `c2-share-replay-tu-s5120{0,1,2}-a1` | replay of TU, not a new seed | identical to the original | |

**Replays.** All six reproduce their originals exactly ([out/replays.txt](out/replays.txt)). That covers all 201 logged rows of r1 metrics, `held_subset` against N/¬N from the original predictions, and every array of `ckpt-1000` and `ckpt-10000`: weights, both Adam moments and the step count. This is a fourth host on which this code path reproduces r1 bitwise. The scientific values for TT and TU are the originals'. The replays add `--diag` and the 2k/4k/7k checkpoints, which `share_groups.py` uses for TT/TU.

**Provenance note.** All 16 runs ran at commit `0b65db5`. The calibration runs and the first two research runs recorded a clean tree apart from the registry. Runs started later record uncommitted lane analysis files in `git_status`:
- `share_figure.py` was being edited;
- `share_replays.py` and `share_publish.py` were new;
- `figures/` existed.

`c2train.py` imports none of these. In every config, the SHA-256 of `c2task.py`, `c2rnn.py`, `c2train.py` and `nets.py` equals the pin `b3e8132`, and the replays' bitwise identity confirms the training path.

## Reviewer handoff

- **Pre-registration.** QUESTION.md `f84dbc3` (commit 04:06:04, pushed before 04:07) precedes the calibration launch at 04:10:54 and the research launch at 04:12:13. Explicit recipe flags were added at `0b65db5` (04:10:38), before launch (pin check C1).
- **Pairing.** Init identity, gradient aggregation and the recovered `ckpt-0`: `out/checks.txt`. Replays: `out/replays.txt`.
- **N/¬N.** For TT and TU these are computed from r1's logged predictions. They match the replays' logged `held_subset` at every step, and the reviewer's 10k numbers.
- **The reviewer's own arm is TU.** Claims about TU, and every paired difference involving TU, rest on the reviewer's r1 runs.
- **Exposure.** Parameter counts are 19,208, 21,272, 53,256 and 55,320, not matched. Supervision is identical: 384 rows × 3 labels × 10k updates. Decay is wd 0.1 on every parameter and every copy. A tied group receives summed gradients; an untied copy receives its own step's term.
- **Group diagnostics** (`out/groups.csv`). Step 0 has no Adam step. TT/TU at 2k, 4k and 7k come from the replays; the `source` column names the run each row was read from.
