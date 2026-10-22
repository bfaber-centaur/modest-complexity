# R4 competing predictive routes (r1 lane `routes`)

Question, design, predictions and stopping rules: [QUESTION.md](QUESTION.md), written and pushed (020c86b) before any result. The handoff note will be NOTE.md (M4).

## Files

| file | what it is |
|---|---|
| `task.py` | domain (1,024 inputs, ±1), split by context, nested exceptions, evaluation environments, flip pairs, metrics |
| `test_routes.py` | independent rebuild of the domain with `itertools` and pure-Python parity; checks of the split, conditions, pairs and metrics on oracle functions; finite-difference gradients of the actual 10-128-2 network on ±1 inputs |
| `train.py` | one run: full-batch Adam on 256 inputs, scoring the whole domain at about 100 log-spaced steps |
| `figures.py` | figures and CSVs from the published run files and the registry |
| `plan.json`, `registry.jsonl` | lane plan (every entry pushed before its launch) and the append-only run registry |
| `runs/<ID>/` | published `config.json`, `metrics.jsonl`, `runtime.json`, `final.npz` (weights + Adam state); hashes in `runs/MANIFEST.sha256` |
| `figures/` | PNG + SVG + CSV per figure |

## Commands

```bash
export PATH=~/.local/bin:$PATH   # uv 0.12.23, installed as in ../PLAN.md §3
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m unittest -v research/regimes/routes/test_routes.py        # 8 tests, all pass
# calibration / positive controls, then the initial comparison (IDs as in plan.json):
uv run --locked python research/regimes/common/run_jobs.py --plan research/regimes/routes/plan.json \
    --root runs/regimes --registry research/regimes/routes/registry.jsonl --session-label r1-route-cloud-1 \
    --workers 2 --ids <IDs>
# publish small files with hashes:
for d in runs/regimes/r1-route-*/; do id=$(basename $d); mkdir -p research/regimes/routes/runs/$id
  cp $d/{config.json,metrics.jsonl,runtime.json,final.npz} research/regimes/routes/runs/$id/; done
(cd research/regimes/routes/runs && sha256sum */* > MANIFEST.sha256)
uv run --locked python research/regimes/routes/figures.py          # initial-comparison figures
```

## Environment and calibration (this host)

- **Host.** Linux 6.18.44 x86_64, glibc 2.39, 4 visible CPUs (Intel Xeon @ 2.80GHz), 16 GB RAM.
- **Software.** CPython 3.12.14, NumPy 2.3.5, OpenBLAS 0.3.30, all thread variables set to 1, and the malloc settings from `run_jobs.py`.
- **Calibration** used the actual script at T = 20,000 with init seed 54300, run two at a time. These runs are also the positive controls.
  - `r1-route-calib-rho1of2-s54300-a1` (rule-only) took 14.5 s wall and 14.4 s CPU, about 1,440 updates/s, with a peak RSS of 46 MiB.
  - `r1-route-calib-shortcut_only-s54300-a1` took 14.2 s wall and 14.0 s CPU, about 1,470 updates/s, with a peak RSS of 46 MiB.
  - Two concurrent jobs did not slow each other. Evaluation at 100 log steps costs about 0.2 s per run.
- **Positive controls.**
  - **Rule-only.** It reaches held-out reversed and independent accuracy 1.0 by about 1,000 updates.
  - **Shortcut-only.** It reaches train accuracy 1.0 and a pure-shortcut held-out signature (reliable 1, reversed 0) by about 90 updates.
  - Both routes are trainable and the evaluation pipeline can register either, so T = 20,000 was kept as planned.
- **Timeouts.** 60 s for 20,000-update runs, about 4× the calibrated time.

## Budget tally (elapsed wall time with at least one job running; two concurrent jobs count once)

| step | run IDs | wall |
|---|---|---|
| smoke test (200 updates) | `r1-route-smoke-rho7of8-s54300-a1` | 0.6 s |
| calibration and positive controls, 2 concurrent | `r1-route-calib-*-s54300-a1` | 14.9 s |
| initial comparison, 21 runs, 2 concurrent | `r1-route-{7 conditions}-s{54200,54201,54202}-a1` | 149.4 s |
| **total** | | **165 s ≈ 2.8 of 25 min** |

Figure rendering takes about 9 s and is not counted (PLAN §4).

## Run manifest

There are 24 planned runs, and all 24 are complete. None failed, none was capped and none is missing (`registry.jsonl`). The initial comparison is 7 conditions × 3 paired init seeds (54200–54202), all at T = 20,000. Every condition shares the split seed 54100 and the exception seed 54101. The calibration and smoke runs use init seed 54300 and are not pooled.

## First figure (M2)

[figures/routes_over_training.png](figures/routes_over_training.png) (also `.svg`; data in `routes_over_training.csv`) has one row per condition. Its columns are:
- train accuracy;
- held-out accuracy in the reliable, independent and reversed environments;
- prediction flip rates;
- signed logit-margin changes.

All three seeds are drawn as individual traces. [figures/routes_at_T.png](figures/routes_at_T.png) (`routes_at_T.csv`) shows the values at T for each seed.

Observations at T = 20,000, with each range covering the three seeds:
- **ρ = 1** and **shortcut-only** show a pure shortcut signature: held-out reversed accuracy 0.00, shortcut flip 1.00, and a signed rule Δmargin of 0.03–0.13 against a shortcut Δmargin of 38–40.
- **ρ = 31/32.** Predictions follow the shortcut (reversed accuracy 0.03–0.06), and the 8 exceptions are fit (train accuracy 1.0). The rule's logit contribution is small but positive and growing: Δrule = 4.5–7.2 against Δs = 37–39.
- **ρ = 15/16** is mixed. Reversed accuracy is 0.25–0.42, Δs = 23–27 against Δrule = 18–21, and the irrelevant-bit flip rate is 0.11–0.14, so which route wins depends on the held-out context.
- **ρ = 7/8** has mostly switched to the rule: reversed accuracy 0.95–0.98, Δrule = 28 against Δs = 12.
- **ρ = 3/4 and ρ = 1/2.** Reversed accuracy is 1.00. The residual Δs is 4.7–5.5 at 3/4 and 0.5–0.7 at 1/2.
- **Dynamics.** Every condition with ρ > 1/2 first passes through a shortcut phase, with shortcut flip ≥ 0.95 at 22–41 updates.
  - **Fit.** The training set, exceptions included, is fully fit at 366–603 updates. At that point reversed accuracy is still low: 0.00 at 31/32, 0.04–0.06 at 15/16, 0.11–0.12 at 7/8 and 0.30–0.33 at 3/4.
  - **Rule majority.** Reversed accuracy crosses 0.5 at 546–737 updates for ρ = 3/4, when train loss is 1–2·10⁻². For ρ = 7/8 it crosses at 2,989–3,651 updates, when train loss is 1–3·10⁻⁴. It has not crossed by T for 15/16 or 31/32.
  - **Still moving.** The rule's contribution is still rising at T for ρ = 31/32, 15/16 and 7/8, so the apparent reliability threshold at T depends on T.
- **Flip-rate identity.** Whenever reliable accuracy is 1, shortcut flip = 1 − reversed accuracy and rule flip = reversed accuracy. The prediction-level flip rates then add nothing beyond the environment accuracies. The logit-level signed deltas are the more informative dependence measure.
