# c3 scaffolding lane: commands, environment, budget, runs

- **Design and predictions.** [QUESTION.md](QUESTION.md), committed at b1928fe before any result.
- **Adaptive choices.** [ADAPTIVE.md](ADAPTIVE.md).
- **Observations.** [NOTE.md](NOTE.md).
- **Primary figure.** [figures/scaf_main.png](figures/scaf_main.png), with `.svg` and the CSVs beside it.

## Files

| file | what it is |
|---|---|
| `plan.json`, `registry.jsonl` | every planned run, and append-only status events from `research/regimes/common/run_jobs.py` |
| `scaf_analyze.py` | `publish`: copies the small run files and the checkpoints listed with `--commit-checkpoints` into `runs/`, with a SHA-256 manifest of everything local. `check`: joins at S, step weights against the schedule, recorded schedule points, exposure recount, recipe, and the frozen evaluation against c2's published values; writes `checks.json`. `measure`: the CSVs in `figures/`. `figure`: draws `scaf_main` from them. `followup-check`, `followup-measure`, `followup-figure`: the same for the follow-up (`scaf_followup*`). |
| `test_scaf.py` | plan constants; schedule restatement against c3train; threshold and retention rules on hand-made inputs; checkpoint inventory; frozen forward against the training forward and a plain per-row loop |
| `checks.json` | output of `check` |
| `followup_checks.json` | output of `followup-check`: the no-reset matched-control checks, and the join, schedule, reset step, exposure and Adam count of each reset run |
| `runs/<ID>/` | `config.json`, `metrics.jsonl`, `runtime.json`, `stdout.log` and `MANIFEST.json`, plus the committed checkpoints |

The shared code is `../common/` at pin `be41d16`, unedited. It imports c2's `../../composition-c2/common/` (`b3e8132`).

## Commands (exact)

```bash
export PATH=~/.local/bin:$PATH   # uv 0.12.23
uv sync --locked --group analysis
uv run --locked python -m unittest -v research/composition-c3/common/test_c3.py          # own process
uv run --locked python -m unittest -v research/composition-c3/scaffolding/test_scaf.py   # own process
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c3/scaffolding/plan.json --root runs/c3 --registry research/composition-c3/scaffolding/registry.jsonl --session-label c3-scaf-cloud-1"
$R --workers 2 --ids c3-scaf-calib-drop-s900210-a1 c3-scaf-calib-taper-s900210-a1
$R --workers 2 --ids c3-scaf-drop-at1k-s51200-a1 c3-scaf-taper-at1k-s51200-a1 c3-scaf-drop-at1k-s51201-a1 c3-scaf-taper-at1k-s51201-a1 \
    c3-scaf-drop-at1k-s51202-a1 c3-scaf-taper-at1k-s51202-a1 c3-scaf-drop-at10k-s51200-a1 c3-scaf-taper-at10k-s51200-a1 \
    c3-scaf-drop-at10k-s51201-a1 c3-scaf-taper-at10k-s51201-a1 c3-scaf-drop-at10k-s51202-a1 c3-scaf-taper-at10k-s51202-a1
A="uv run --locked python research/composition-c3/scaffolding/scaf_analyze.py"
$A publish --root runs/c3 --ids c3-scaf-calib-drop-s900210-a1 c3-scaf-calib-taper-s900210-a1
$A publish --root runs/c3 --ids <the 12> --commit-checkpoints <ID>/ckpt-<t>.npz ...   # every saved checkpoint of the 12
$A check && $A measure && $A figure
# follow-up (QUESTION.md addendum; pin 82cef77): matched-control checks first, then the six reset runs
$R --workers 2 --ids c3-scaf-verify-carried-drop-at1k-s51200-a1 c3-scaf-verify-carried-keep-at1k-s51200-a1 \
    c3-scaf-verify-carried-keep-at1k-s51201-a1 c3-scaf-verify-carried-keep-at1k-s51202-a1
$A publish --root runs/c3 --ids <the four checks> --commit-checkpoints c3-scaf-verify-carried-drop-at1k-s51200-a1/ckpt-3000.npz
$A followup-check     # must pass before the next line
$R --workers 2 --ids c3-scaf-drop-reset-at1k-s51200-a1 c3-scaf-keep-reset-at1k-s51200-a1 c3-scaf-drop-reset-at1k-s51201-a1 \
    c3-scaf-keep-reset-at1k-s51201-a1 c3-scaf-drop-reset-at1k-s51202-a1 c3-scaf-keep-reset-at1k-s51202-a1
$A publish --root runs/c3 --ids <the six> --commit-checkpoints <ID>/ckpt-<t>.npz ...   # every saved checkpoint of the six
$A followup-check && $A followup-measure && $A followup-figure
```

## Environment and calibration

- **Host.** `vm`, a cloud container: 4 visible CPUs, Intel Xeon @ 2.10 GHz, CPython 3.12, NumPy 2.3.5, one BLAS thread per job (`run_jobs.py` sets the thread variables). Each run's `config.json` → `provenance` holds the full inventory.
- **Calibration.** `c3-scaf-calib-{drop,taper}-s900210-a1` ran from a fresh init (seed 900210) for 2,000 updates, with the switch at 500, `--diag`, and two concurrent jobs.
  - Times: 6.3 / 6.8 ms per update, or 6.5 / 7.1 including evaluation.
  - Peak RSS: 44 MiB.
  - The schedule is as declared: the intermediate weight is 0.999 at 500, 0.499 at 1,000 and 0 at 1,500 (taper), and the exposure is [500, 500, 2000] (drop) and [1499, 1499, 2000] (taper).
- **Follow-up pin.** `82cef77` (DECISIONS D10) adds `--reset-optimizer-at`. The reviewer passed it (c3 `review/REVIEW.md` §4, PR #66), as the coordinator relayed at 06:18Z. The 12 initial branches ran at `be41d16`, and their arithmetic is unchanged at the new pin. The four no-reset check runs show this on this host (`followup_checks.json`).
- **Pin.** The reviewer passed pin `be41d16` (c3 `review/REVIEW.md` §1, PR #62), which the coordinator relayed at 05:30Z. The 12 branches launched at 05:31Z.
- **Recipe flags.** Every branch spells out the full recipe, and `c3train.check_source` compared each one with its r1 source before training. Each run records the result in `config.json` → `resumed_from.source`.

## Budget tally (elapsed wall time with at least one job running)

| block | runs | wall |
|---|---|---|
| calibration (2 concurrent), 05:22:37–05:22:51Z | 2 | 14 s |
| 12 branches (2 concurrent), 05:31:23–05:58:21Z | 12 | 1,618 s |
| **total, initial** | 14 | **about 27.2 of 40 min** (80% was not reached) |
| follow-up matched-control checks (2 concurrent), 06:18:23–06:18:51Z | 4 | 28 s |
| follow-up reset runs (2 concurrent), 06:19:27–06:35:47Z | 6 | 980 s |
| **total, follow-up** | 10 | **about 16.8 of 20 min**. The 16-minute mark was passed less than a minute before the last run ended, and the coordinator was told. |

- **Per run, follow-up.** 298–301 s (drop-reset) and 323–330 s (keep-reset) for 49,000 updates; the checks took 13–14 s each.
- **Per run.** S = 1k: 284–305 s for 49,000 updates (5.8–6.2 ms per update, including evaluation). S = 10k: 234–254 s for 40,000 (5.9–6.3 ms). Every run stayed under its 900 s timeout.
- **Analysis.** `check`, `measure` and `figure` take seconds to a minute per call and are not counted.

## Run manifest

None failed or was capped (`registry.jsonl`). Each `runs/<ID>/` holds `config.json`, `metrics.jsonl`, `runtime.json`, `stdout.log` and `MANIFEST.json`, which gives the SHA-256 and size of every local file and whether it is committed.

- **Checkpoints.** Every saved checkpoint of the 12 branches is committed: 66 files, about 29 MB. So is every saved checkpoint of the six reset runs: 36 files (3k, 10k, 20k–50k), about 16 MB. The check runs' checkpoints are listed in their manifests, and only `c3-scaf-verify-carried-drop-at1k-s51200-a1/ckpt-3000.npz` is committed, for the equality check. Each holds weights, both Adam moments and the step count (`nets.save`). That is S + 2,000, 10,000 (S = 1k only) and 20k–50k. The switch states are the recovered r1 files in `../../composition-c2/recovered/`. The calibration checkpoints are not committed and are listed in their manifests.
- **Provenance.** Every branch records commit `de9ff42` and a `git status` showing only `registry.jsonl`, which `run_jobs.py` appends to. The training-code and environment hashes are identical in all 12. The four follow-up checks record `c7fde16` and the six reset runs `abfa30b`, each again with only `registry.jsonl` in `git status`, and all ten share one c3train hash (pin `82cef77`).

| run ID | role | resumes | status | held-out final at S → 50k; threshold |
|---|---|---|---|---|
| c3-scaf-calib-drop-s900210-a1 | calibration, fresh init, 2,000 updates, switch at 500 | – | complete, 13 s | |
| c3-scaf-calib-taper-s900210-a1 | calibration, fresh init, 2,000 updates, switch at 500 | – | complete, 14 s | |
| c3-scaf-drop-at1k-s51200-a1 | B drop at 1,000 | r1-comp-rnn-inter-s51200-a1/ckpt-1000 | complete, 303 s | 0.336 → 1.000; 14,600 |
| c3-scaf-drop-at1k-s51201-a1 | B drop at 1,000 | r1-comp-rnn-inter-s51201-a1/ckpt-1000 | complete, 293 s | 0.352 → 1.000; 20,000 |
| c3-scaf-drop-at1k-s51202-a1 | B drop at 1,000 | r1-comp-rnn-inter-s51202-a1/ckpt-1000 | complete, 284 s | 0.359 → 1.000; 20,450 |
| c3-scaf-taper-at1k-s51200-a1 | C taper (2,000) at 1,000 | r1-comp-rnn-inter-s51200-a1/ckpt-1000 | complete, 305 s | 0.336 → 1.000; 12,050 |
| c3-scaf-taper-at1k-s51201-a1 | C taper (2,000) at 1,000 | r1-comp-rnn-inter-s51201-a1/ckpt-1000 | complete, 295 s | 0.352 → 1.000; 21,400 |
| c3-scaf-taper-at1k-s51202-a1 | C taper (2,000) at 1,000 | r1-comp-rnn-inter-s51202-a1/ckpt-1000 | complete, 286 s | 0.359 → 1.000; 15,300 |
| c3-scaf-drop-at10k-s51200-a1 | B drop at 10,000 | r1-comp-rnn-inter-s51200-a1/ckpt-10000 | complete, 234 s | 1.000 → 1.000; 7,700 (before S, from r1) |
| c3-scaf-drop-at10k-s51201-a1 | B drop at 10,000 | r1-comp-rnn-inter-s51201-a1/ckpt-10000 | complete, 240 s | 0.961 → 1.000; 9,150 (before S, from r1) |
| c3-scaf-drop-at10k-s51202-a1 | B drop at 10,000 | r1-comp-rnn-inter-s51202-a1/ckpt-10000 | complete, 252 s | 1.000 → 1.000; 8,200 (before S, from r1) |
| c3-scaf-taper-at10k-s51200-a1 | C taper (2,000) at 10,000 | r1-comp-rnn-inter-s51200-a1/ckpt-10000 | complete, 236 s | 1.000 → 1.000; 7,700 (before S, from r1) |
| c3-scaf-taper-at10k-s51201-a1 | C taper (2,000) at 10,000 | r1-comp-rnn-inter-s51201-a1/ckpt-10000 | complete, 242 s | 0.961 → 1.000; 9,150 (before S, from r1) |
| c3-scaf-taper-at10k-s51202-a1 | C taper (2,000) at 10,000 | r1-comp-rnn-inter-s51202-a1/ckpt-10000 | complete, 254 s | 1.000 → 1.000; 8,200 (before S, from r1) |
| c3-scaf-verify-carried-keep-at1k-s51200-a1 | follow-up check, no reset under 82cef77, 1k→3k | r1-comp-rnn-inter-s51200-a1/ckpt-1000 | complete, 14 s, identical to r1 | |
| c3-scaf-verify-carried-keep-at1k-s51201-a1 | follow-up check, no reset under 82cef77, 1k→3k | r1-comp-rnn-inter-s51201-a1/ckpt-1000 | complete, 14 s, identical to r1 | |
| c3-scaf-verify-carried-keep-at1k-s51202-a1 | follow-up check, no reset under 82cef77, 1k→3k | r1-comp-rnn-inter-s51202-a1/ckpt-1000 | complete, 14 s, identical to r1 | |
| c3-scaf-verify-carried-drop-at1k-s51200-a1 | follow-up check, no reset under 82cef77, 1k→3k | r1-comp-rnn-inter-s51200-a1/ckpt-1000 | complete, 13 s, identical to c3-scaf-drop-at1k-s51200-a1 | |
| c3-scaf-keep-reset-at1k-s51200-a1 | A′ keep, Adam reset at 1,000 | r1-comp-rnn-inter-s51200-a1/ckpt-1000 | complete, 327 s | 0.336 → 1.000; 8,150 |
| c3-scaf-keep-reset-at1k-s51201-a1 | A′ keep, Adam reset at 1,000 | r1-comp-rnn-inter-s51201-a1/ckpt-1000 | complete, 323 s | 0.352 → 1.000; 8,750 |
| c3-scaf-keep-reset-at1k-s51202-a1 | A′ keep, Adam reset at 1,000 | r1-comp-rnn-inter-s51202-a1/ckpt-1000 | complete, 330 s | 0.359 → 1.000; 8,950 |
| c3-scaf-drop-reset-at1k-s51200-a1 | B′ drop at 1,000, Adam reset at 1,000 | r1-comp-rnn-inter-s51200-a1/ckpt-1000 | complete, 299 s | 0.336 → 1.000; 12,250 |
| c3-scaf-drop-reset-at1k-s51201-a1 | B′ drop at 1,000, Adam reset at 1,000 | r1-comp-rnn-inter-s51201-a1/ckpt-1000 | complete, 298 s | 0.352 → 1.000; 14,100 |
| c3-scaf-drop-reset-at1k-s51202-a1 | B′ drop at 1,000, Adam reset at 1,000 | r1-comp-rnn-inter-s51202-a1/ckpt-1000 | complete, 301 s | 0.359 → 1.000; 16,350 |

Threshold: the first of five consecutive logged evaluations with held-out final accuracy ≥ 0.95, at 50-update resolution, read on the trajectory from init. r1 tied/inter's rows come before S.
