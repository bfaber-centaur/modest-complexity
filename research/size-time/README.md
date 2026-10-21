# Size/time lane: expedition e1

Session label `st-cloud-1` (host `vm`, 4 vCPU). Branch `claude/project-thread-j8mgkm` (this session's designated branch, used in place of `swarm/size-time`). Started from `swarm/exploration` at `c7c1c2ad8ac0bb23c9d0f84f9f1c9438764d03c6`, whose shared code is identical to Stage 0 `ebd874e17028f2c2f04de5ecea9fe4b51b77e425`. Coordinator preview runs read from `swarm/exploration` at `67a10d8d282fd1d57ae6b3aba5a992f8d040203d`; all 35 published files there match their manifest hashes.

Figures first; interpretations are in [INTERPRETATIONS.md](INTERPRETATIONS.md).

## Claim (committed before execution, 70503aa)

The 21 plan entries with `owner: size-time`, queue positions 4-24, run in queue order with two workers:

```
e1-st-add-w16-i30001-a1   e1-st-add-w256-i30002-a1  e1-st-add0-w128-i30000-a1
e1-st-shuf-w128-i30000-a1 e1-st-add-w256-i30001-a1  e1-st-add0-w128-i30001-a1
e1-st-add-w64-i30002-a1   e1-st-add-w128-i30001-a1  e1-st-add0-w128-i30002-a1
e1-st-add-w128-i30002-a1  e1-st-add-w16-i30000-a1   e1-st-add-w16-i30002-a1
e1-st-add-w64-i30000-a1   e1-st-add-w32-i30002-a1   e1-st-add-w512-i30002-a1
e1-st-add-w512-i30001-a1  e1-st-add-w64-i30001-a1   e1-st-shuf-w128-i30001-a1
e1-st-add-w256-i30000-a1  e1-st-shuf-w128-i30002-a1 e1-st-add-w32-i30001-a1
```

## Run status

All 21 attempted runs are `complete` at 50,000 updates; none failed or was capped (timeout 1,500 s). No run was repeated, shortened or retried, and no adaptive change was made to the queue. Registry: [registry.jsonl](registry.jsonl) (42 events). With the coordinator's three preview runs, all 24 runs of the first map are complete.

Elapsed numerical execution: 18:38:02-19:00:59 UTC, 22 min 57 s (budget 30 min). Per-job wall time including startup, two jobs at once: width 16 28-30 s, 32 42-43 s, 64 67-68 s, 128 115-121 s, 256 199-269 s, 512 368-369 s. This host ran about 1.4x slower than the coordinator's (512: 368 s vs 264 s).

Every run's `config.json` records `git_commit` `c7c1c2a` with a clean tree: runs executed from a separate worktree checked out (detached) at that commit, with the registry path pointed at this checkout. `runtime.source_sha256` equals the coordinator preview runs'. Split hash `aa8f7504…1f24` was checked by every run before training.

Environment: installed uv 0.11.32 had no CPython 3.12.14 download, so uv 0.12.23 was used as described in the worker prompt. `uv sync --locked --group analysis`; `uv run --locked python -m unittest test_exploration test_instrumentation`: 9 tests OK, in both the lane checkout and the launch worktree, before any run.

## Published runs

All 21 runs are under [runs/](runs/) with `config.json`, `runtime.json`, `metrics.jsonl`, `data.npz`, the last checkpoint, the first-fit checkpoint, and a `manifest.json` with SHA-256 for every local file (published or not). Full default checkpoint sets (steps 0, 1,000, 2,000, 5,000, 10,000, 20,000, 50,000 plus first fit) are published for a rule fixed before results: the first run of each width to finish in queue order, plus all six controls.

| width | full checkpoint set | last + first-fit only |
|---|---|---|
| 16 | `e1-st-add-w16-i30001-a1` | i30000, i30002 |
| 32 | `e1-st-add-w32-i30002-a1` | i30001 (coordinator: i30000) |
| 64 | `e1-st-add-w64-i30002-a1` | i30000, i30001 |
| 128 | `e1-st-add-w128-i30001-a1` | i30002 (coordinator: i30000) |
| 256 | `e1-st-add-w256-i30002-a1` | i30000, i30001 |
| 512 | `e1-st-add-w512-i30002-a1` | i30001 (coordinator: i30000) |
| controls, w128 | `add0` and `shuf`, i30000-30002 | none |

The published tree needs `git add -f` on base `c7c1c2a` (its `.gitignore` line `runs/`); fixed on `swarm/exploration` at 67a10d8.

## Figures

All figures are PNG plus SVG, unsmoothed, one line or row per run, with source tables beside them. The exported PNGs were opened and checked for labels and clipping.

| figure | shows | source table |
|---|---|---|
| [trajectories/overview.png](figures/trajectories/overview.png) | width-by-time map: train, audit and zoomed audit accuracy, all 24 runs | `figures/*/curves.csv` |
| [add/accuracy.png](figures/add/accuracy.png), [add/loss.png](figures/add/loss.png) | addition with decay: accuracy and cross-entropy at matched steps, 18 runs | [add/curves.csv](figures/add/curves.csv), [status.csv](figures/add/status.csv) |
| [add0/](figures/add0/), [shuf/](figures/shuf/) | the same for the w128 controls | `curves.csv`, `status.csv` |
| [trajectories/loss_matched_add.png](figures/trajectories/loss_matched_add.png) | held-out accuracy, cross-entropy and median margin against each run's own train loss | [matched.csv](figures/trajectories/matched.csv) |
| [trajectories/loss_matched_w128_controls.png](figures/trajectories/loss_matched_w128_controls.png) | the same for add, add0 and shuf at width 128 | `matched.csv` |
| [trajectories/margins_add.png](figures/trajectories/margins_add.png) | train minimum margin and audit median margin by step | `add/curves.csv` |
| [trajectories/updates_add.png](figures/trajectories/updates_add.png) | actual last-step AdamW adaptive and decay displacement and their cosine | run `metrics.jsonl` |
| [trajectories/norm_add.png](figures/trajectories/norm_add.png) | raw and width-normalized parameter norm (descriptive) | run `metrics.jsonl` |
| [operand_labels/operand_labels_step50000.png](figures/operand_labels/operand_labels_step50000.png) | where held-out predictions go (packet 3) | [results.jsonl](figures/operand_labels/results.jsonl) |

Per-run descriptors (first and sustained fit, drops, maximum audit accuracy, final losses and margins): [trajectories/summary.csv](figures/trajectories/summary.csv). Thresholds, chosen for description and revisable: fit = train accuracy >= .99, generalize = audit accuracy >= .95, sustained = holds at every later logged step.

## Evidence packets

### 1. Which widths fit, and which generalize, within 50,000 updates?

Figure: [overview.png](figures/trajectories/overview.png); per-step curves [add/accuracy.png](figures/add/accuracy.png), [add/loss.png](figures/add/loss.png).

Look before reading: compare the left and middle panels row by row. Where does each row turn yellow? Does any row turn yellow in the middle panel? In the zoomed right panel, which side of white (chance) are the addition rows on, and the shuffled rows?

Observation. All 18 addition runs reach train accuracy >= .99: first at step 100 (width 512), 200 (256), 300 (128), 400-500 (64), 1,000 (32) and 3,300-6,500 (16). Width 16 does not hold it: 2, 25 and 21 later logged steps fall below .99, train cross-entropy ends at 0.44-0.49, and the final minimum train margin is 0.18-0.31 logits. No run reaches audit accuracy >= .95. For every addition run the highest audit accuracy over all logged steps is at step 0 (0.024-0.044; chance 1/31 = 0.032); final audit accuracy is 0-0.009 (0-5 of 544 pairs). The three seeds at each width are close throughout. At width 128 the shuffled-label runs fit at step 300, like addition, with nearly the same train loss curve (train cross-entropy 0.0125 at step 5,000 and 1.8e-3 at step 10,000 in both conditions).

Missing check. Timing is censored at 50,000 updates; absence of generalization here is not evidence that it cannot occur later. One split (31% train, 299 pairs) only.

### 2. At the same training loss, do widths differ on held-out pairs? Is that the same as at the same step?

Figure: [loss_matched_add.png](figures/trajectories/loss_matched_add.png) (middle panel); compare [add/loss.png](figures/add/loss.png) (bottom row) at a fixed step; controls in [loss_matched_w128_controls.png](figures/trajectories/loss_matched_w128_controls.png).

Look before reading: in the middle panel, at a fixed position on the x axis, how are the widths ordered? Is the order the same when you read the audit cross-entropy at a fixed step instead?

Observation (first logged step at which train cross-entropy <= the level, ranges over three seeds, from [matched.csv](figures/trajectories/matched.csv)):

| width | audit CE at train CE <= 1e-2 | step | audit CE at train CE <= 1e-4 | step | audit CE at step 50,000 |
|---|---|---|---|---|---|
| 16 | not reached (train CE >= 0.44) | | not reached | | 7.5-7.7 |
| 32 | 9.5-9.7 | 28,900-31,100 | not reached | | 9.8-10.1 |
| 64 | 7.7-8.2 | 9,000-9,200 | 10.1-11.0 | 30,200-30,900 | 11.1-11.5 |
| 128 | 7.4-7.5 | 5,600 | 9.0-9.3 | 18,200-18,300 | 9.9-10.5 |
| 256 | 7.1-7.3 | 3,600 | 8.2-8.4 | 14,300-14,400 | 9.1-9.6 |
| 512 | 7.1-7.2 | 2,200 | 8.5-8.6 | 12,100 | 8.8 |

At a given train loss, narrower networks have higher held-out cross-entropy from width 32 to 256; widths 256 and 512 are close and swap at 1e-4. Parameter norms at the 1e-4 crossing are similar (55.5-58.0). At the same step the order is different (width 64 highest at 50,000). Without decay (add0, width 128), audit cross-entropy at train CE <= 1e-4 is 13.4-14.1 (9.0-9.3 with decay) and reaches 23-25 nats at step 50,000 with train cross-entropy 1.2e-9 and norm 90 (60 with decay). Audit cross-entropy is higher at step 50,000 than at step 1,000 in all 18 addition runs (by at least 2.0 nats).

Counterexample. The comparison at equal train loss is not at equal time: width 64 reaches 1e-4 at about 30,000 steps, width 512 at about 12,000, and held-out loss grows with steps in every addition run. Width and elapsed training are confounded here. Held-out cross-entropy measures how confidently wrong the model is; it is not a generalization measure in these runs.

### 3. Why is held-out accuracy below chance?

Figure: [operand_labels_step50000.png](figures/operand_labels/operand_labels_step50000.png); all steps in [results.jsonl](figures/operand_labels/results.jsonl).

Look before reading: for the addition rows, compare the filled dot with the orange tick and with the diamond. What changes in the three shuffled rows?

Definitions: for a held-out pair (a, b), L(a, b) is the set of labels of training pairs (a, b') or (a', b) that share one operand in the same position. For addition, a+b is never in L(a, b), and a pair and its transpose share a split. Computed by [operand_labels.py](operand_labels.py) with its own forward pass from stored checkpoints and `data.npz`.

Observation. In all 21 addition runs (with and without decay) the correct label is in L for 0 of 544 audit pairs. At step 50,000, 96-100% of argmax audit predictions are in L (the expected fraction if wrong predictions were uniform over the 30 wrong labels is 0.536; L has 16.1 labels on average), and the mean softmax mass on L is 0.91-1.00. Width 16 and 32 runs are lowest (0.96-0.99) and widths >= 64 are 0.99-1.00. For the shuffled-label runs the correct label is in L for 44% of audit pairs, predictions are in L for 100%, and audit accuracy is 0.037-0.042. The same pattern holds at steps 1,000, 5,000 and 20,000 in the runs with full checkpoint sets (0.95-1.00 in L). Independent checks in this lane: labels equal (a+b) mod 31, train/cal/audit are disjoint and cover all 961 pairs, no audit pair's transpose is in train, train class counts are 9-10 per class.

Missing check. This is an association between predictions and a label set, not an intervention, and L covers about half the labels, so it is a coarse test. Not yet recomputed by the auditor.

## Proposed follow-ups (not run; ownership to be agreed with the coordinator)

1. Horizon: rerun `add` (decay 1) at widths 64 and 128, init seed 30000, for 300,000 updates as new run IDs (no resume), to address the censoring in packet 1. Estimated 7 and 12 minutes on this host, in parallel.
2. Training fraction: a separately labeled comparison at width 128, three seeds, with a larger train fraction (for example 50%), to ask whether packet 1's absence of generalization belongs to this split's 299 training pairs rather than to width or horizon. Needs a shared-code change (`data()` fixes 30/10/60), so it is the coordinator's call.

## Reproduction

From this branch, with the coordinator runs fetched from `swarm/exploration` at 67a10d8 into `CO` (on an integrated branch use `CO=research/coordinator`):

```bash
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m unittest test_exploration test_instrumentation
# Training (from a checkout of c7c1c2a; never reuse run IDs, so use a fresh registry and attempt number for a replay):
uv run --locked python launch_runs.py --root runs/e1 --registry research/size-time/registry.jsonl \
    --session-label st-cloud-1 --timeout 1500 --workers 2 --ids <the 21 IDs above, in queue order>
# Figures, from published files only:
REG="research/size-time/registry.jsonl $CO/registry.jsonl"
for c in add add0 shuf; do
  uv run --locked --group analysis python report_map.py --runs research/size-time/runs $CO/runs \
      --registries $REG --condition $c --out research/size-time/figures/$c
done
uv run --locked --group analysis python research/size-time/trajectories.py --runs research/size-time/runs $CO/runs \
    --registries $REG --out research/size-time/figures/trajectories
uv run --locked --group analysis python research/size-time/operand_labels.py research/size-time/runs/* $CO/runs/* \
    --out research/size-time/figures/operand_labels
```

A replay of the same configuration and seed is a reproducibility check, not a new seed. Bitwise agreement across hosts is not promised.
