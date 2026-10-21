# Mechanism/measurement lane: figures first

Lane branch `claude/project-thread-psralm` (the session's designated branch, standing in for `swarm/mechanisms`), started from `c7c1c2ad8ac0bb23c9d0f84f9f1c9438764d03c6` (shared code = Stage 0 `ebd874e`). Shared code is unchanged. Interpretations are in [INTERPRETATION.md](INTERPRETATION.md). Nothing here is audited by the auditor lane yet.

Definitions used throughout (`freq.py`):

- **Frequency k** = rfft bin k = 1..15 of the input embeddings U and V along the token axis; energy = |coefficient|² summed over hidden units and over U and V. W is never filtered. This is the historical `interventions()` definition generalized from K=4 to every K; parity at K=4 is a unit test.
- **Checkpoint-local selection**: the ranking comes from the analysed checkpoint itself. No retrospective (final-checkpoint) tracing yet.
- **Retain top-K** keeps DC plus the K highest-energy frequencies (K=0: DC only, K=15: the original). **Remove top-K** subtracts them (K=0: the original, K=15: DC only).
- **Random K-subsets**: 16 uniform draws per K (seed 33100), a reference that is **not energy-matched**; the energy fraction of each draw is in `random_subsets.csv`.
- **Rescaling**: hidden unit j gets gain c_j > 0 on its input weights and 1/c_j on its output weights. relu(c z)/c = relu(z), so logits must not change; this is checked on all 961 pairs before any proxy is read.
- **Codec** = `experiment.compress()` unchanged: hidden-unit pruning plus per-matrix scalar quantization that must keep train/calibration agreement ≥ .99 and relative MSE ≤ .01.

## Packet 1. Retention/removal on a late, non-generalizing checkpoint

Checkpoint: `e1-co-add-w128-i30000-a1`, step 50,000 (final), exploratory, width 128, init seed 30000. It fits the training split (train CE 3.7e-6) and does not generalize (held-out audit accuracy 0, audit CE 10.5). Chosen because the assignment named it; its outcome was known (coordinator note) before analysis.

![retention](figures/e1-co-add-w128-i30000-a1-s50000/retention.png)

Look before reading: in the top-left panel, how different are the 15 frequencies from each other and from the step-0 marks? Then compare the coloured top-K lines with the pale random-subset dots of the same split. Does any small K reproduce the training fit? What happens to the held-out loss as more frequencies are kept?

Observations:

- Input energy is spread across all 15 frequencies: shares 0.052 to 0.077 (uniform 0.067). The top 4 (k = 8, 1, 11, 4) hold 30.0% of non-DC energy. At step 0 of the same run shares were 0.058 to 0.078 and the top 4 were k = 2, 14, 8, 9.
- Retaining the top K: train accuracy rises gradually, 0.42 at K=4 (train CE 2.14), 0.82 at K=8, 1.0 first at K=12. Train CE reaches the original 3.7e-6 only at K=15.
- Removing the top K: train accuracy stays 1.0 after removing 2 (CE 0.004), 0.95 after 4, 0.55 after 8.
- Top-K versus random K-subsets on train accuracy: retention top-4 is above 14 of 16 random 4-subsets (random range 0.27 to 0.43); removal top-8 is below all 16 (random 0.55 to 0.74). Top-K subsets carry somewhat more energy than random subsets of the same size (K=4: 0.30 vs 0.25 to 0.28), so this is not a matched comparison.
- Held-out audit: accuracy is at most 0.072 at every K in both directions (0.105 for the best random subset; chance 0.032). Audit CE rises monotonically from 3.54 (DC only) to 10.55 (original) as frequencies are retained, and the audit median margin is negative at every K.
- Full retention and empty removal reproduce the checkpoint's metrics to 5.3e-15 (reconstruction control).

Files: [retention.csv](figures/e1-co-add-w128-i30000-a1-s50000/retention.csv) (all three splits, accuracy, CE, margin min/median/p05, energy fraction), [random_subsets.csv](figures/e1-co-add-w128-i30000-a1-s50000/random_subsets.csv), [retention.svg](figures/e1-co-add-w128-i30000-a1-s50000/retention.svg), [summary.json](figures/e1-co-add-w128-i30000-a1-s50000/summary.json).

Missing checks: no generalizing checkpoint exists yet in e1, so there is no contrast between a memorizing and a generalizing model; one seed, one width, one step; no energy-matched null.

## Packet 2. Does a function-preserving rescaling change the frequency ranking or the codec?

Same checkpoint.

![rescaling](figures/e1-co-add-w128-i30000-a1-s50000/rescaling.png)

Look before reading: bottom-left first (is the function really unchanged?), then compare the ranking and codec panels against the size of the rescaling.

Observations:

- Targeted: multiplying the inputs of the 7 hidden units whose dominant frequency is k=5 (rank 5) by 1.0406, and their outputs by 1/1.0406, changes no logit by more than 3.6e-15 (max |logit| 22.2). k=5 replaces k=4 in the top 4. Retained-top-4 train accuracy changes from 0.421 to 0.448 and train CE from 2.14 to 2.00 for the same function. The codec stays at 95,496 bits (all 128 units, q=8). Parameter norm 60.338 to 60.361.
- Random log-normal gains c_j = exp(σ z_j), 16 draws per σ: logits unchanged to ≤ 4.8e-16 relative in all 64 draws. The top-4 set changed in 11/16 draws at σ=0.1 and 16/16 at σ = 0.25, 0.5, 1. Kendall tau versus the checkpoint ranking: 0.58 to 0.94 at σ=0.1, −0.22 to 0.50 at σ=1.
- Codec: unchanged at σ=0.1; 190,728 bits (q 8 → 16, twice as long) in 3/16 draws at σ=0.25, 7/16 at σ=0.5 and 16/16 at σ=1. No draw shortened it. The pruning order never changed: the codec's unit score (‖u_j‖+‖v_j‖)·‖w_j‖ is rescaling-invariant, so only quantization responds.
- Every random draw increased the parameter norm (60.34 → 60.7 to 61.1 at σ=0.1, up to 252 at σ=1).
- Filtering a fixed frequency set commutes with rescaling (unit test), so retention/removal results change only through which frequencies are selected.

Files: [rescaling_sweep.csv](figures/e1-co-add-w128-i30000-a1-s50000/rescaling_sweep.csv) (one row per draw), targeted result in [summary.json](figures/e1-co-add-w128-i30000-a1-s50000/summary.json) under `targeted_rescaling`, [rescaling.svg](figures/e1-co-add-w128-i30000-a1-s50000/rescaling.svg).

Missing checks: no normalization convention tried yet; the codec was not checked under rescaling on a generalizing model; the targeted gain grid has ratio 1.01, so 1.0406 is the smallest grid value that works, not an exact threshold.

## Method check (synthetic, not a trained model)

A hand-built network with known ground truth (`freq.synthetic`): frequencies 3, 7, 11, 14, 32 ReLUs each with random phases, input gains 1, .9, .8, .7, and each group's output weights a ridge fit of its own unit-amplitude target cos(2πk(a+b−c)/31). The four groups therefore contribute about equally to the function, but rank by input gain.

![synthetic retention](figures/synthetic/retention.png)

- Energy is only at the four built-in frequencies (others < 1e-20 of the maximum), ranked by gain.
- Accuracy is 1.0 with any single built-in frequency retained, while CE and margins keep improving until all four are kept: accuracy alone would call one frequency sufficient.
- Removing all four makes every hidden unit zero: logits 0, CE = ln 31 exactly.
- A 1.116 gain on the k=7 units moves k=7 to rank 1 without changing logits ([rescaling.png](figures/synthetic/rescaling.png)). Random rescalings did not change this network's codec (190,728 bits, q=16, in all 64 draws), so codec sensitivity depends on the model.

## Checks run

- Shared tests at the base commit: `uv run --locked python -m unittest test_exploration test_instrumentation`, 9 tests OK.
- Lane tests: `uv run --locked --group analysis python -m unittest discover -s research/mechanisms -v`, 8 tests OK (reconstruction, full removal, parity with `interventions()` at K=4, rescaling invariance of logits and pruning order, filter/rescaling commutation, synthetic ground truth, targeted reordering).
- Handoff: every file used was verified against the coordinator's `manifest.json` SHA-256; split hash equals the plan's `aa8f7504…`; target hash equals the manifest's `6f1576d4…`; recomputed step-50,000 accuracy, CE and margins equal `metrics.jsonl` exactly (max abs diff 0.0).
- Separate recomputation (`crosscheck.py`: cosine/sine projection instead of rfft, own forward pass and metrics, own rescaling): all 32 retention rows agree with `retention.csv` to 2.9e-14 on train/audit accuracy, CE and median margin; the targeted rescaling reproduces 7 units, logit change 3.6e-15 and top 4 {1, 5, 8, 11} ([crosscheck.json](figures/e1-co-add-w128-i30000-a1-s50000/crosscheck.json)). The codec itself was not independently recomputed: it is `experiment.compress()`, the object being measured.

## Provenance and reproduction

- Input: `research/coordinator/runs/e1-co-add-w128-i30000-a1/` read from `swarm/exploration` at `67a10d8d282fd1d57ae6b3aba5a992f8d040203d`; checkpoint_050000.npz SHA-256 `326cfc1e…11b4`, checkpoint_000000.npz for the step-0 reference; produced by code `ebd874e`.
- Analysis code: `research/mechanisms/freq.py` at the commit recorded in each `summary.json` (`analysis_code_commit`, clean tree), CPython 3.12.14, NumPy 2.3.5, Matplotlib 3.11.2 from the uv lock, one BLAS thread. Seeds: random subsets 33100, rescalings 33200, synthetic phases 34000.
- Elapsed numerical execution in this session: under 2 minutes. No training jobs were run.

```bash
uv sync --locked --group analysis
# The published run must be present; on swarm/exploration after 67a10d8 it is. From another branch:
#   git archive 67a10d8d282fd1d57ae6b3aba5a992f8d040203d research/coordinator/runs | tar -x
uv run --locked --group analysis python research/mechanisms/freq.py \
    --run research/coordinator/runs/e1-co-add-w128-i30000-a1 --source-commit 67a10d8d282fd1d57ae6b3aba5a992f8d040203d \
    --reference-step 0 --label "late checkpoint, fits train, does not generalize" \
    --out research/mechanisms/figures/e1-co-add-w128-i30000-a1-s50000
uv run --locked --group analysis python research/mechanisms/freq.py --synthetic --top 1 \
    --label "synthetic, hand-built, not trained" --out research/mechanisms/figures/synthetic
uv run --locked python research/mechanisms/crosscheck.py research/coordinator/runs/e1-co-add-w128-i30000-a1 50000 \
    research/mechanisms/figures/e1-co-add-w128-i30000-a1-s50000
```

## Next (not yet run)

1. The same two figures on an earlier checkpoint of this run (step 300, first train fit) and on the width-32 and width-512 final checkpoints; on a generalizing checkpoint as soon as the size/time lane produces one.
2. A normalization convention on analysis copies (per-unit balanced input/output norms), then repeat the invariance check.
3. Retrospective tracing (final-checkpoint frequencies scored at earlier steps), kept separate from checkpoint-local selection.
