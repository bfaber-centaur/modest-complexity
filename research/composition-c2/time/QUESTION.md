# c2 time lane: question and pre-registration (M0)

Charter: [../coordinator/charter/02-time.txt](../coordinator/charter/02-time.txt). Assignment, pin, seeds and budget: [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md) D3, D4, D6, D7.

Written before any c2 measurement by this lane. The only evidence used is r1's published records: [composition NOTE.md](../../regimes/composition/NOTE.md), [REVIEW.md](../../regimes/review/REVIEW.md) §1, and the published 0–10,000-update `metrics.jsonl` of the nine source runs.

## Question

Are the original RNN gaps persistent within a longer budget, or mainly differences in learning speed?

At r1's 10,000 updates, held-out final accuracy on the original task (`ops51000-ls51100`) was:

| arm | transition / readout | step weights | params | source runs (r1) | 51200 | 51201 | 51202 |
|---|---|---|---|---|---|---|---|
| `tt-inter` | tied / tied | 1 1 1 | 19,208 | `r1-comp-rnn-inter-s5120{0,1,2}-a1` | 1.000 | 0.961 | 1.000 |
| `tu-inter` | tied / per-step | 1 1 1 | 21,272 | `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1` (reviewer's arm) | 0.633 | 0.523 | 0.516 |
| `tt-final` | tied / tied | 0 0 1 | 19,208 | `r1-comp-rnn-final-s5120{0,1,2}-a1` | 0.273 | 0.312 | 0.320 |

Two gaps are in question.
- **Readout sharing.** `tt-inter` against `tu-inter`: the same initial function and supervision. The arms differ only in per-step readouts, which add 2,064 parameters.
- **Supervision.** `tt-inter` against `tt-final`: the same model and init. The arms differ only in step weights.

`tu-inter` and `tt-final` were both still rising at 10,000 updates. `tt-inter` is a continuing reference, not a known ceiling.

## Runs

**Nine continuations.** Each run resumes its recovered `ckpt-10000` with weights, both Adam moments and step count (`common/c2train.py --resume … --resume-sha256 …`). It then continues to 50,000 total updates.
- **Recipe.** r1's, unchanged: width 128, full-batch Adam (lr 1e-3, b1 0.9, b2 0.999, eps 1e-8), decoupled weight decay 0.1 on every parameter, and 384 training rows.
- **Logging and checkpoints.** `--diag`, logged every 50 updates. Checkpoints at 20k, 30k, 40k and 50k.
- **Hashes.** The SHA-256 of each source file is the one in [../recovered/RECOVERY.json](../recovered/RECOVERY.json).

| run ID | resumes | scheme flags | step weights |
|---|---|---|---|
| `c2-time-tt-inter-s5120{0,1,2}-a1` | `r1-comp-rnn-inter-s5120{0,1,2}-a1/ckpt-10000.npz` | (defaults) | 1 1 1 |
| `c2-time-tu-inter-s5120{0,1,2}-a1` | `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1/ckpt-10000.npz` | `--readout untied` | 1 1 1 |
| `c2-time-tt-final-s5120{0,1,2}-a1` | `r1-comp-rnn-final-s5120{0,1,2}-a1/ckpt-10000.npz` | (defaults) | 0 0 1 |

Each continuation is a continuation of its original run (D2), not a new seed and not a replay. The three init seeds are exploratory coverage, not statistical power.

**Full trajectory.** A run's 0–50,000 trajectory is r1's published 0–10,000 record joined to the continuation's 10,050–50,000 record.
- **Joining check.** The continuation's own step-10,000 row must equal r1's published step-10,000 row in every r1 key, as exact floats and the prediction string.
- **On a mismatch.** That run is reported as broken and not joined.

**On-host continuation check, which is also the calibration.** It runs before the main continuations, as the charter asks. `c2-co-verify-*` already showed bitwise continuation on another host.
- **Runs.** `c2-time-verify-{tt-inter,tu-inter,tt-final}-s51200-a1` each resume the recovered `ckpt-1000` of seed 51200 to 3,000 updates with `--diag`, two jobs at a time.
- **Pass.** Every logged r1 key from 1,000 to 3,000 equals the published r1 record.
- **Timing.** The same runs give ms per update on this host for both schemes.

## Measures

All held-out measures use the 128 held-out rows: 16 programs × 8 starts. Training measures use the 384 training rows.

1. **Accuracy and loss.** Train and held-out final accuracy and mean cross-entropy. For the `inter` arms, also the step-1 and step-2 heads. Held-out step-k accuracy is not an unseen-prefix test (r1 C1.1), so it is reported only as a training-like quantity.
2. **N and ¬N subsets.** Held-out final accuracy on the 86 rows that a one-op-slot training neighbour's label answers (N), and on the 42 rows it does not (¬N), using `c2task.neighbour_answerable`.
   - For 10,050–50,000 the values are taken from `held_subset`.
   - For 0–10,000 they are recomputed from r1's `held_final_pred`, and must agree with `held_subset` at 10,000.
3. **Time to threshold.**
   - **Definition.** The first logged update t at which held-out final accuracy is ≥ 0.95 (≥ 122 of 128 rows) at t and at the next four logged evaluations (t, t+50, …, t+200).
   - **Resolution.** Logging is every 50 updates, so times are known to 50 updates, and a crossing before 10,000 uses r1's record at the same resolution.
   - **Censoring.** A run with no such window by 50,000 is censored at 50,000 and reported as "> 50,000". A window that starts after 49,800 cannot complete and is also censored.
   - **After the crossing.** Whether the run stays ≥ 0.95 to 50,000, and its minimum after the crossing.
   - **Secondary.** The same rule on ¬N, at ≥ 0.95, which is ≥ 40 of 42 rows.
4. **Optimization diagnostics.** These come from `--diag` at every logged update of the continuation. They are not available before 10,000, because r1 did not log them.
   - global gradient L2 norm;
   - global update L2 norm, including decay;
   - Adam √v̂: median, max, and the fraction of coordinates below eps = 1e-8;
   - train final-step margin (correct logit minus the largest other): min and median.
5. **Function change against parameter change.** Diagnostics alone do not separate slow numerical drift from a still-changing function, so two more measures are used.
   - **Churn.** The fraction of held-out predictions that change between consecutive logged evaluations.
   - **Checkpoint distances.** At each checkpoint (10k, 20k, …, 50k): the parameter norm, and the distance to the previous checkpoint, set against the summed update norms between them.
   - **Reading.** A stable function is one with no churn and flat held-out CE, even while parameters still move.
6. **Neighbour compatibility of wrong predictions.** This is reported at 10k, 20k, 30k, 40k and 50k, and at every 1,000 updates in the CSV. For each wrong held-out prediction:
   - (a) whether it equals the final label of any one-op-slot training neighbour;
   - (b) whether it equals the label of a same-prefix neighbour, where only op 3 differs.
   - **Uniform baselines.** Each is compared with the rate expected if wrong answers were uniform over the 7 wrong symbols, computed per row from the neighbours' labels.
   - **Limit.** Compatibility does not show that the network copies a neighbour.

## Predictions

These were written after reading the published 0–10,000 r1 trajectories. Examples of held-out final at 2k, 5k, 8k and 10k for seed 51200: `tt-inter` 0.45, 0.73, 0.97, 1.00; `tu-inter` 0.28, 0.37, 0.53, 0.63; `tt-final` 0.20, 0.23, 0.26, 0.27.

- **`tt-inter`.**
  - Seeds 51200 and 51202 meet the threshold before 10,000. Seed 51201 meets it by 20,000.
  - All three stay ≥ 0.95 to 50,000.
  - ¬N reaches ≥ 0.95 in all three.
- **`tu-inter`: the readout gap is mainly speed within this budget.**
  - All three exceed 0.80 by 50,000, and at least two of the three meet the threshold.
  - Its rate of rise from 5k to 10k (about +0.05 per 1,000 updates for seed 51200) slows as accuracy rises. So any crossing comes after 25,000.
  - Low confidence. The alternative is a plateau between 0.6 and 0.9, where the per-step readouts settle into partly position-specific codes.
- **`tt-final`: the supervision gap persists within 50,000.**
  - No seed meets the threshold, so all three are censored.
  - Held-out final at 50,000 is between 0.30 and 0.60, with ¬N accuracy below 0.25 in every seed.
  - It is still rising slowly at 50,000.
  - Moderate confidence. r1 rose by only about 0.08 from 2k to 10k.
- **Errors.** In `tt-final` and `tu-inter`, wrong predictions stay far above the uniform baseline in neighbour compatibility (a) at every checkpoint, as in r1's MLP.
- **Diagnostics.**
  - Once the training set is fit, gradient norm falls, and the decay equilibrium holds the train margin roughly constant. The margin does not grow without bound, because decoupled decay limits weight growth.
  - The fraction of √v̂ below eps stays ≤ 0.01. Any rise comes from coordinates with exactly zero gradient, such as units dead on every training row. Those coordinates then receive decay only.

## Interpretation rules, fixed in advance

These apply per arm, at the calibrated endpoint of 50,000 updates under this recipe. Revised as R1 below.
- **Speed within 50k.**
  - All three seeds meet the threshold.
  - In every seed, ¬N accuracy is ≥ 0.90 at each of the last five logged evaluations, 49,800–50,000.
- **Gap persists within 50k.** In every seed, all four of the following hold. Together the last three are measure 5's stable function, with tolerances.
  - The seed does not meet the threshold.
  - Held-out final accuracy changes by < 0.02 between 40,000 and 50,000.
  - The held-out predictions at 50,000 differ from those at 40,000 in at most 2 of 128 rows.
  - No held-out prediction changes over the last 1,000 updates (the last 21 logged evaluations, 49,000–50,000). Added in R2.
  - Held-out final CE changes by less than 10% of its 40,000 value between 40,000 and 50,000.
- **Unresolved.** Anything else, for example still rising at 50,000, or mixed outcomes across seeds.

Neither endpoint proves an asymptote. A plateau at 50,000 is a finite-budget observation under one optimizer, not a necessity result. Reaching the threshold shows that this recipe can get there, not that it must.

## Controls and checks

- **Pin.** The shared-code pin `b3e8132` is unedited. The 10 shared tests run in their own process on this host.
- **Before main runs.** The on-host continuation check above, and the joining check on every continuation.
- **Hashes.** Source checkpoints are cited by SHA-256. Resume refuses a mismatch.
- **Lane tests.** A small `test_time.py`, run in its own process, checks the threshold and censoring rule and the neighbour-compatibility counts on hand-made inputs.
- **No optimizer change** in the main comparison.

## First figure (planned)

`figures/time_main.png`, with SVG and CSV beside it. The x-axis is update count from 0 to 50,000 on a linear scale, with a marker at 10,000 where r1's record ends and the continuation begins. Colour shows the arm; each of the three seeds is drawn as its own line.

1. **Top, full width.** Held-out final accuracy for all nine runs, with train final accuracy as thin lines and the 0.95 threshold drawn. Each crossing is marked, and censored runs are marked at 50,000.
2. **Accuracy by subset.** Left: held-out ¬N accuracy (42 rows). Right: held-out N accuracy (86 rows).
3. **Loss and margin.** Left: held-out and train final CE, on a log scale. Right: train final margin, min and median.
4. **Optimizer diagnostics, from 10,000 on.** Left: gradient norm and update norm, on a log scale. Right: median √v̂ with eps drawn, and the fraction below eps.

## Follow-up (to be appended before running)

There is one follow-up, chosen after the nine continuations and the first figure, and approved by the coordinator before launch.
- **Default.** If an arm is unresolved (still rising below threshold at 50,000), extend its three seeds from `ckpt-50000` to 100,000 with the same recipe.
- **Alternative.** If an arm plateaus and the diagnostics point at the optimizer, branch one narrowly specified optimizer change from `ckpt-50000`, together with a matched unchanged continuation over the same updates. Such signs would be update norm dominated by decay, √v̂ near eps, or margin pinned at the decay equilibrium. A changed optimizer changes the trajectory and may change implicit bias, so it cannot isolate a pure speed effect.

## Budget

- **Cap.** 40 minutes of elapsed wall time with at least one job running (D6), with `--workers 2`.
- **Expected cost of the nine.** 9 × 40,000 updates. At the coordinator host's 7.5 ms per update that is about 25 minutes, with two at a time. The r1 host ran at 3.9 ms, which would make it about 13 minutes.
- **Timeouts.** Set from this host's calibration at about 3× the expected run time.
- **Reporting.** The coordinator is told at 80% of the budget, and capped or failed runs stay in the registry.

## Deliverables

- QUESTION.md (this file);
- `plan.json` and `registry.jsonl`;
- `time_analyze.py`, which publishes, measures and draws the figure, with `test_time.py`;
- `runs/<ID>/`, holding the small run files, `MANIFEST.json`, and the committed `ckpt-50000` of all nine runs (the 10k states are already in `../recovered/`), plus the 20k–40k checkpoints for the mechanisms lane;
- the primary figure as PNG, SVG and CSV;
- README.md, NOTE.md and a reviewer handoff list.

## Revisions

**R1. Written 2026-10-22 04:12Z, while the nine continuations were running and before any of their records had been read.** These are the reviewer's wording notes T0.1 and T0.2 on b7d1ab6, relayed by the coordinator.
- **T0.1.** The "gap persists" rule gets tolerances. "Churn ≈ 0" becomes at most 2 of 128 held-out predictions differing between 40,000 and 50,000. The rule also requires flat held-out CE: a change of less than 10% over the same interval. Both match measure 5.
- **T0.2.** The ¬N ≥ 0.90 condition of the "speed" rule now has a time: each of the last five logged evaluations.
- **Checkpoints.** `ckpt-40000` is committed with `ckpt-50000`, as the reviewer's pre-committed counterexamples use both. The 20k and 30k checkpoints are committed too, as already planned.

**R2. Written 2026-10-22 04:21Z, again before any continuation record had been read.** The reviewer's own suggestion for T0.1 (REVIEW.md §2) arrived after R1.
- **Prediction stability.** Its condition that no held-out prediction changes over the last 1,000 updates is added to the "gap persists" rule, alongside R1's 40k-to-50k tolerance.
- **CE subset.** Its ¬N cross-entropy is not logged; `c2train.py` logs CE over all 128 held-out rows. So R1's held-out final CE tolerance stands in for it.
