# R3 training history and recovery: question and design (M0)

Lane: `research/regimes/history/`, run IDs `r1-hist-…`, seeds 53000–53999, session label `r1-hist-cloud-1`. Written before any lane run.

## Question

A small ReLU network learns task A, then task B, then returns to A. What part of A survives B, and what determines how fast A comes back? Specifically:
1. How much of A's function, measured through A's original output head, is lost after B, as a function of B's duration and of whether B is structurally related to A?
2. Can a newly trained linear readout on the frozen post-B trunk still recover A on held-out inputs, and by more than the same readout on a trunk that never saw A?
3. How many updates (and examples) does it take to regain A, compared with learning A from scratch from the same initialization?

## Task, split, architecture, objective

**Domain.** All 1,024 inputs x ∈ {0,1}^10, with input index i giving bit x_k = (i >> k) & 1.

**Tasks.** Each is a 3-bit parity, so all are balanced over the full domain, and every pair is uncorrelated (pairwise label agreement 1/2 over the domain):

| name | label | head | role |
|---|---|---|---|
| `A` | x0 ⊕ x1 ⊕ x2 | A | task A |
| `Brel` | x0 ⊕ x1 ⊕ x3 | B | related B: shares the subfunction z = x0 ⊕ x1 with A |
| `Bunrel` | x4 ⊕ x5 ⊕ x6 | B | unrelated B: disjoint bits |
| `notA` | 1 − A | A | conflicting mapping, separate experiment only |

Bits x7–x9 are irrelevant to every task. `A`, `Brel` and `Bunrel` are images of one another under input-bit permutations, and the initialization distribution is permutation-symmetric, so their from-scratch difficulty is matched in distribution. The scratch runs below check this with the actual seeds. "Related" is defined by task structure. Whether the network actually computes and shares z is not assumed.

**Split.** One split for every run: 256 training inputs drawn uniformly without replacement from the 1,024 (data seed 53100), and the other 768 held out. A and B use the same training inputs. The README reports actual label balance and 3-bit pattern coverage per task in each part. The split is fixed across init seeds, so the three seeds vary initialization only, and dataset randomness is not sampled.

**Network.** `nets.init(10, (128,), {A: 2, B: 2}, seed)`: one ReLU layer of 128 units (the trunk) and two 2-class softmax heads. Task identity is explicit through the heads, with no task-ID input. Init seeds 53200, 53201 and 53202 are paired across every condition.

**Objective.** Full-batch Adam, lr 1e-3, default betas, wd = 0. The loss is the mean cross-entropy of the active task on its own head over the 256 training rows. In each phase only W1, b1 and the active task's head are trainable. With wd = 0, B phases therefore leave the A head bitwise unchanged, so any change in A through its original head is trunk drift. Each run asserts this.

**Optimizer.** It is reset (`nets.adam_state`) at the start of every phase in every arm, including continued-A and the zero-duration arm, so the reset is common to all arms. Preserving optimizer state is a possible follow-up only.

## Phase protocol and arms

For each init seed s:

| arm | run ID pattern | starts from | phases |
|---|---|---|---|
| phase A = fresh-A reference | `r1-hist-A-s{s}-a1` | init | A for T_A = 2000; saves `ckA` |
| scratch B (difficulty check, history-free readout reference) | `r1-hist-scratch-{Brel,Bunrel}-s{s}-a1` | init | B for 10000; readouts and checkpoints at 100, 1000, 10000 |
| related B | `r1-hist-rel-d{d}-s{s}-a1` | `ckA` | Brel for d, then A for T_R = 4000 |
| unrelated B | `r1-hist-unrel-d{d}-s{s}-a1` | `ckA` | Bunrel for d, then A for T_R |
| continued-A | `r1-hist-contA-d{d}-s{s}-a1` | `ckA` | A for d, then A for T_R |
| zero duration | `r1-hist-d0-s{s}-a1` | `ckA` | A for T_R (common d = 0 point of every arm) |

Here d ∈ {100, 1000, 10000}, about one and two orders of magnitude apart. Each branch records `ckA`'s SHA-256 and checks that the loaded weights hash to the values saved by the A run. The fresh-A reference for seed s is that seed's own phase-A trajectory: same initialization, same data, optimizer from zero. Continued-A has two resets (at the start of each of its phases), as the B arms do.

**Separate experiment: conflicting mapping without task identifiers.** `r1-hist-conflict-d{d}-s{s}-a1` starts from `ckA` and trains `notA` through the A head (the shared head) for d, then returns to A for T_R. It has its own IDs and figure row, and it is never pooled with the coexisting-task arms. A linear readout cannot separate "A kept" from "notA learned" here, because notA needs exactly the features A needs. Its informative measures are the relearning dynamics and the functional change.

**Exposure.** Every update is full-batch over the same 256 inputs, so examples = 256 × updates in every phase. Per-task exposure in a branch run: A gets 256 × (2000 + 4000), and B gets 256 × d. Continued-A gets 256 × d extra A exposure. It controls for elapsed updates and resets, not for A exposure.

## Measurements

All are logged on the complete domain, split into train and held-out rows, at phase step 0 and then every update up to step 100, every 10 up to 1,000 and every 100 beyond (resolution 1, 10 or 100 updates):
- train and held-out accuracy and loss of head A on A, and of head B on Brel and on Bunrel, in every phase of every run;
- relative trunk drift ‖θ_trunk − θ_trunk(run start)‖ / ‖θ_trunk(run start)‖.

**Criterion.** Held-out A accuracy ≥ 0.95, through the A head, at the first logged step that reaches it. It is the same in every arm. Relearning cost is the return-phase step at which it is first met, in updates and in examples (×256). A return phase that never meets it within T_R is censored at 4000, never assigned a time. Savings = relearning updates / the same seed's phase-A updates to criterion. B acquisition uses the same criterion on head B.

**Frozen-feature readout.** At every phase end, at init, and at the scratch-B points:
- Take trunk activations on all 1,024 inputs and standardize each unit with training-row mean and standard deviation. A unit with zero training variance is set to 0 and counted as dead.
- Fit a fresh linear softmax readout, `nets.init(128, (), {R: 2}, 53300)`, with Adam lr 1e-2 for a fixed 2,000 full-batch updates on A's training labels.
- Report train and held-out accuracy at 200 and 2,000 updates.
- Row-shuffled null: the same fit with A's training labels permuted across training rows (seed 53301), to show what the readout can memorize.

The comparison set for each init seed is the readout on: untrained features (init), end-of-A features (`ckA`), end-of-B features of each branch, and the history-free reference, a scratch-B trunk after the same number of B updates d.

**Functional change.** Over all 1,024 inputs, against the run's start checkpoint (`ckA` for branches), at every phase end:
- the fraction of inputs whose A-head prediction changed;
- the RMS change in the A-head margin (logit₁ − logit₀), raw and divided by `ckA`'s RMS margin;
- linear CKA between hidden activations;
- the same prediction-change fraction for head B between the end of B and the end of the return phase.

## Plausible outcomes and what distinguishes them

The thresholds below are descriptive guides, not tests. Each is judged per seed, and all three seeds are shown.

1. **A survives B.** Original-head held-out A stays ≥ 0.95 at the end of B, and relearning cost is about 0.
2. **Functionally lost, features kept (readout misaligned).** Original-head held-out A falls below 0.8 at the end of B, but the readout on end-of-B features reaches ≥ 0.95 held-out and beats the history-free scratch-B readout at the same d by ≥ 0.10.
3. **Overwritten as far as a linear readout can see.** The end-of-B readout is within 0.05 of the scratch-B reference.
4. **Neither outcome 2 nor outcome 1, yet relearning is still faster than fresh-A** (savings ≤ 0.5). Something not linearly decodable survived. By design one candidate is the A head itself, which B never touches, so savings could be head alignment rather than retained features. This run set cannot separate those. A re-initialized A head at return is the follow-up that would.
5. **Related versus unrelated B.** Intuition says related B protects A, because the shared z survives. The opposite is just as plausible: units already carrying x0 and x1 are the ones most useful to Brel, so Brel may recruit and rewrite exactly A's units, swapping their x2 dependence for x3. Then forgetting through the A head would be larger for related B at matched d, while relearning might still be faster. The distinguishing observations are original-head A at end of B, relative trunk drift and A-head prediction change, all compared at matched d.

**My predictions, recorded before running.**
- Original-head A forgetting grows with d for both B tasks. At d = 100, B may not yet be learned.
- Related B causes at least as much functional forgetting as unrelated B (outcome 5's counterintuitive branch), with low confidence, about 55/45.
- For unrelated B, the end-of-B readout recovers A above the scratch-B reference at d ≤ 1000 (outcome 2 or 1). I make no confident prediction at d = 10000.
- Relearning is faster than fresh-A in every branch.
- Continued-A shows no forgetting. The d = 0 arm shows only a small reset transient.
- Conflict: notA is learned quickly, mostly by reorienting the A head and trunk together. The end-of-notA readout is high and uninformative, and the return to A is much faster than fresh-A.

## Controls and known confounders

**Controls.**
- Continued-A: elapsed updates and resets without B.
- The d = 0 arm: reset only.
- Fresh-A: the phase-A trajectory of the same seed.
- Scratch B: difficulty match, and the history-free readout reference.
- Untrained-feature and shuffled-label readouts.
- The conflict experiment is kept separate.

**Known confounders.**
- The preserved A head (by design) can produce savings without retained trunk features.
- An Adam reset makes the first steps about lr in size along sign(g), even near a minimum. The d = 0 arm measures this transient.
- Long phases grow weights and logit scale, so loss and accuracy can diverge. Readout features are standardized for that reason, but fit budget and lr still matter, which is why both the 200- and 2,000-update readouts are reported.
- With 128 features and 256 training rows the readout can memorize. Held-out evaluation and the null bound this.
- Branches from one `ckA` are dependent, so there are three independent units (init seeds) per comparison, on one data split.
- Continued-A is matched in updates, not in A exposure.

## Checks before the main runs (M1)

- **Oracle.** An independent enumeration with `itertools.product` and string bit counts, compared with the vectorized labels on all 1,024 inputs, including balance and pairwise agreement of the tasks.
- **Split.** Disjoint parts covering the domain, sizes 256/768, no duplicate rows, a recorded hash, and per-task label balance and 3-bit pattern coverage.
- **Gradients.** A finite-difference check of the depth-0 readout through `nets.loss_grad`, plus the shared `test_nets.py`.
- **Branching.** The A head is bitwise unchanged after a B phase, the optimizer step count is 0 at each phase start, and loaded `ckA` weights hash-match.
- **Positive control.** Fresh-A (the phase-A runs) and scratch B reach ≥ 0.95 held-out. That validates trainability, not just the task.
- **Calibration.** The first runs on this host record wall and CPU time, peak RSS, versions and thread settings, and set the timeouts for the rest (about 3× expected, at most 900 s).

## First figure

A and B accuracy, train (dashed) and held-out (solid), aligned at the A→B and B→A switches:
- one panel per condition: rows related B, unrelated B, continued-A and (visibly separated) conflict; columns d = 100, 1000, 10000;
- three seeds as individual traces;
- the same seed's fresh-A held-out curve drawn in grey in each return segment, and the d = 0 return as a dotted reference;
- censored relearning marked;
- chance (0.5) drawn.

Phase A is drawn on a linear segment. B and return phases are drawn on log segments of steps since the switch, with step 0 placed explicitly. A second, summary figure shows end-of-B original-head A, the readout comparisons and relearning savings per seed.

## Budget and stopping rules

**Budget.**
- Total cap 25 min of elapsed numerical execution, tallied in README.md.
- Estimated about 355k full-batch updates for the initial comparison and the M1 runs together, about 3–6 min at two workers, to be confirmed by calibration.
- Plan caps: M1 ≤ 4 min, initial comparison ≤ 10 min, follow-up ≤ 6 min.
- Readouts and functional measures run inside the jobs.

**Positive-control failure.** If any fresh-A seed fails to reach ≥ 0.95 held-out by 2000 updates, I will change lr to 3e-3 first, and then the training set to 512, each as a new, labelled attempt with the old runs kept. There will be at most these two changes before I stop and report.

**Difficulty mismatch.** If scratch Brel and Bunrel median times to criterion differ by more than 2×, I will record it and keep the design. They are symmetric by construction, so a difference reflects seed variability.

**Scope.** The initial comparison has no extra seeds, durations or widths. Capped runs are kept and rerun at most once, under a new attempt ID.

**Stopping.** I stop and ask the coordinator before exceeding 22 min.

## Follow-up (M3, written after M2 at 9d758e1 and before any follow-up run)

The candidates were (a) the return phase with a re-initialized A head, (b) optimizer state preserved across boundaries, and (c) a task-ID input bit with one shared head. I chose (a).

**Motivation.** At M2 every B branch relearned A in 37–560 updates, against 540–700 for fresh-A, while the frozen-feature readout recovered 0.91–1.00 held-out from end-of-B trunks. Because B never touches the A head, those savings could be carried entirely by the preserved head (outcome 4's confound). Resetting head A to its own initial draw (`nets.init` with the run's init seed, so the head is identical to fresh-A's) makes the trunk the only thing that differs from fresh-A.

**Arms.** Init seeds 53200–53202. Each arm is one return phase of A for 4000 updates with a fresh optimizer. Everything else is as before (lr 1e-3, wd 0, same split, same criterion).

| arm | run ID | trunk (and head B) loaded from | head A |
|---|---|---|---|
| reset after related / unrelated B | `r1-hist-fu-reset-{rel,unrel}-d{1000,10000}-s{s}-a1` | the end-of-B checkpoint of the M2 branch with that d | reset to its init draw |
| B-only trunk | `r1-hist-fu-reset-scratch-{Brel,Bunrel}-d{1000,10000}-s{s}-a1` | the scratch-B checkpoint after the same d | reset (already its init draw; this is a checked no-op) |
| end-of-A trunk | `r1-hist-fu-reset-ckA-s{s}-a1` | ckA | reset to its init draw |

That is 27 runs of 4000 updates, an estimated 1–1.5 min at two workers. d = 100 is left out because A was not forgotten there. Results stay in their own figure and table, separate from the initial comparison.

**Predictions, recorded now.** Here t(·) is the number of return updates to held-out A ≥ 0.95, and fresh-A is 540–700.
- With the head reset, t(end-of-B trunk) is below fresh-A in every seed and at most 0.5× fresh-A in most cases, because the readout already finds A in these trunks. It is above the preserved-head relearning time of the same branch, because the old head carried some alignment.
- t(B-only trunk) is close to fresh-A. It is perhaps somewhat faster for Brel, whose B-only readout reached 0.65–0.79.
- t(end-of-A trunk, head reset) is the smallest of the reset arms.

**What would change the reading.**
- If t(end-of-B, reset) ≈ t(B-only), the M2 savings were carried by the preserved head, and linear decodability of A in the trunk did not translate into faster learning at this lr.
- If t(end-of-B, reset) ≈ the preserved-head time, the head contributed little.

**Budget.** At most 6 min. There are no further follow-ups after this one.

## Literature

Kirkpatrick et al. 2017, "Overcoming catastrophic forgetting in neural networks", arXiv:1612.00796. I read the abstract page in this session; the full body was not retrieved. The abstract frames forgetting as the loss of weights important to earlier tasks, which EWC protects by slowing their learning. Not re-read here, from memory and so unverified: the experiments use permuted MNIST with a shared output and Atari with task-specific gains and biases.

The bearing on this design is that separate heads with wd = 0 make A's original-head loss pure trunk drift, the quantity EWC constrains. This lane does not implement EWC.
