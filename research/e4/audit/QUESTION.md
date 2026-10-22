# e4 audit: questions and decision rules (written 2026-10-22, about 02:20 UTC, before any e4 measurement)

Session `e4-au-cloud-1`, branch `claude/project-thread-id1tmm`, base `swarm/exploration` `7c4389c` (D11 on the `52e7879` pin; shared code unchanged since `72cfb5d`). Budget 25 min of elapsed numerical execution (D9), at most two processes with one BLAS thread each.

**What I had seen before writing this.**
- The charter, DECISIONS D1–D11 and LITERATURE.md.
- The mechanisms QUESTION.md (`d584af7`), and the dynamics QUESTION.md with its addendum (`e5fd1c2`), including the numbers it measured on the existing wd 0 and wd 1 runs.
- The e3 mechanisms README and INTERPRETATION, and my own e2 and e3 audits. From e2 cycle 3 I know the random-feature start's update-0 common accuracy: 0.195 / 0.143 / 0.277 at ridge r = 1e-6, and 0.234 / 0.145 / 0.300 at r = 1e-2.

**What I had not seen.** No e4 run output, no lane figure, and no random-feature trajectory; none exists yet.

**The main audit question.** Are the lanes measuring a computation that is already present, or does a decomposition or intervention supply the structure it then finds?

## 1. Control C1: random-feature starts through the B protocol (D11, six reserved runs)

**Prior observations.**
- B (e3 `filt`, wd 1), on the 483 common pairs, at update 0 and update 100 (e3 README):
  - i30000: 0.431 → 0.735 (rise R_B = 0.304);
  - i30001: 0.540 → 0.789 (R_B = 0.249);
  - i30002: 0.580 → 0.863 (R_B = 0.283).
- B at wd 1 falls below its update-0 level at 600 / 470 / 470 updates and below 0.10 at 1,680 / 1,540 / 1,800. B at wd 0 is censored at both thresholds (dynamics `e5fd1c2`).
- The e3 sham C never leaves chance. Random features with a ridge readout reach 0.14–0.30 under the filter, with no training (e2 audit cycle 3).

**Construction** (`rf_starts.py`, written after this file). For each init seed 30000, 30001, 30002:
1. U and V are the step-0 checkpoint of B's source run. `load_run` checks the manifest hashes, the split hash and the step-0 logged metrics.
2. W is the closed-form ridge readout on relu(U[a] + V[b]) over the 299 training pairs, fitted to one-hot a+b minus 1/p, as in e2. λ = r · trace(G)/128 with r = 1e-6.
   - Why this r: it is e2's first planned value and the closer of the two to plain least squares.
   - It gives the lower update-0 held-out accuracy in two of three seeds, so it was not picked for held-out performance.
3. The e2 `filter_units(w, dominant(w)[0])` is applied, exactly as `edits.py` built B.
4. U and V are scaled by one factor, so that ‖U‖² + ‖V‖² equals B's, and W by another, so that ‖W‖ equals B's. With no biases this changes no argmax, rank or σ-margin; the script checks this at update 0.
   - Why scale: Adam's steps are about lr per coordinate, so B's step size relative to its weights is matched only at B's norms.

The script reports, at update 0:
- the reproduction of e2's r = 1e-6 values (it must give 0.195 / 0.143 / 0.277 to three decimals);
- the train fit, the kept energies and the gate flips against the unfiltered ridge network.

**Alternatives.**
- **G (generic).** The filter plus fitting the 299 pairs supplies the peak computation. The random-feature start rises as B does, collapses at wd 1 and holds at wd 0. Any mechanism found between B's peak and its collapse then describes structure that the filter and the fit supply, not something the failed runs contain.
- **F (the failed runs contribute).** The random-feature start barely rises. Something in the trained features (which frequencies the units use, their amplitudes and gates, their phases) matters beyond the readout fit. This control cannot say which.
- Intermediate outcomes are possible.

**Decision rule, per seed, fixed now.** It uses the coordinator's two edits (D11).
- **Rise.** R_rf = common accuracy at update 100 minus at update 0, in the wd 1 run. It is compared with R_B for the same seed, on the same 483 pairs, at the same update.
  - R_rf ≥ R_B / 2: the rise is generic.
  - 0.1 ≤ R_rf < R_B / 2: inconclusive.
  - R_rf < 0.1: the rise is specific to B.
- **Collapse at wd 1.** These are the dynamics lane's declared definitions:
  - own held-out accuracy (cal ∪ audit, every 10 updates);
  - the peak is the maximum over updates 0–1,000;
  - T_θ is the first logged update after the peak below θ, for θ = the run's update-0 accuracy and θ = 0.10;
  - a run that has not crossed by 20,000 is censored.
  "Collapses" means it crosses both thresholds.
- **Hold at wd 0.** The wd 0 run is censored at θ = its update-0 accuracy.
- **Verdict.**
  - **G is SUPPORTED LOCALLY** if all three seeds have a generic rise, all three wd 1 runs collapse, and all three wd 0 runs hold.
  - **F is SUPPORTED LOCALLY** if all three seeds have R_rf < 0.1.
  - Anything else is **AMBIGUOUS**, reported per seed and per component (rise, collapse, hold), with no pooling.
- **Reported alongside** but not used in the verdict:
  - the update-0 → 100 rise on own held-out pairs;
  - the update-0 → t_fit rise, where t_fit is the first update with train accuracy 1 (label-free; the launcher checkpoints the first update with train accuracy ≥ .99);
  - train accuracy and cross-entropy, P(a+b over N), P(a+b over S), P(S over N) and the σ-margin at every checkpoint.

**Limits stated now.**
- Three init seeds and one split. The six runs are paired by start file, not independent of B.
- The random features keep less of their energy under the filter than B does (e2: concentration 0.17 against 0.40–0.44). After rescaling the norms match, but the share of each unit in its DC and in its frequency does not. That works against G, so a G result is conservative and an F result is weaker.
- Fresh Adam at both starts. The random-feature start begins less fit than B (e2: train accuracy 0.16–0.29 after the filter, against 0.69–0.79), so its rise may take longer than 100 updates. That is why the t_fit rise is reported.

## 2. Audit of the mechanisms decomposition (design `d584af7`)

These checks are written against the lane's design. They are run on the lane's code and output once it is pushed, and on my own reimplementation from its definitions.

| # | check | PASS if |
|---|---|---|
| M1 | Reconstruction identity | My implementation gives label offset μ·W + reference (h − μ)·W^ref + off-reference (h − μ)·W^off = the centred logits on the 483 pairs at every checkpoint used, with maximum absolute error ≤ 1e-10 × max \|centred logit\|. The lane's channels agree with mine to the same tolerance. The label-axis DC of W is a per-pair constant, so it drops exactly from every centred quantity and linear margin |
| M2 | Basis and labels | k_j is read from update-0 weights only, with no labels, pairs or a+b−c, and equals e2 `dominant()` on the start file. μ uses training inputs only. Any quantity that uses a+b−c is listed as task-informed |
| M3 | Invariances | Accuracy, ranks, the P measures and the σ-margin are unchanged (to 1e-12) when every logit is multiplied by a positive factor, when a per-pair constant is added, and when (U, V) and W are scaled separately. Linear margins in logits are unchanged by the per-pair constant but scale with the factor. I report which of the lane's measures are which |
| M4 | Decay is a uniform factor | `experiment.adam` multiplies U, V and W by (1 − lr·wd) on every update, update 1 after the reset included. One shared update with zero gradient gives exactly (1 − lr·wd)·w. So the start weights' surviving factor is ρ_t = (1 − lr·wd)^t and −log ρ_t is the dynamics lane's exposure E(t) |

**Readings to check.** Each is SUPPORTED LOCALLY, AMBIGUOUS or CONTRADICTED.
- **R1. Absolute size is not loss.** The network is 2-homogeneous, so shrinking all weights changes no argmax. A claim that "the reference channel shrinks" (E1) is read in two ways:
  - against the pure-decay benchmark ρ_t², which is what a logit channel built only from decayed start weights would keep: a ratio above 1 means the channel was replenished, about 1 means passive decay, below 1 means it was actively removed;
  - by the channel's share of the total linear margin, which is scale-free and is what accuracy depends on.
  E1 and E2 are distinguishable only in absolute units. If the lane's evidence for one over the other is absolute only, the reading is AMBIGUOUS.
- **R2. Off-reference starts at zero by construction.** At update 0 the filter leaves each W row at k_j only, so the off-reference channel is exactly zero. Training at any wd adds to it. So its growth at wd 1 is evidence for E2 only relative to its growth at wd 0 at matched updates, and only if it is larger at wd 1.
- **R3. Channels are per unit, not per frequency.** Unit i's off-reference part can sit at another unit's reference frequency and carry a+b. I split the off-reference channel's held-out linear margin into the part at frequencies in K = {k_j} and the part outside K. If the part inside K carries most of the off-reference channel's positive a+b margin, the channel labels confound "competitor" with "useful signal moved to another unit".
- **R4. Masking needs an argmax check, and h is held fixed.**
  - Linear margins against the mean over S or N add across channels, but accuracy is max-based. So "reference still favours a+b, outweighed by the rest" also needs:
    - the accuracy of the reference channel alone (sufficiency);
    - the accuracy of the total with the off-reference channel removed (ablation).
  - Every channel uses the current h, so a change in a channel mixes readout changes with feature and gate changes. Between update 100 and update 2,000 (wd 1, and wd 0 at the same updates) I use the exact bilinear split Δ[(h − μ)W^ref] = Δ(h − μ)·W^ref_100 + (h − μ)_2000·ΔW^ref. It is reported both ways round, because the split depends on the order.

**Control C2: the same decomposition on a memorizer that was never filtered.**
- **What.** The lane's channels, with its basis rule (each unit's update-0 dominant input frequency), applied to the unedited arm A (`e3-me-keep`, wd 1) at the same updates. C (`e3-me-sham`) is added as a second reference, using its update-0 dominant frequency.
- **Why.** A memorizer's readout restricted to its units' dominant frequencies may carry a+b on its own (e2: each half of the filter alone gives 0.13–0.21 on decayed runs).
- **Decision rule.** "B's reference channel still carries correct answers after the collapse" is specific to B only if, at 5,000 and 20,000 updates:
  - B's reference-channel sufficiency accuracy exceeds A's by at least 0.10 in every seed;
  - B's reference-channel linear margin toward a+b exceeds A's, with non-overlapping seed ranges.
  Otherwise the decomposition gives a memorizer the same thing, and a persisting reference channel in B is not evidence of a preserved computation.
- **Full-table projections.** If the lane's coherence index (or any quantity) projects the logit table onto a+b−c over all 961 pairs, I also compute it for A. A projection onto functions of a+b−c alone gives the same answer offset to every pair: held-out accuracy is all or nothing. A network that fits its training pairs can get all of it from those pairs. If A scores like B on such a quantity, the quantity cannot show that a computation exists.

## 3. Dynamics runs (D6) and any intervention

These are checked after the runs are published; the rules are fixed now.

| # | check | PASS if |
|---|---|---|
| Y1 | Provenance and accounting | Six run directories, each with one running and one complete registry event and 20,000 completed updates. Each config's `init_weights` hash equals the plan's and the e3 start file's. Config differences from `e3-me-filt[-wd0]-w128-i<seed>-a1` are only wd, the checkpoint list and bookkeeping fields (run ID, purpose, paths, host, times). Each producing commit's shared code equals `72cfb5d` |
| Y2 | Identical start, data and evaluation | Step-0 weights equal the start file and the wd 1 run's step-0 checkpoint bitwise. The split hash is `aa8f7504…`. The step-0 metric row equals the wd 1 run's step-0 row except wall-clock fields |
| Y3 | Moment and step reset | `init_age` 50,000 is recorded and there is no `resume`. My replay of updates 1–10 from the start file, with zero moments and the update count starting at 1 at the run's wd, equals the published update-10 checkpoint to 1e-12 relative |
| Y4 | Short replay | For one run per wd chosen by me now (i30001 at wd 0.3 and i30002 at wd 0.1), my independent replay to update 300 reproduces the logged accuracies exactly and cross-entropies within 1e-8 relative + 1e-14 at every logged row |
| Y5 | Checkpoint scores | My common-set accuracy and σ-margin at every published checkpoint equal the lane's table to 1e-9 |
| Y6 | Timing and censoring | My T_θ values from `metrics.jsonl` under the lane's declared definitions equal the lane's. Censored runs are reported as > 20,000, and re-crossings are counted. Each ratio r is recomputed against the lane's pre-stated predictions, and its classification follows the pre-stated bands |
| Y7 | Exposure | E(t) = −t·log(1 − lr·wd) is the logarithm of the shrinkage factor that `experiment.adam` applies (M4) |

**Interventions.** Before any number:
- the immediate change at the edit is reported separately from recovery under training;
- I check that selection and amplitude use no held-out accuracy;
- the training damage (train accuracy and cross-entropy at the edit) and the perturbation size (‖Δ‖/‖w‖ per matrix and the gate flips) are disclosed;
- a frozen-parameter run states whether the frozen block gets neither updates nor decay, or only no updates.
The specific rules are written into this file as a dated addendum before the intervention's runs are examined.

## Expectation (held loosely)

- **C1.** The rise is generic in one or two seeds and inconclusive in the rest; e2's random-feature start is less fit and less concentrated than B. The random-feature start collapses at wd 1, and it holds at wd 0 except near chance.
- **M1–M4** pass.
- **R2** is the reading most at risk: the off-reference channel grows at both wd values.
- **C2.** A's reference channel gives low but nonzero sufficiency accuracy, about 0.1–0.2.
