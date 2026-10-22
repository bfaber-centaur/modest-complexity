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

## Addendum 1 (2026-10-22, about 02:40 UTC, before computing section 2)

Written before any channel, coherence or projection value was computed. Nothing from the lanes' output had been seen. Three parts of section 2 were under-specified and are fixed here.

- **C2 margin rule.**
  - Raw logits of different networks are on different scales, so B and A are compared on margins divided by that checkpoint's RMS centred logit over the 483 pairs. That normalisation is scale-free.
  - "Exceeds A's, with non-overlapping seed ranges" must hold for the margin over N and for the margin over S.
  - Margins are means over pairs, so that they add across channels.
- **Pure-decay benchmark (R1).** The ratio is the reference channel's mean margin at t divided by ρ_{t−100}² times its value at update 100, for t ≥ 100.
- **Coherence and full-table projections (my versions; the lane's are compared when they are pushed).**
  - For unit j in frequency group k = k_j, A_j = Σ_{(a,b)} Σ_c h_j(a,b) W_j(c) e^{−2πik(a+b−c)/p}.
  - The aggregate index is Σ_k |Σ_{j∈k} A_j| / Σ_k Σ_{j∈k} |A_j|.
  - It is computed over all 961 pairs, the 483 common pairs, and the 299 training pairs.
  - The full-table projection is g(d) = mean over a pair set of z(a, b, a+b−d). It gives common accuracy 1 if argmax_d g = 0 and 0 otherwise, and is computed over the same three pair sets.
  - Both are computed for A and B at updates 0, 100, 2,000 and 20,000.

## Addendum 2 (2026-10-22, about 02:35 UTC): the coordinator's requests on the mechanisms packet (D14 item 4), before computing them

**What had been seen when this was written.**
- My own implementation of the channels had been run on the published e3 states and the C1 runs. `channels_check.py` was not yet committed. It gave M1–M4 PASS. It also showed that the reference set K covers all 15 bins in every run, which makes R3's pre-stated split degenerate.
- I had read the lane's README (PR #35) and `channels.py`.
- I had not run the lane's code, any basis sham, or any start-content measure.

**(a) M1 against the lane's code.**
- `channels.py` `channels()` is compared with mine on every published e3 state of B (wd 1 and 0) and A, and on every published C1 state.
- PASS if each label-centred channel agrees to ≤ 1e-10 × max |centred z|.
- The lane puts units that were not live at update 0 wholly into the off-reference channel; if there are any, they are reported separately.

**(b) The lane's `analyse()` on the six C1 runs**, at the published updates. "Same channels, same order" is judged per seed at wd 1, against B from the lane's own rows, using the S-margin change from 300 to 2,000. That window is after the C1 runs' peaks (200–250) and is the D5 decline window.
- **SUPPORTED LOCALLY** if, in every seed:
  - both the reference and the off-reference channel fall;
  - the reference channel's share of the total fall is within ±0.15 of B's;
  - the label offset carries under 5%;
  - and at wd 0 the reference channel's S-margin rises over the same window.
- **CONTRADICTED** if, in every seed, one channel carries ≥ 90% of the fall, or the reference channel rises at wd 1.
- Otherwise **AMBIGUOUS**.

**(c) Is "outvoted, not erased" forced by the basis?**
- **Floor.** A's reference-only accuracy is 0.16–0.27, and B's late value must exceed it by ≥ 0.10 per seed. This is C2's accuracy part, applied at 5,000 and 20,000.
- **Basis sham.** The reference-only accuracy of B and the C1 runs at wd 1, at 5,000 and 20,000, is recomputed with k_j permuted across the live units. That keeps the frequency histogram. There are 8 draws, numpy seed 41100.
  - If the median sham is within 0.10 of the true basis in every seed, the half-correct reference channel is a property of any one-frequency-per-unit projection. The reading is then CONTRADICTED as a statement about the reference frequencies.
  - If the sham is at most A's floor + 0.05 in every seed, the reference frequencies carry it.
- **Start content (readout side).** By M4, W_t = ρ_t W_0 + ΔW_t exactly. At 2,000, 5,000 and 20,000 I report:
  - s_t = ρ_t ‖W_ref,0‖ / ‖W_ref,t‖, the largest share of the reference readout that the decayed start can be;
  - the projection coefficient c_t = ⟨W_ref,t, W_ref,0⟩ / ‖W_ref,0‖² against ρ_t.
  - If s_t < 0.05 at 5,000 in every seed, "not erased" describes the frequency slot, not the edited weights. Those decay on schedule, and the reference content is rebuilt by the gradient.
  - c_t ≫ ρ_t would mean the rebuilt content points along the start's direction.

## Addendum 3 (2026-10-22, 02:40 UTC by the container clock): section 3 completed for the decay-off runs (D13), the restricted-decay patch (D15) and the xref/onlyref runs, before examining any of them

**Clock note.** The times in the earlier headings were my estimates and are wrong. The commit times are the record: section 1–3 at `33bf731` (02:13:38 UTC), addendum 1 at `2d81d64` (02:16:50), addendum 2 at `4e459bf` (02:29:19). Every rule above was committed before the computation it governs.

**What had been seen when this was written.**
- D13 and D15 in DECISIONS.md, the dynamics lane's Q2 (its rules and reference table), the mechanisms lane's INTERVENTION.md, and the diff of the shared-code commit `e9cbc84` against `72cfb5d`.
- The names of the six D13 run directories and their registry timestamps. I have not opened their configs, metrics or checkpoints, nor the lane's `offswitch.py` or `figures/offswitch/`.
- One D7 replay's config, first metric rows and manifest (to learn the format). The xref/onlyref runs are not published yet.
- Nothing of the D6 runs beyond the README, its tables and their column names (section 3 rules stand as written).

**Tolerances used below.** "Replay-equal" means: train, cal and audit accuracies identical at every logged row; cross-entropies, norm, adaptive and decay norms within 1e-8 relative + 1e-14; weights at each published checkpoint within 1e-10 relative (Frobenius, per matrix). Replays are mine: `research/audit/independent.py` gradients and my own AdamW, which imports nothing from `experiment.py`.

### Restricted-decay patch, shared commit `e9cbc84` (pinned at `8108d8b`)

| # | check | PASS if |
|---|---|---|
| Z1 | No-flag path unchanged | (a) `adam()` of `72cfb5d` and of the pin, fed the same states and gradients for 300 full-batch updates from each B start at wd 1 and wd 0.3, give bitwise equal weights and moments and equal measured norms. (b) A diagnostic CLI run at the pin without the flags, i30001 at wd 1 to 300 updates with the D7 replay's settings, equals that published replay's states at 0, 50, 100 and 300 bitwise and its metric rows except wall-clock fields |
| Z2 | The projection | My P, written from the definition with an explicit DFT per unit (keep the cosine and sine at k_j), equals `decay_parts` on the start weights and on 10 random Gaussian matrices to 1e-12 relative, for both `on` values. P is idempotent and orthogonal (‖Pw‖² + ‖w − Pw‖² = ‖w‖², and ⟨Pw, w − Pw⟩ ≈ 0, all to 1e-12 relative). The rest holds every column's DC: the token means of (I − P)U and U are equal to 1e-14 |
| Z3 | One step | With zero gradient and zero moments, one `adam()` step at the pin gives exactly w − lr·wd·D(w), with D = P or I − P computed by `decay_parts` (bitwise), and equals it with my P to 1e-14 relative. With a real gradient from the start file, the moments after the step are bitwise those of the no-flag step (moments gradient-only), and w_new − (w − lr·wd·D(w)) equals the no-flag step's w_new − (1 − lr·wd)·w bitwise or to 1e-15 absolute (the decay sits outside the Adam update). The measured `decay_norm` equals ‖lr·wd·D(w)‖ to 1e-12 relative |
| Z4 | Projection files | Each file's SHA-256 equals plan.json and `decay_projection.json`. Its k equals my dominant input frequency (argmax over 1..15 of the summed energy of U and V columns, labels not used) of the B start file at update 0 for every unit, and its live mask equals mine (a unit is live if its U and V columns are not all zero). P(W₀) = W₀ to 1e-12 relative, as D15 states |

### Decay-off runs, D13 (`e4-dy-filt-wd1to0-u{300,1000}-w128-i{30000,30001,30002}-a1`)

| # | check | PASS if |
|---|---|---|
| Y1′ | Provenance | Six directories, each with one running and one complete registry event and 20,000 completed updates. `resume` is the e3 wd 1 run's checkpoint at s, its recorded hash equals Q2's table and the file. The config differs from the source e3 run only in wd (0), resume fields, checkpoint list and bookkeeping. The producing commit's shared code is `72cfb5d`, or the pin with no decay flags (covered by Z1) |
| Y2′ | Same state at the switch | The metric row at s equals the source run's row at s except wall-clock fields. Any checkpoint at s equals the source checkpoint's weights, moments and step bitwise |
| Y3′ | What carried over | My replay from the source checkpoint at s, with its moments and the update count continuing at s + 1, at wd 0, is replay-equal to the published run up to its first checkpoint after s (and every logged row up to it). As a sensitivity control, the same replay with the update count restarted at 1 must fail this tolerance at that checkpoint; if it does not, the check cannot tell a reset from a carry-over and is reported as uninformative |
| Y4′ | The reading | A_1, A_0 recomputed from the e3 metrics reproduce Q2's reference table to its 3 decimals. A_off(s + 1,000), Δ's and f per run equal the lane's to 1e-12 and fall in the band the lane reports, and the recovery label (any logged A_off after s ≥ A_1(s) + 0.05) agrees. Secondary values (A at 20,000, T(0.10) after the switch, censoring) are recomputed and compared, not judged |

**Intervention checklist for D13.** The immediate effect is nil by construction, which Y2′ verifies, so every difference is recovery under training. Perturbation size: only wd changes, so ‖Δw‖ = 0 at the switch, and the training damage is nil. Selection: s = 300 and 1,000 come from the D5 decline window, which was set on held-out accuracy. That is a choice of time, disclosed, not a FAIL. Fragility (reported, not judged): f with W = 500 and 2,000 and with the common-pair accuracy at the checkpoints nearest s + W.

### Restricted-decay runs, D15 (`e4-me-filt-wd1-{xref,onlyref}-w128-i{30000,30001,30002}-a1`), once published

| # | check | PASS if |
|---|---|---|
| X1 | Provenance | Registry events and 5,000 completed updates. The config equals the D7 wd 1 replay's from the same start except `decay_projection`, its hash, `decay_on` (`rest` for xref, `reference` for onlyref) and bookkeeping. The producing commit's shared files equal the pin's |
| X2 | Update 0 | The update-0 state equals the start file and the D7 replay's update 0, and the metric rows are equal there except wall-clock fields. So the immediate effect is nil |
| X3 | Step-by-step | My replay with w ← w − lr·wd·D(w) + Adam update, D from my own P, is replay-equal to every logged row and checkpoint up to 300, for all six runs. From 300 to 5,000 my replay must agree at every published checkpoint and logged row with a relative weight difference no more than 10× that of my no-flag replay of the D7 wd 1 replay from the same start (the floating-point baseline), and with identical accuracies at the published checkpoints. Larger gaps are investigated before any reading |
| X4 | Decay norm | The logged `decay_norm` at each logged update t equals ‖lr·wd·D(w_{t−1})‖ from my replayed w_{t−1} within the replay tolerance up to 300, and within 10× the baseline gap after that. At update 1, for one source, ‖D_rest‖² + ‖D_ref‖² from the xref and onlyref runs' first replayed update equals the full-decay ‖lr·wd·w₀‖² to 1e-12 relative |
| X5 | Reading | f = (acc_arm − acc_wd1)/(acc_wd0 − acc_wd1) on the 483 common pairs at 5,000 and 2,000, per source, recomputed and equal to the lane's to 1e-12. The band follows INTERVENTION.md as written |

**Intervention checklist for D15.** The immediate effect is nil (X2). Perturbation size is reported as the per-update decay norm of each arm against full decay, and ‖w_arm − w_replay‖/‖w_replay‖ per matrix at 300, 2,000 and 5,000. Training damage is reported as train accuracy and cross-entropy at the same updates. Selection: k_j comes from the start file without labels (Z4). Nothing is frozen; every parameter gets its Adam update, which Z3 verifies.

**Budget.** The diagnostic runs here are Z1(b) (300 updates, about 1 s), and possibly a diagnostic CLI rerun of one xref run if X3 needs investigating (about 16 s). My numpy replays count toward the audit's elapsed time; I estimate under 2 minutes in total.

## Addendum 4 (2026-10-22, about 02:55 UTC): control X0 for the exempt-reference result, before any run

**What had been seen.** The coordinator's summary of the mechanisms intervention (PR #40): xref reaches 0.998–1.000 common accuracy at 5,000, and onlyref collapses to 0.000, faster than uniform wd 1. I have not opened those runs or the lane's write-up. The projection files below were made by `xref_control.py`, from start weights only.

**Question.** Is xref's rescue a property of B's start, or does the exemption itself supply the structure? The exemption is label-free, but its rule (one input frequency per unit, the readout kept at the same frequency) is the known Fourier structure of modular addition.

**Design.** Four arms, each with three sources. All use exempt-reference decay (`decay_on rest`) with D15 settings otherwise: fresh Adam, lr 1e-3, wd 1, 5,000 updates, data seed 31000, log every 10, `diagnostics none`, shared code at the pin `8108d8b` or later. Checkpoints come every 1,000 plus 100, 300 and 500, and all are published.

| arm | start | k_j (projection file) | run IDs |
|---|---|---|---|
| I | the untrained init, `experiment.init(31, 128, seed)` (no start file; the e1 runs' step 0) | dominant() of the init, `intervention/decay_projection_init_i<seed>.npz` | `e4-au-init-xref-w128-i<seed>-a1` |
| RF | the C1 random-feature start file | the same file as I. Gate passed: dominant() of the RF start equals the init's k for all units, since the RF start keeps the init's U and V filtered at that frequency | `e4-au-rf-xref-w128-i<seed>-a1` |
| A | the unedited 50k start, `e3-me-keep-w128-i<seed>-a1.npz` | B's own file, `research/e4/mechanisms/intervention/decay_projection_i<seed>.npz`. Gate passed: dominant() of A equals B's k and live mask for all units | `e4-au-keep-xref-w128-i<seed>-a1` |
| S | B's start, `e3-me-filt-w128-i<seed>-a1.npz` | B's k permuted across the live units (numpy seed 41100; 6–10% of units keep their own k), `intervention/decay_projection_sham_i<seed>.npz` | `e4-au-filt-xrefsham-w128-i<seed>-a1` |

**Measure.** The primary measure is accuracy on the 483 common pairs at update 5,000, per run. Also reported:
- the same at 2,000;
- own held-out accuracy every 10 updates (figure);
- train accuracy and cross-entropy;
- the per-update decay norm against full decay.

**Labels, per seed, no pooling.**
- **Rescued**: accuracy at 5,000 ≥ 0.90.
- **Not rescued**: ≤ 0.30. For scale, B under uniform wd 1 is at 0.02–0.03 and B at wd 0 at 0.63–0.74.
- Otherwise **intermediate**.
- An arm is labelled when all three seeds agree; otherwise it is **mixed**.

**Readings, stated now.**
- **I rescued.** The exemption builds a working solution from untrained weights and 299 pairs. B's xref 1.0 is then a property of the intervention and says nothing about B's computation. The claim "the 1.00 is built in" is SUPPORTED LOCALLY. The lane's reading of necessity (decay on the reference parts is needed for B's collapse) is not affected, since it rests on f within B.
- **I not rescued, RF rescued.** A start with one input frequency per unit is needed, but the failed run's features are not.
- **A rescued.** B's edit is not needed, given the 50k features.
- **S rescued.** The exempted frequency need not be the unit's own, so any single frequency per unit suffices, which again points to the intervention supplying the structure. **S not rescued** while B is means the match between unit and frequency matters.
- **Only B rescued.** B's start matters, and "built in" is CONTRADICTED.

**Comparators (existing runs).**
- I: the e1 runs from the same init at wd 1 (`e1-co-add-w128-i30000-a1`, `e1-st-add-w128-i3000{1,2}-a1`), where a checkpoint at 5,000 exists, and otherwise their own held-out accuracy from `metrics.jsonl`.
- RF: C1 at wd 1 and wd 0.
- A: `e3-me-keep-w128-i<seed>-a1` at wd 1.
- S: B's xref and B's D7 replays.

**Checks on these runs.**
- Provenance as X1 (registry, config against plan, projection hash, shared code at the pin).
- Update 0 equals the start (X2).
- My replay with my own P is replay-equal up to 300 (X3, early part).

**Prediction (held loosely).** I, RF and A rescued. S not rescued, or intermediate.

**Cost.** 12 runs of about 16–20 s each: about 4 min of process time, about 2 min elapsed on 2 workers. After it, the audit has used about 7 of its 25 min, counting §3's replays.
