# e2 mechanisms: figures first

Question and predictions, written before each cycle's measurements: [QUESTION.md](QUESTION.md). Interpretation is kept separately in [INTERPRETATION.md](INTERPRETATION.md). Session `e2-me-cloud-1`, branch `claude/project-thread-psralm`. Packet 1 read runs at `9e56faf` (e2 base `6aaa3a2`), packet 2 at `129e3df`; shared code `fe284a5`. No runs were launched, so there are no registry entries. Nothing here is audited.

## Packet 2. Weights-only edits: does the 30% runs' frequency structure carry correct answers?

![unitfilter](figures/unitfilter/unitfilter.png)

Vector version: [unitfilter.svg](figures/unitfilter/unitfilter.svg). Look before reading:
- Top row: where do the filled red points sit relative to the black crosses (unedited), the grey nulls and chance?
- Bottom left: when do the solid lines (filtered) rise compared with the dashed ones (unedited)?
- Purple (shuffled-label runs): does the edit produce anything there?

Definitions (`unitfilter.py`; written before measuring in QUESTION.md, cycle 2). No edit looks at labels, pairs or a+b−c.
- k_j is unit j's dominant input frequency (largest share of U[:, j] and V[:, j] energy in rfft bins 1..15), read at each analysed checkpoint.
- **Filter**: every live unit's U and V columns (token axis) and W row (label axis) are projected onto frequency k_j. The DC of U and V is kept, because it sets the unit's gate.
- **Scale**: the output rows of *frequency-aligned* units (dominant input frequency = dominant output frequency) are multiplied by s = 2, 4, 8. "Aligned only" removes all other units.
- **Score**: plain argmax over all 31 labels against a+b on the common set (483 pairs neither split trains on) and on the run's own cal + audit split. Train accuracy is against the run's own training labels. The RMS logit change is pair-centred, against the unedited model.
- **Nulls**:
  - the same filter at step 0;
  - each unit filtered to a random non-dominant frequency (8 draws, seed 36000);
  - the dominant-frequency filter with each unit's kept U, V and W components rotated by independent random phases (8 draws; same frequencies and energies; added after seeing the data, see below);
  - scaling random live-unit sets matched in count and total ‖u_j‖‖w_j‖ (±10%, 8 draws);
  - the e1 shuffled-label width-128 runs, on the 30% split, whose training labels agree with a+b on 2.7% of training pairs.

Source table, final step (50k). "Random other freq." and "phases scrambled" give the median (max) over 8 draws.

| run | unit concentration / in-out agreement | unedited acc: common / own held-out / train | **filtered** acc: common / own held-out / train | RMS logit change | kept energy, input / output | random other freq., common | phases scrambled, common |
|---|---|---|---|---|---|---|---|
| 50% i30000 | 0.85 / 1.00 | 0.913 / 0.913 / 1.000 | **1.000 / 1.000 / 1.000** | 2.50 | 0.86 / 0.88 | 0.017 (0.027) | 0.030 (0.043) |
| 50% i30001 | 0.83 / 0.99 | 0.878 / 0.878 / 1.000 | **1.000 / 1.000 / 1.000** | 2.72 | 0.84 / 0.85 | 0.032 (0.046) | 0.037 (0.058) |
| 50% i30002 | 0.85 / 1.00 | 0.909 / 0.909 / 1.000 | **1.000 / 1.000 / 1.000** | 2.58 | 0.86 / 0.87 | 0.021 (0.035) | 0.025 (0.052) |
| 30% i30000 | 0.40 / 0.66 | 0.000 / 0.000 / 1.000 | **0.431 / 0.437 / 0.689** | 4.08 | 0.42 / 0.37 | 0.078 (0.106) | 0.032 (0.052) |
| 30% i30001 | 0.44 / 0.75 | 0.000 / 0.000 / 1.000 | **0.540 / 0.562 / 0.732** | 4.00 | 0.45 / 0.39 | 0.064 (0.081) | 0.037 (0.054) |
| 30% i30002 | 0.44 / 0.71 | 0.004 / 0.003 / 1.000 | **0.580 / 0.607 / 0.789** | 3.98 | 0.46 / 0.39 | 0.068 (0.095) | 0.032 (0.060) |
| shuffled i30000 | 0.17 / 0.08 | 0.029 / 0.029 / 1.000 | 0.041 / 0.044 / 0.067 | 4.62 | 0.19 / 0.07 | 0.030 (0.046) | 0.025 (0.035) |
| shuffled i30001 | 0.17 / 0.10 | 0.035 / 0.033 / 1.000 | 0.035 / 0.039 / 0.094 | 4.57 | 0.19 / 0.07 | 0.035 (0.043) | 0.036 (0.039) |
| shuffled i30002 | 0.17 / 0.02 | 0.029 / 0.032 / 1.000 | 0.035 / 0.030 / 0.040 | 4.64 | 0.19 / 0.07 | 0.028 (0.046) | 0.031 (0.056) |

The 50% runs are the e1f2 runs, the 30% runs are e1-co-add-w128-i30000-a1 and e1-st-add-w128-i3000{1,2}-a1, and the shuffled runs are e1-st-shuf-w128-i3000{0,1,2}-a1. Chance is 1/31 = 0.032.

Variants and the scaling edit, final step, common-set accuracy (train accuracy in brackets):

| run | filter, input DC removed | filter inputs only | filter output only | aligned units only | matched random set only (median) | aligned × 8 |
|---|---|---|---|---|---|---|
| 50% i30000–i30002 | 1.000 (1.000) all three | 0.996–1.000 | 0.961–0.967 | no-op: 128, 127, 128 of 128 units aligned | | |
| 30% i30000 | 0.636 (0.836) | 0.147 (0.896) | 0.133 (0.833) | 0.006 (0.997) | 0.002 | 0.006 |
| 30% i30001 | 0.714 (0.870) | 0.157 (0.923) | 0.205 (0.900) | 0.002 (1.000) | 0.000 | 0.002 |
| 30% i30002 | 0.720 (0.913) | 0.188 (0.957) | 0.197 (0.860) | 0.008 (0.993) | 0.005 | 0.004 |
| shuffled, three runs | 0.023–0.041 | 0.027–0.037 | 0.031–0.039 | 0.029–0.037 | 0.030–0.036 | 0.029–0.041 |

Observations:

- **30% runs, final step.** Filtering every unit to its own dominant frequency takes common-set accuracy from 0.000–0.004 to 0.43–0.58, and own held-out accuracy to 0.44–0.61. Train accuracy falls from 1.000 to 0.69–0.79. The edit is large: RMS logit change about 4, keeping 0.42–0.46 of the input energy.
- **Nulls for the 30% runs.**
  - Random other frequency: median 0.064–0.078, max 0.106. That is above chance, so the non-dominant components carry a little a+b information too.
  - Phases scrambled: median 0.032–0.037, max 0.060.
  - Step 0, three distinct initializations: 0.029–0.064. Step 0 is the same network in every run with the same init seed.
  - Shuffled-label runs at 50k: 0.035–0.041.
- **50% runs.** The filter takes accuracy from 0.88–0.91 to 1.000 on both held-out sets, with train accuracy unchanged at 1.000. Random-frequency and phase-scrambled nulls are at chance (medians 0.017–0.037).
- **Over training** (bottom left; filtered common accuracy, unedited in brackets).

  | runs | 1k | 5k | 20k | 50k |
  |---|---|---|---|---|
  | 50% | 0.59–0.67 (0.000) | 0.77–0.89 (0.02–0.08) | 0.97–1.00 (0.59–0.79) | 1.000 (0.88–0.91) |
  | 30% | 0.28–0.34 (0.000–0.002) | 0.29–0.46 (0.000) | 0.50–0.59 (0.000) | 0.43–0.58 (0.000–0.004) |
  | shuffled | 0.019–0.046 at every step from 1k on | | | |

  The 30% run i30002 publishes only steps 300 and 50k. At step 300 its filtered accuracy is 0.168 (train 0.331); step 300 is after training began, so that point is not a null.
- **Which part matters (30% runs).**
  - Filtering only the inputs, or only the outputs, gives 0.13–0.21.
  - Both together give 0.43–0.58.
  - Also dropping the DC of U and V gives 0.64–0.72, with train accuracy 0.84–0.91.
- **Scaling edit.** In the 30% runs, upweighting the frequency-aligned units (s = 2, 4, 8) or keeping only them leaves common accuracy at 0.000–0.008, the same as matched random sets (0.000–0.012). Train accuracy stays 0.99–1.00. In the 50% runs nearly every unit is aligned, so the edit is close to a no-op, as stated in advance.
- **Shuffled-label runs.** No edit lifts them clearly above chance on a+b: the dominant-frequency filter gives 0.019–0.046 at every step from 1k on, and all 672 edited rows from 1k on give 0.014–0.056. But their units are no more concentrated than at initialization (0.17 at 50k against 0.17 at step 0), so they are a weak match for the 30% runs (0.40–0.44).

Revisions made after seeing data:
- The input-only, output-only and DC-removed variants were added to see which part of the filter matters.
- The phase-scrambled null was added because the shuffled-label runs turned out not to be frequency-concentrated (0.17, the step-0 level), which QUESTION.md listed as a failure mode. The scrambled filter keeps exactly the frequencies and energies the real filter keeps, in the same units, and changes only the phases.
- All three are separate rows in the CSV; the primary edit and its planned nulls are unchanged.

Checks:
- Tests (`test_unitfilter.py`, 5 OK; with `test_groups.py`, 9 OK).
  - Positive control: a hand-built addition network (units cos(ka+α), cos(kb+β) → cos(kc+α+β) for all k on a phase grid) scores 1.0 on all 961 pairs. With three other-frequency components added to every unit it scores below 0.5. The filter, which sees no labels, recovers the original weights to 1e-10 and the score of 1.0.
  - The filter is a projection: idempotent, keeps each unit's dominant frequency.
  - Phase scrambling preserves every unit's per-frequency magnitudes and destroys the control network's accuracy.
  - The random frequency never equals the dominant one, and covers all others.
  - Scaling all units scales the logits, and s = ∞ keeps only the chosen units.
- Every checkpoint read passes the manifest hash, split and target hashes (each run against its planned split and its manifest target) and logged-metric replay (maximum difference 0.0; `unitfilter_summary.json`). The common-set file is verified by SHA-256.
- Outputs were generated from clean commit `dd0c17e`. The tree status recorded in the summary lists only the output directory. The CSV is byte-identical to the run before the last figure-only code change.

Reproduce packet 2:

```bash
C=research/coordinator/runs; S=research/size-time/runs; T=0,1000,5000,20000,50000
uv run --locked --group analysis python research/e2/mechanisms/unitfilter.py --source-commit 129e3df --out research/e2/mechanisms/figures/unitfilter \
    --runs $C/e1f2-co-add-w128-i30000-tf50-a1:$T $C/e1f2-co-add-w128-i30001-tf50-a1:$T $C/e1f2-co-add-w128-i30002-tf50-a1:$T \
    $C/e1-co-add-w128-i30000-a1:$T $S/e1-st-add-w128-i30001-a1:$T $S/e1-st-add-w128-i30002-a1:300,50000 \
    $S/e1-st-shuf-w128-i30000-a1:$T $S/e1-st-shuf-w128-i30001-a1:$T $S/e1-st-shuf-w128-i30002-a1:$T
uv run --locked --group analysis python -m unittest discover -s research/e2/mechanisms -v
```

Budget: e2 numerical execution so far is about 5 minutes (packet 1 about 4, packet 2 under 1), in one process with one BLAS thread. e1 usage was about 14 minutes, recorded separately.

## Packet 1. Unit groups by dominant frequency, in failed (30%) and successful (50%) runs

![groups](figures/groups/groups.png)

Vector version: [groups.svg](figures/groups/groups.svg). Look before reading:
- Top left: where do the red groups sit relative to their own random sets and to their first-step points?
- Top middle: which way do the red phases point?
- Bottom: do the coloured points separate from the grey cloud at equal logit change?

Definitions (`groups.py`):
- A group is the set of live units whose dominant input frequency is k at the analysed checkpoint (checkpoint-local).
- Its contribution is relu(U[a]+V[b]) W restricted to those units, centred over labels for each pair.
- The *share* is the fraction of that contribution's variance, on a set of pairs, explained by a function of d = a + b − c mod p.
- The *own-frequency term* is the cos(2πkd/p + phase) component of that function. Phase 0 means it peaks at the correct label.
- The ideal "pizza" term |cos(πk(a−b)/p)| cos(2πkd/p), the form expected for this architecture (QUESTION.md), has share 0.81 on these pairs.
- Random sets: 16 per group, drawn from the live units with the same count and a total ‖u_j‖‖w_j‖ within 10% of the group's (seed 35000).
- The common set is the 483 pairs neither split trains on (`research/e2/data/common-never-trained-30-50.npz`, verified by SHA-256 in the summary).

Source table, final step (50k) of each run. Group and random-set columns are medians over the 15 groups and 240 random sets.

| run | accuracy, common / own held-out | restricted rank (common) | whole-model share | group share (range) | random-set share | group share, own held-out | own-frequency amplitude (logits) | groups with phase within ±π/2 | ablation Δ common acc, group / random | ablation Δ rank, group / random | RMS logit change, group / random |
|---|---|---|---|---|---|---|---|---|---|---|---|
| e1f2-co-add-w128-i30000-tf50-a1 | 0.913 / 0.913 | 0.002 | 0.266 | 0.305 (0.162–0.459) | 0.028 | 0.305 | 0.66 | 15/15 | −0.041 / −0.062 | +0.0005 / +0.0011 | 0.94 / 1.09 |
| e1f2-co-add-w128-i30001-tf50-a1 | 0.878 / 0.878 | 0.002 | 0.241 | 0.264 (0.087–0.452) | 0.029 | 0.264 | 0.67 | 15/15 | −0.054 / −0.079 | +0.0006 / +0.0013 | 0.96 / 1.09 |
| e1f2-co-add-w128-i30002-tf50-a1 | 0.909 / 0.909 | 0.002 | 0.266 | 0.306 (0.182–0.386) | 0.029 | 0.306 | 0.70 | 15/15 | −0.039 / −0.058 | +0.0006 / +0.0014 | 0.92 / 1.10 |
| e1-co-add-w128-i30000-a1 | 0.000 / 0.000 | 0.300 | 0.009 | 0.013 (0.001–0.051) | 0.004 | 0.011 | 0.12 | 15/15 | 0.000 / 0.000 | +0.0126 / +0.0136 | 1.07 / 1.04 |
| e1-st-add-w128-i30001-a1 | 0.000 / 0.000 | 0.277 | 0.009 | 0.012 (0.002–0.037) | 0.004 | 0.015 | 0.12 | 15/15 | 0.000 / 0.000 | +0.0173 / +0.0194 | 1.01 / 1.03 |
| e1-st-add-w128-i30002-a1 | 0.004 / 0.003 | 0.258 | 0.011 | 0.017 (0.003–0.039) | 0.005 | 0.015 | 0.15 | 15/15 | 0.000 / 0.000 | +0.0149 / +0.0132 | 0.99 / 1.01 |

For the 50% runs the own held-out split is the common set. For the 30% runs it has 662 pairs: the common set plus the 179 pairs the 50% share trains on.

Observations:

- **50% runs.**
  - Each group's held-out contribution is mostly a single cosine of a+b−c at the group's own frequency. Share 0.09–0.46, about 10× the matched random sets, and almost all of it at frequency k.
  - All 45 own-frequency terms peak at the correct label (|phase| < 0.5 rad), with amplitude 0.39–0.91 logits.
  - The whole model's share is 0.24–0.27, against 0.81 for the ideal pizza term. Most of the held-out logit variance is not a function of a+b−c.
- **30% runs.**
  - Group shares are 0.012–0.017 (median by run), 3× their random sets (0.004–0.005) and about 15× step 0 (0.001), but 20× below the 50% runs.
  - All 45 own-frequency terms point at the correct label: |phase| < π/2 for 45/45 and < 0.5 rad for 40/45.
  - Their amplitude, 0.02–0.36 logits (median 0.12–0.15), is about one fifth of the 50% runs'.
  - The null reference: groups at step 0 point within ±π/2 in 34/75 cases, with amplitude about 0.0004.
- **Over training** (top right; groups read at each checkpoint).
  - The 30% runs' group share rises from 0.001 at step 0 to 0.005 at 5k and 0.012–0.017 at 50k. At 5k, 27/30 phases are aligned, with amplitude 0.03.
  - The 50% runs reach that amplitude by step 1k (0.04, 15/15 aligned) and 0.18 by 5k, while common-set accuracy is still 0.02–0.07.
- **Ablation, 50% runs.**
  - Removing one group costs 0.02–0.10 common-set accuracy; a matched random set costs 0.01–0.15.
  - Groups change the logits less (RMS 0.92–0.96 against 1.09–1.10). At equal logit change, group and random-set ablations fall on the same trend (bottom middle).
  - Train accuracy stays 1.000 throughout.
- **Ablation, 30% runs.** Group and random-set ablations are indistinguishable on the restricted rank (+0.013 to +0.017 against +0.013 to +0.019) and on logit change. Accuracy stays 0.
- **Training pairs.** Every share is higher here (bottom left: 0.08–0.20 for 30% groups, 0.3–0.6 for 50% groups). Training labels equal a+b, so memorization contributes to this share; only held-out shares bear on the question.

Revision made after seeing data: `peak_d0` (does the group's function of d peak at d = 0) was the planned readout. On held-out pairs no single group peaks at d = 0, in either share, while 237/240 random sets in the 50% runs do. Each group's profile has a dip at d = 0, below its cosine. The label at d = 0 is always absent from both operands' training pairs, so the profile mixes the present/absent contrast into the cosine. I added the phase of the own-frequency component (`k_phase`, `k_amplitude`) as the readout for "peaks at the correct label". `peak_d0` stays in the CSV.

Checks:
- Tests (`test_groups.py`, 4 OK): a pure clock term has share 1 and phase 0, and a shifted one moves its phase by 2πk/p. The pizza reference matches the derived 8/π². Noise share is below 0.01. Group contributions add up to the centred logits.
- Every checkpoint read passes the manifest hash, split hash (each run against its planned split) and logged-metric replay (`groups_summary.json`).
- The figure was inspected and regenerated from a clean commit; the CSV is byte-identical across reruns.

Budget at packet 1: about 4 minutes of e2 numerical execution.

Reproduce packet 1:

```bash
C=research/coordinator/runs; S=research/size-time/runs
uv run --locked --group analysis python research/e2/mechanisms/groups.py --source-commit 9e56faf7 --out research/e2/mechanisms/figures/groups \
    --runs $C/e1f2-co-add-w128-i30000-tf50-a1:0,1000,5000,10000,20000,50000 $C/e1f2-co-add-w128-i30001-tf50-a1:0,5000,50000 \
    $C/e1f2-co-add-w128-i30002-tf50-a1:0,5000,50000 $C/e1-co-add-w128-i30000-a1:0,5000,50000 \
    $S/e1-st-add-w128-i30001-a1:0,5000,50000 $S/e1-st-add-w128-i30002-a1:300,50000
uv run --locked --group analysis python -m unittest discover -s research/e2/mechanisms -v
```

e1-st-add-w128-i30002-a1 publishes only steps 300 and 50,000, so its first point is step 300.
