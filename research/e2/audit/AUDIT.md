# e2 independent audit: the restricted rank, the common set and "the rank precedes accuracy"

Session `e2-au-cloud-1`, branch `claude/project-thread-id1tmm`, base `6aaa3a2` (dispatch `48ea420`, shared code `fe284a5`). Inputs are the published e1, follow-up 1 and follow-up 2 runs; no new training runs. The question was written down before the new comparisons were computed ([QUESTION.md](QUESTION.md), with one dated note added afterwards). Software checks are PASS / FAIL / NOT CHECKED; interpretations are SUPPORTED LOCALLY / AMBIGUOUS / CONTRADICTED. The audit pairs have guided exploration since e1, so everything here is exploratory evaluation, not a confirmatory holdout.

## Figure 1: is the held-out preference specific to a+b?

![offset profile](figures/offset-profile.png)

Look before reading: is there any dip or bump away from k = 0, and is it larger than the step-0 profiles in light gray?

For each held-out pair, the label a+b+k is ranked among the labels never seen with either operand in training (the restricted-rank mask), only on pairs where both a+b and a+b+k are absent. At k = 0 this is the restricted-rank statistic. At k ≠ 0, a+b is left out of the comparison set, because a highly ranked a+b sits above every other probe and would push them all above 0.5. Source: [offset-profile.csv](figures/offset-profile.csv), from [offset_profile.py](offset_profile.py).

- **30% share.** The dip at k = 0 deepens with width: 0.47–0.51 at width 16 and 0.18–0.19 at width 512. Away from k = 0 no offset differs from 0.5 by more than 0.09 in any run. The spread over k is 0.016–0.034, against 0.016–0.019 for the no-decay runs and 0.010–0.016 for the same split at step 0.
- **50% share.** The dip reaches 0.002–0.003. The off-zero profile is noisier (largest deviation 0.11–0.14, spread 0.043–0.061) because only 2–21 labels are absent per pair, against 0.027–0.033 at step 0.
- **Controls.** The no-decay and shuffled-label runs and every step-0 profile are flat at 0.5 within ±0.11.
- **Seed consistency.** Off-zero profiles correlate across seeds at 0.09–0.68 in decayed addition runs of width ≥ 64. The correlation is −0.13 to 0.26 for the controls and about 0 at step 0.
- **No ties or singletons.** No pair is left out for lack of a comparison label, and no logit ties occur.

## Figure 2: does the restricted rank lead the two shares apart before accuracy does?

![precedes](figures/precedes.png)

Look before reading: at which checkpoint does each panel first separate blue from gray? Then compare the gray band in panel 1 with where the blue lines cross it.

All runs are width 128 and all values are on the 483 pairs that neither share trains on. Panel 1 ranks a+b among the labels absent from the 50% training set. Those labels are absent from the 30% training set too, so one mask serves both shares; the dotted lines use the 30% runs' own mask and track the solid ones within 0.03. The same init seed gives byte-identical step-0 weights in both shares (checked for 30000 and 30001). Source: `results/rank-checks/shares.jsonl`.

| step | restricted rank, 50% / 30% | rank among all labels, 50% / 30% | cross-entropy, 50% / 30% | accuracy, 50% / 30% |
|---|---|---|---|---|
| 1k | 0.39–0.44 / 0.48–0.49 | 0.76–0.79 / 0.73 | 6.67–6.98 / 6.56–6.72 | 0 / 0–0.002 |
| 2k | 0.25–0.33 / 0.47–0.48 | 0.64–0.71 / 0.71–0.73 | 5.71–6.33 / 6.85–6.90 | 0.002 / 0 |
| 5k | 0.06–0.12 / 0.43–0.44 | 0.30–0.46 / 0.70–0.71 | 3.65–4.74 / 7.38 | 0.019–0.075 / 0 |
| 50k | 0.0015–0.0025 / 0.26–0.31 | 0.011–0.015 / 0.56–0.60 | 0.38–0.49 / 9.86–10.46 | 0.878–0.913 / 0–0.004 |

The 30% values at 1k, 2k and 5k come from two seeds; init 30002 has only steps 300 and 50,000.

## Software checks

| # | Check | Result | Evidence |
|---|---|---|---|
| G1 | Restricted-rank mask uses training pairs only | PASS | All three implementations build it from the `train` indices alone: `research/audit/heldout_rank.py`, mechanisms `relational.py` and the loop in mechanisms `crosscheck.py` (read). In all 28 published runs the splits are disjoint, cover the 961 pairs and equal the frozen split files (`definition.jsonl`) |
| G2 | Operand positions and transposes | PASS | Every training set is transpose-closed. With addition targets, the position-specific and position-ignored masks are identical (0 entries differ), and a+b is absent for every held-out pair (662 of 662 and 483 of 483). With shuffled targets the two masks differ in 4,897 (pair, label) entries and change which of 154 pairs are usable, so the shuffled control depends on that choice; G6 removes the dependence. Note: (a, b) and (b, a) are two correlated observations. In the decayed addition runs their ranks correlate at 0.76–0.87 at widths 128–512 and at 0.95–1.0 at widths 16–64 and at 50%. The 662 pairs are 341 transpose groups (248 of the 483), so any interval that treats pairs as independent is too narrow by up to √2 |
| G3 | Ties, set sizes, empty and singleton sets, normalization, missing values | PASS for every published output | No ties occur at any final checkpoint of the 28 runs; counting ties against a+b gives the same mean. Absent labels per pair, a+b included: 6–27 (median 15) at 30%, 2–21 (median 8) at 50%, 5–21 (median 10) for shuffled targets under their own mask. There are no singletons, so no division by zero. Shuffled-target pairs whose a+b is present are left out and counted (197 of 662 remain), not silently. Latent edge case: `heldout_rank.py` and `relational.py` divide by zero if a+b is a pair's only absent label. That cannot happen in these splits but can at larger shares; minimal patch: drop and count rows with `absent_other.sum(1) == 0`, as `offset_profile.py` does |
| G4 | Chance references | PASS | Top-of-absent chance is the mean of 1/#absent over usable pairs: 0.0698 (30%), 0.1354 (50%), 0.1009 (shuffled, own mask). These match the 0.070 and 0.135 quoted in `research/coordinator/INDEX.md`, e1 AUDIT Part 4 and the mechanisms README. The rank mean has chance 0.5 in every split, so rank means compare across splits; top fractions compare only with their own split's reference |
| G5 | Like-for-like across shares | PASS | Nesting was checked from the frozen files: the 50% training set is the 30% training set plus 118 e1 calibration and 61 e1 audit pairs. On the 483 common pairs, a label absent under the 50% mask is absent under the 30% mask for every pair. At 50k the 30% runs rank 0.26–0.31 with the 50% mask (0.26–0.30 with their own) and the 50% runs 0.0015–0.0025. Top fractions: 0.29–0.37 against 0.99 (chance 0.135). All 30% widths share one split and mask, so the width comparison was like-for-like already |
| G6 | Nulls that keep the mask (width 128, three seeds, step 50,000) | PASS: the mask alone does not produce the signal | Trained rank: 0.27–0.31. Nulls: a random other absent label as the probe, same logits and mask, 0.500 (95% of 1,000 draws within 0.476–0.524); step 0, 0.499–0.504; the no-decay model, whose mask is identical (verified), 0.479–0.492; the shuffled-label model scored under the addition run's mask, pairs and probe, 0.498–0.504 (top fraction 0.057–0.070, chance 0.070). `nulls.jsonl` |
| G7 | `research/e2/coordinator/common_set.py` (uses `experiment.forward`) | PASS | Recomputed from the frozen split files with independent code. All 12 fields of `common_set.json` are equal, including the hash, class counts and "483 = 50% cal ∪ audit" (483 transpose-closed pairs, all inside the e1 audit split). The saved index is equal. All 51 rows of `common_set_scores.csv` match: accuracy identical, cross-entropy within 1.8e-15, median margin identical, no ties. At 50k: 30% runs 0 / 0 / 0.004, 50% runs 0.913 / 0.878 / 0.909. `results/common-set-check.json` |
| G8 | 50% runs: data, hashes, partition, recomputation, replay | PASS, by citation | e1 AUDIT Part 4 covers this: F10 (split rebuild, hash, 8 / 2 / 6 groups per residue), F11 (handoff, recipe, step-0 weights), F12 (all 11 checkpoints of all three seeds recomputed; CE within 4.1e-16), F13 (replays to 5k, all seeds, ≤ 9.5e-14), F14 (full replay of seed 30001; rounding only) and F15 (rank files). New here: G5 and G7 |
| G9 | Re-running every script in this directory reproduces every committed result file | PASS | Byte-identical, in 2.6 s total |

## Interpretations

| Claim | Verdict | Evidence and scope |
|---|---|---|
| The restricted-rank signal is specific to a+b: no other label at a fixed offset is preferred | SUPPORTED LOCALLY | Figure 1. Away from k = 0, deviations stay within 0.09 at 30% and 0.14 at 50%, comparable to the step-0 profiles (up to 0.10). This does not show that the readout is non-periodic: a readout spread evenly over many frequencies is itself a single sharp peak at a+b (dated note in QUESTION.md) |
| Decayed addition runs carry weak, seed-consistent structure at k ≠ 0 | AMBIGUOUS | Cross-seed correlations of 0.09–0.68 at width ≥ 64, against about 0 for controls and step 0. Each k uses a different subset of pairs, and the effect is a few hundredths of rank |
| The restricted rank reflects learned structure, not the mask | SUPPORTED LOCALLY | G6: four nulls that keep the mask all give 0.48–0.50 |
| In the 50% runs the restricted preference precedes held-out accuracy (e1 AUDIT Part 4; DISPATCH item 4) | SUPPORTED LOCALLY, partly by construction | The top-of-absent fraction is never below accuracy, by construction: if a+b is the argmax it tops the absent labels. This holds at all 51 checkpoints. The rank mean is not tied that way. On matched pairs and mask it separates the shares at 1k in all three 50% seeds against both 30% seeds with a 1k checkpoint, by 0.04–0.10. Cross-entropy and the unrestricted rank do not separate until 2k; at 1k both are slightly *worse* for 50%. Accuracy separates at 5k. Scope: 3 against 2 seeds, one nested split pair, checkpoints 1k apart |
| The restricted rank is an early signature of the eventual successful computation | CONTRADICTED as a level; AMBIGUOUS as a rate | The 50% runs pass 0.25–0.33 at 2k. The 30% runs reach the same 0.26–0.31 by 50k with accuracy 0–0.004 and cross-entropy near 10 nats, and the follow-up 1 run at width 512 plateaus at 0.177 to 200k without accuracy. A given value does not indicate eventual accuracy. What differs early is the rate, and that rests on the scope above |

I did not attack the mechanisms per-unit concentration reading this round. As the coordinator's DECISIONS.md already records, it is an observation on outputs not yet audited.

## Most consequential remaining uncertainty

Whether an early fall of the restricted rank predicts eventual generalization, rather than coinciding with it in one nested split pair. The two shares differ in which pairs are trained as well as how many (e1 AUDIT Part 4). The earliest separation is one published checkpoint (1k) on two 30% seeds, and the restricted rank only separates the shares earlier because it removes the labels seen with an operand: at 1k the 50% runs rank a+b *lower* among all labels than the 30% runs. A test would use a second 50% partition, or a 30% partition drawn from the 50% one, with checkpoints every 250 updates to 5k. Its prediction, fixed in advance, would be the restricted rank at 1–2k against accuracy at 50k.

## Reproduce

From this branch, with BLAS threads set to 1 and uv 0.12:

```bash
CO=research/coordinator/runs; ST=research/size-time/runs; E=research/e2/audit
uv run --locked python $E/offset_profile.py 50000 $CO/e1-co-* $CO/e1f2-* $ST/* > $E/results/offset-profile-50k.jsonl
uv run --locked python $E/offset_profile.py 0 $(for r in $CO/e1-co-* $ST/* $CO/e1f2-*; do [ -f $r/checkpoint_000000.npz ] && echo $r; done) > $E/results/offset-profile-step0.jsonl
uv run --locked python $E/rank_checks.py $E/results/rank-checks
uv run --locked python $E/check_common_set.py > $E/results/common-set-check.json
uv run --locked --group analysis python $E/plot_offset_profile.py $E/figures/offset-profile $E/results/offset-profile-50k.jsonl $E/results/offset-profile-step0.jsonl
uv run --locked --group analysis python $E/plot_precedes.py $E/figures/precedes $E/results/rank-checks/shares.jsonl
```

## Budget and accounting

- **Numerical time.** e2 used under one minute in total (2.6 s for a full regeneration), one process at a time, one BLAS thread, no paid compute. e1 usage (about 23 of 30 minutes) is reported separately in `research/audit/AUDIT.md`.
- **Runs and registry.** No training runs were launched, so there are no `plan.json` claims and `registry.jsonl` is not created.
- **Shared code.** No shared code was changed.

---

# Cycle 2: mechanisms packet 1 and the onset checkpoints

Base `8b11c2c`. The question and the decision rule were written down before computing: [QUESTION-2.md](QUESTION-2.md), committed at `145b29e`. The target is mechanisms packet 1 (`research/e2/mechanisms/`, PR #15). Its claim: units grouped by dominant input frequency carry an own-frequency a+b−c term aligned with the correct label in both shares, and removing a group does no more damage than removing a matched random set. Its own strongest alternative is that the aligned terms come from fitting labels equal to a+b.

## Figure 3: does the aligned term need a+b labels, weight decay, or neither?

![groups check](figures/groups-check.png)

Look before reading: in the top left, which conditions fill the shaded band (phase within ±π/2), and with what marker sizes? In the top right, how far does each colour sit above the open gray step-0 markers of the same runs?

Independent reimplementation (`groups_check.py`) at step 50,000, three seeds per condition, width 128, on the 483 common pairs. Step 0 has only three distinct networks, because every condition with the same init seed starts from the same weights.

| condition | input concentration | groups within ±π/2 (per seed) | within 0.5 rad | own-frequency amplitude, logits (median per seed) | amplitude / RMS of centred logits | share: groups / random sets | random sets within ±π/2 at the group's k | restricted rank |
|---|---|---|---|---|---|---|---|---|
| step 0 (3 networks) | 0.17 | 21/45 (7, 6, 8) | 11/45 | 0.0003–0.0005 | 0.0018–0.0027 | 0.0008 / 0.0008 | 0.50–0.57 | 0.50 |
| shuffled labels, decay 1 | 0.17 | 21/45 (6, 7, 8) | 10/45 | 0.012–0.016 | 0.0027–0.0035 | 0.0008–0.0011 / 0.0008–0.0009 | 0.53–0.63 | 0.50 under the addition mask |
| a+b labels, no decay | 0.22–0.23 | 34/45 (13, 12, 9) | 20/45 | 0.041–0.061 | 0.0047–0.0074 | 0.0023–0.0030 / 0.0023–0.0025 | 0.10–0.11 | 0.48–0.49 |
| a+b labels, decay 1, 30% | 0.40–0.44 | 45/45 | 40/45 | 0.12–0.15 | 0.030–0.037 | 0.012–0.017 / 0.0044–0.0047 | 0.25–0.28 | 0.26–0.31 |
| a+b labels, decay 1, 50% | 0.83–0.85 | 45/45 | 45/45 | 0.66–0.70 | 0.23–0.25 | 0.26–0.31 / 0.027–0.030 | 0.40–0.45 | 0.002–0.003 |

- **Shuffled labels.** The groups sit at the step-0 level on alignment, relative amplitude and share, so the readout does not make any trained network look aligned.
- **No decay.** The groups are aligned more often than at step 0 (34/45 against 21/45). Their amplitude relative to the logits is 2–3× the step-0 level and about one fifth of the decayed 30% runs'. Their share is no higher than that of random sets.
- **The background.** Random sets in the no-decay runs are mostly *anti*-aligned at the same k (10–11% within ±π/2). That fits a dip at d = 0 from the operand-associated labels, which the packet also saw.

## Figure 4 (post hoc): amplitude of the aligned term against the restricted rank

![amplitude vs rank](figures/amplitude-vs-rank.png)

Added after the diagnostic above was computed. It covers all 18 runs of the 30% width map at 50k (`groups-by-width.csv`), plus the width-128 controls and the 50% runs.

- **30% width map.** The groups' relative amplitude orders the restricted rank across the map: Spearman −0.99 over 18 runs. Most of that ordering is between widths, so it is confounded with width.
- **Fixed width 128.** At width 128 the three a+b conditions follow the same order: no decay 0.005–0.007 (rank 0.48–0.49), 30% 0.030–0.037 (0.26–0.31), 50% 0.23–0.25 (0.002).
- **Width 16.** The decayed runs at width 16 have the same amplitude as the no-decay runs (0.005) and the same null rank.

## Figure 5: the share separation with three seeds per share at 1k spacing

![precedes dense](figures/precedes-dense.png)

The figure adds the D9 checkpoints (3k, 4k, 6k–9k of all runs, and 1k–10k of 30% init 30002).

| readout | first checkpoint where the 50% runs separate from the 30% runs (three seeds each) |
|---|---|
| restricted rank | 1k: 0.39–0.44 against 0.48–0.49 |
| rank among all labels | 2k: 0.64–0.71 against 0.71–0.73. At 1k the 50% runs are worse (0.76–0.79 against 0.73) |
| cross-entropy | 2k: 5.71–6.33 against 6.85–6.90. At 1k they overlap (6.67–6.98 against 6.56–6.72) |
| accuracy | 2k: 1 of 483 pairs in each 50% run. Above chance (1/31) in two seeds at 5k and in all three at 6k |

The 50% runs pass the 30% runs' final level (0.26–0.31) between 2k and 3k: 0.25–0.33 at 2k, 0.14–0.25 at 3k.

## Software checks, cycle 2

| # | Check | Result | Evidence |
|---|---|---|---|
| G10 | Group readout reimplemented from the packet's written definitions | PASS | On all 255 group rows shared with `groups.csv` (six runs, 0 / 5k / 50k), group sizes are identical. Share, phase, amplitude, common-set accuracy after ablation and RMS logit change agree within 5e-7, the rounding of the audit CSV |
| G11 | Grouping uses only the analysed checkpoint | PASS | `groups.py` takes dominant frequencies from `unit_shares(w)` of the checkpoint it loads. The audit recomputes groups from each checkpoint alone and gets the same group sizes at every shared step. The over-training panel reads groups at each checkpoint, so no future grouping is used |
| G12 | Random sets matched as claimed | PASS, with a note | **Count and norm:** all 16 draws were found for every group at every checkpoint (240 per run and step). **Gates:** ablating hidden units cannot change another unit's gate in a one-layer network, and the audit's ablation, base logits minus the group's contribution, reproduces the packet's accuracies. **Logit change:** groups move the logits less than their random sets (RMS over all pairs 0.92–0.96 against 1.09–1.10; over the common pairs 0.86–0.92 against 1.07–1.08), so count-and-norm matching does not match functional size. Against the random-set trend at equal RMS change over all pairs, group residuals average −0.000 to +0.005 (random-set scatter 0.014–0.018), with 5–7 of 15 groups below the trend. Measured on the common pairs only, groups fall slightly below the trend: −0.002 to −0.007, with 8–11 of 15 below. "No worse than a matched random set" holds either way, within the scatter |
| G13 | Step-0 reference | Statement correction | The packet's "34 of 75 groups at step 0" counts byte-identical networks twice: the 30% and 50% runs with init seeds 30000 and 30001 start from the same weights. The three distinct step-0 networks give 21/45 |
| G14 | D9 extra checkpoints | PASS | All 50 files match the SHA-256 and byte count in their run's e1 manifest. The step stored inside each equals its name, and none duplicates a file in the run directory. `results/rank-checks/extra-checkpoints.json` |
| — | 30% runs: group against random-set ablation on the restricted rank | NOT CHECKED | The packet's figures are cited, not recomputed |

## Interpretations, cycle 2

| Claim | Verdict | Evidence and scope |
|---|---|---|
| The aligned terms are produced by the readout, for any trained network | CONTRADICTED | The shuffled-label groups match step 0 on alignment (21/45), relative amplitude and share |
| The aligned terms are a side effect of fitting labels equal to a+b (packet's strongest alternative) | AMBIGUOUS: partly | Fitting a+b labels without decay produces a weak aligned term: 34/45 groups aligned, about one fifth of the decayed 30% relative amplitude, with no group-specific share and a null restricted rank. With decay the term is about five times larger relative to the logits, group-specific (share 3× random sets) and aligned in 45/45 groups. So label fitting alone accounts for a small part of the term |
| The failed 30% runs build the same kind of term as the successful 50% runs, about five times weaker (packet INTERPRETATION) | SUPPORTED LOCALLY, with a qualification | Reproduced exactly (G10). The qualification: a weaker version of the same term is present in no-decay runs that show no restricted preference. Its presence and sign do not by themselves mark progress towards the 50% computation; its amplitude orders the restricted rank (Figure 4), which is an association only |
| Removing a group is no worse than removing a matched random set (50% runs) | SUPPORTED LOCALLY | G12, under either logit-change measure. Matching on count and norm alone would favour the groups, because they change the logits less |
| The packet's proposed decisive test 1: project the logits onto the a+b−c-structured part and see whether accuracy appears | Not informative as stated | A function of d alone whose largest value is at d = 0 predicts a+b for every pair, so the test is decided by the phase signs already measured |
| The restricted rank separates the shares before cross-entropy, the unrestricted rank and accuracy (cycle 1) | SUPPORTED LOCALLY, now three seeds per share | Figure 5. The first separation is at the first checkpoint after 300/400 updates, so the onset itself is not resolved |

## Most consequential remaining uncertainty, cycle 2

Whether the amplitude of the aligned term, relative to the rest of the logits, is what the restricted rank and eventual accuracy depend on, or only grows with them. All the evidence is observational: one split pair, and across widths a confound with width. The cheapest causal test is to scale only the groups' aligned own-frequency component in a 30% checkpoint. That means changing the output weights of each group along its own-frequency direction, not removing units, and asking whether the restricted rank, and then accuracy, follow. The sham would scale the same groups' non-aligned components by the same RMS logit change.

## Reproduce, cycle 2

```bash
C=research/coordinator/runs; S=research/size-time/runs; E=research/e2/audit
uv run --locked python $E/groups_check.py $E/results/groups-check.csv \
  $C/e1f2-co-add-w128-i30000-tf50-a1:0,5000,50000 $C/e1f2-co-add-w128-i30001-tf50-a1:0,5000,50000 $C/e1f2-co-add-w128-i30002-tf50-a1:0,5000,50000 \
  $C/e1-co-add-w128-i30000-a1:0,5000,50000 $S/e1-st-add-w128-i30001-a1:0,5000,50000 $S/e1-st-add-w128-i30002-a1:300,50000 \
  $S/e1-st-add0-w128-i30000-a1:0,5000,50000 $S/e1-st-add0-w128-i30001-a1:0,5000,50000 $S/e1-st-add0-w128-i30002-a1:0,5000,50000 \
  $S/e1-st-shuf-w128-i30000-a1:0,5000,50000 $S/e1-st-shuf-w128-i30001-a1:0,5000,50000 $S/e1-st-shuf-w128-i30002-a1:0,5000,50000
uv run --locked python $E/groups_check.py $E/results/groups-by-width.csv $(for r in $C/e1-co-add-w* $S/e1-st-add-w*; do echo $r:50000; done)
uv run --locked python $E/shares_dense.py $E/results/rank-checks
uv run --locked --group analysis python $E/plot_groups_check.py $E/figures/groups-check $E/results/groups-check.csv
uv run --locked --group analysis python $E/plot_amplitude_rank.py $E/figures/amplitude-vs-rank $E/results/rank-checks/definition.jsonl $E/results/groups-check.csv $E/results/groups-by-width.csv
uv run --locked --group analysis python $E/plot_precedes.py $E/figures/precedes-dense $E/results/rank-checks/shares-dense.jsonl
uv run --locked --group analysis python $E/plot_precedes.py $E/figures/precedes $E/results/rank-checks/shares.jsonl
```

The last command redraws the cycle-1 figure with the footnote now generated from the data. The numbers are unchanged.

## Budget, cycle 2

About 12 s of numerical execution, one process at a time, one BLAS thread. The e2 total is still under one minute. No runs were launched and no shared code was changed.
