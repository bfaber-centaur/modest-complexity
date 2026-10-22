# c3 scaffolding lane: observations (M2 first figure, M3 follow-up)

**Status.** These are observations from the 12 pre-registered branches and the first figure (reviewed: REVIEW.md §3, nits N-S1 to N-S3 folded in below), and from the one follow-up, Adam reset versus carried at S = 1k (section F). Sources: [figures/scaf_main.png](figures/scaf_main.png), [figures/scaf_followup.png](figures/scaf_followup.png) and the CSVs beside them, plus [checks.json](checks.json) and [followup_checks.json](followup_checks.json). Design and predictions: [QUESTION.md](QUESTION.md) (b1928fe). Adaptive choices: [ADAPTIVE.md](ADAPTIVE.md).

**Scope.** One task draw and split (`ops51000-ls51100`), one tied/tied width-128 recipe, three paired init seeds, two switch times, horizon 50,000 updates. The branches from one checkpoint are not independent replicates. Nothing below is a necessity, impossibility, hidden-algorithm or novelty claim, and the arms tested are not bounds.

## Checks (all pass)

- **Joins.** In all 12 branches, the first row (step S) equals r1 tied/inter's published row at S in every r1 key.
- **Schedules.** Step weights equal the restated schedule at every logged row (0 differences). The recorded schedule matches at S − 1, S, S + 1, S + 1,998, S + 1,999 and S + 2,000.
- **Exposure.** It matches a recount: drop [0, 0, 49,000] or [0, 0, 40,000] intermediate/final-labelled updates after S; taper [1,999, 1,999, …].
- **Frozen evaluation.** The lane's frozen L1–6 evaluation equals c2 generalization's published `length.csv` exactly for keep and final-only at 10k and 50k (12 of 12).
- **Logged rows.** At every fixed time, the frozen held-out prediction string equals the logged one.
- **Runs.** None failed. Wall time was 27.2 of 40 min (README).

## Observations

**O1. At 50,000, every branch is "retained" on every endpoint measure** (`scaf_retention.csv`, `status`). That is 84 of 84: 12 branches × {held-out L3, ¬N, L1, L2, L4, L5, L6}, with |Δ_keep| < 0.05.

| measure at 50k | branches |
|---|---|
| held-out length-3 | 1.000 |
| ¬N | 1.000 |
| L4 | 1.000 |
| L5 | ≥ 0.998 |
| L6 | 0.990–1.000 |

- **Final-only from init (D) at 50k:** held-out 0.56 / 0.70 / 0.66, ¬N 0.12 / 0.26 / 0.14, L4–6 0.10–0.15.
- **So Δ_D(50k)** is +0.30 to +0.44 on held-out L3 and +0.85 to +0.90 on L4–6 in every branch.

**O2. Switch at 1,000** (every seed partial: held-out 0.34–0.36, ¬N 0.00–0.02, L6 0.14). The branches keep improving without intermediate labels, more slowly than keep.

First of five logged evaluations ≥ 0.95 (`scaf_threshold.csv`):

| arm | held-out | ¬N |
|---|---|---|
| keep | 7,700 / 9,150 / 8,200 | 8,650 / 11,200 / 9,000 |
| drop | 14,600 / 20,000 / 20,450 | 16,650 / 24,100 / 22,800 |
| taper | 12,050 / 21,400 / 15,300 | 14,800 / 23,400 / 18,550 |
| D | censored at 50k | censored at 50k |

- **The largest deficits against keep come at the fixed times 10k–20k** (`scaf_retention.csv`). Examples: drop s51202 at 10k has held-out Δ_keep −0.39 and ¬N −0.86, and L6 at 20k is −0.57.
- **L6 reaches ≥ 0.97 by 40k** in every branch. Keep's L6 is 1.0 at 20k in s51200 and s51202, and 0.986 at 20k in s51201 (1.0 from 30k).
- **Branch delta at 50k** is +0.64 to +0.66 on held-out and +0.85 to +0.86 on L6, so this is "further improvement" by QUESTION.md's rule.

**O3. Switch at 10,000, length 3.** At S, s51200 and s51202 are at 1.0 and s51201 at 0.961 (¬N 0.88).
- **Once at 1.0, no branch falls again.** Held-out and ¬N have no logged row below 1.0 after their first 1.0, in all six branches.
- **s51201's ¬N first sustains ≥ 0.95** at 11,300 (drop) and 11,150 (taper), against 11,200 for keep.

**O4. Switch at 10,000, lengths 4–6.** At S, L6 is 0.86 / 0.56 / 0.72.
- **The branches lag keep at 20k.** Drop L6 is 0.93 / 0.65 / 0.98 and taper 0.95 / 0.73 / 0.95, against keep's 1.00 / 0.99 / 1.00. Δ_keep is as low as −0.34 (drop s51201); L5 is −0.13.
- **They close by 40k–50k.** L6 is ≥ 0.993 at 40k and ≥ 0.998 at 50k.
- **s51200 drop:** L6 is 0.950 at 12k and 0.929 at 20k, a small non-monotone dip, then 0.984 at 30k.

**O5. Lengths 1–2 (previously supervised states) are kept without their labels.**
- **The training readout stays near 1.0** on h1 and h2. The minimum logged accuracy after S is 0.97 on step 1 (at 17,200) and 0.92 on step 2 (at 7,100), both in drop at 1k, s51201. It is 1.0 at every logged row after a switch at 10k.
- **Frozen L1 and L2** are ≥ 0.95 at every fixed time and 1.0 at 50k.
- **Training CE at 50k** on step 1 / step 2 is about 1e-5 / 2e-7 in every branch. Keep's is about 5e-9 / 5e-9. D's is 8.6–10.2 / 11.1–11.9 across seeds, far worse than chance (ln 8 ≈ 2.1); D's readout was never trained on h1 or h2.

**O6. Taper against drop.**
- **At 50k**, C is within 5 pp of B on every measure at both S: all are ≥ 0.99.
- **At 1k**, the taper's held-out crossings come 2,550 and 5,150 updates earlier than the drop's in s51200 and s51202, and 1,400 later in s51201.

**O7. Optimizer** (`scaf_optimizer.csv`).
- **Update L2 after the switch.** 50 updates after S it is 0.76–0.98 of its value at S in 11 of 12 branches, and 1.01 in `taper-at1k-s51202`. At S + 2,000 the ratio is 0.45–0.90.
- **The eps regime begins where it does in keep and D.** Every coordinate's √v̂ is below eps from 29,050–29,300 in every branch, against keep's 29,050–29,300 and D's 27,800–29,450. The median falls below eps at 24,900–25,200.
- **Train final CE plateaus at about 1.4e-8** in every branch, against keep's 4.7e-9 and D's 2.2e-8.

**O8. Wrong held-out predictions are neighbour-compatible** (`scaf_compatibility.csv`). At every fixed time, every wrong prediction of every branch equals some one-slot training neighbour's label, against 0.67–0.75 under uniform guessing. The share matching a same-prefix neighbour is 0.90–1.0, against about 0.29–0.39. The c2 time lane found this for all arms.

## Predictions (QUESTION.md), scored

| # | prediction | outcome |
|---|---|---|
| 1 | S = 10k length-3 retained, all seeds; s51201 reaches the threshold | **held** (O3) |
| 2 | S = 10k step-1/2 training accuracy < 0.90 in ≥ 2 seeds by 50k | **failed**: 1.0 throughout (O5) |
| 3 | S = 10k L6 degraded ≥ 5 pp at 50k in ≥ 2 seeds | **failed** at 50k (O1). A lag of up to 34 pp at 20k closes by 40k–50k (O4). |
| 4 | S = 1k between D and keep at 50k; ≤ 1 seed crosses; ¬N and L4–6 < 0.5 | **failed**: every seed crosses, and all measures reach about 1.0 (O2) |
| 5 | taper within 5 pp of drop at the endpoints | **held** (O6) |
| 6 | update norm falls at first; eps onset within ±5k of 27,800–29,650 | **held** (O7) |

## Interpretation under the pre-registered rules

These hold for this task, recipe, horizon and seeds.
- **Lasting benefit (a training-history effect).** Yes, at both S, in all three seeds: B and C end 0.30–0.44 above D on held-out L3.
- **Persistence at S = 10k.** Yes: every measure is retained in all seeds. Length transfer was still forming at S, so S = 10k is also acquisition at L4–6, and it completes without intermediate labels, more slowly than keep.
- **S = 1k.** This is "further improvement": 1,000 updates (drop) or about 3,000 (taper) of intermediate supervision, followed by final-only supervision, end where keep ends by 50k. D, without them, does not.
- **Collapse.** None.
- **Limits.**
  - A drop changes both the loss and the optimizer's carried moments. So the difference between B and D is not attributed to "labels" alone: the state and the moments at S both differ from D's.
  - 50k is not an asymptote.
  - The equal endpoints say nothing about the paths. At S = 1k the lag lasts 5,000–12,000 updates.

## F. Follow-up: Adam reset versus carried moments at S = 1k

**Design.** Declared in QUESTION.md's addendum (84dc657) before any follow-up run, and approved by the coordinator after the first figure. Pin `82cef77` (DECISIONS D10, reviewer PASS, REVIEW.md §4).
- **Arms.** A′ keep and B′ drop are each run from r1 tied/inter `ckpt-1000`. A fresh Adam state (both moments and the bias-correction count) is set just before the update at 1,000, and the weights are carried.
- **Comparison.** Each is compared at equal total updates with the reused carried arms A and B, and with D. Six runs, none failed: 16.8 of 20 min (README).

**Checks** (`followup_checks.json`).
- **The reused carried arms are matched controls.** Under the re-pin with no reset, the 1k→3k check runs reproduce A in every r1 key at every logged row (three seeds). They also reproduce B in every key, `diag` included, with identical `ckpt-3000` arrays (s51200).
- **The reset runs.** Each joins r1's row at 1,000 and has the declared step weights at every logged row (0 differences). It records `reset_at` 1000, its exposure matches a recount, and its Adam count at 3,000 is 2,000.

**F1. Endpoints: the reset changes nothing at 50k** (`scaf_followup_effects.csv`). In all three seeds:
- A′ and B′ are at 1.000 on held-out L3, ¬N and L4.
- L5 and L6 are ≥ 0.992.
- Both reset effects are within 1 pp on every measure: the effect in keep is A′ − A and the effect in drop is B′ − B.

**F2. Paths: the reset speeds up acquisition after the drop, and barely moves keep.** The table gives the first of five logged evaluations ≥ 0.95 (`scaf_followup_threshold.csv`):

| arm | held-out | ¬N |
|---|---|---|
| A keep | 7,700 / 9,150 / 8,200 | 8,650 / 11,200 / 9,000 |
| A′ keep, reset | 8,150 / 8,750 / 8,950 | 9,050 / 11,000 / 9,250 |
| B drop | 14,600 / 20,000 / 20,450 | 16,650 / 24,100 / 22,800 |
| B′ drop, reset | 12,250 / 14,100 / 16,350 | 12,550 / 17,950 / 18,600 |

- **At the fixed times**, B′ − B is ≥ 0 for every measure and seed, and positive on every measure at 10k. On held-out L3 at 10k it is +0.055 / +0.125 / +0.125, and on L6 at 20k it is +0.25 / +0.33 / +0.18.
- **A′ − A** is between −0.06 and +0.09 on every measure and time.
- **Their difference** on L6 at 20k is +0.25 / +0.31 / +0.18.
- **Ordering.** B′ still lags A and A′ in every seed. D is far below all four.

**F3. The reset transient** (`scaf_followup_transient.csv`, logged every 50 updates; the figure's bottom row).
- **The first update after the reset** has 17–22× the L2 of the carried run's update at 1,000.
- **At 1,050–1,500**, the ratio is 1.15–1.75 in both arms.
- **The reviewer's per-update recomputation** (CX-R1, REVIEW.md §4) gives an RMS step 2.1–2.6× the carried one over the first 100 updates, in all six seed × arm pairs.
- **Train final CE** does not rise at any logged row over 1,000–1,300 in any reset run. It is below the carried run's at every logged row over 1,050–6,000, except at 1,050 in s51202 (both arms).

**F4. Optimizer regime.** The eps regime (every coordinate's √v̂ < eps) begins at:

| arm | onset | carried | shift |
|---|---|---|---|
| A′ | 28,600 / 28,750 / 30,200 | 29,250 / 29,050 / 29,300 | −650 / −300 / +900 |
| B′ | 28,450 / 28,650 / 30,250 | (same as B) | −800 / −400 / +950 |

At 50k, train final CE is about 4.7e-9 in A′, as in A, and about 1.4e-8 in B′, as in B.

**F5. Readout on h1 and h2 after the drop.** The lower of the two training accuracies, at every logged row from 1k, has a minimum of 0.906 in B′ and 0.922 in B. At 50k it is 1.0 in both.

### Follow-up predictions (addendum), scored

| # | prediction | outcome |
|---|---|---|
| 1 | B′ within 5 pp of B, and A′ of A, at 50k on L3, ¬N, L4–6, all seeds | **held** (F1) |
| 2 | B′'s held-out crossing within 3,000 updates of B's in ≥ 2 seeds | **failed**: earlier by 2,350 / 5,900 / 4,100, so within 3,000 in one seed (F2) |
| 3 | update L2 in the first 100 updates after the reset > 1.5× carried; train CE may rise | **held** on the first update (17–22×) and on the reviewer's 100-update RMS (2.1–2.6×). At the logged 1,050 and 1,100 the ratio is 1.37–1.75, above 1.5 in 7 of 12. No CE rise (F3). |
| 4 | eps onset within ±2,000 of the carried arm | **held**: −800 to +950 (F4) |

### Interpretation under the addendum's rules

- **"Weights at S suffice" holds** in all three seeds. B′ ends within 5 pp of B on held-out L3 and L6, so under this recipe the state at 1k is enough to reach the endpoint without the carried moments.
- **"Moments implicated" and "reset perturbation"** do not apply. No reset arm ends ≥ 5 pp below its carried arm.
- **Paths differ.** The reset shortens the drop branch's post-switch acquisition by 2,350–5,900 updates. It moves keep by −400 to +750. So carrying the Adam state across the drop slows acquisition, and the cause is the reset, not a single mechanism. The reset both removes the moments' history (v̂ still at the three-head scale) and makes the early steps larger (F3; CX-R1). This run does not separate the two.
- **Consistent with O7.** In carried B the update norm falls to 0.53–0.56 of its value at S by S + 2,000.
- **Limits.** One reset time, one recipe, three seeds and a 50k horizon. The four arms share their weights up to 1k, so they are not independent replicates.

## Reviewer handoff

1. Recompute one join, one exposure and one schedule point per S from `runs/<ID>/` and r1's records.
2. Recompute frozen L6 at 20k for `c3-scaf-drop-at10k-s51201-a1` (0.649) and at 50k for `c3-scaf-drop-at1k-s51202-a1` (0.993) from the committed checkpoints.
3. Check O2's crossings against `metrics.jsonl` (first of five consecutive logged rows ≥ 0.95).
4. Check the claim in O5 that h1/h2 training CE stays about 1e-5 / 2e-7 after a drop (the logged `train.step1.ce`, `train.step2.ce` at 50k).
5. Check that no checkpoint was chosen by outcome. Fixed times only: S, S + 2,000, 10k (S = 1k), and 20k–50k.
6. Follow-up: recompute B′ − B on L6 at 20k for s51201 (+0.33: 0.766 against 0.438) from the committed checkpoints.
7. Follow-up: recompute the four check runs' equality and one reset run's Adam count at 3,000 (`followup_checks.json`).
8. Follow-up: check F2's crossings against `metrics.jsonl`.
