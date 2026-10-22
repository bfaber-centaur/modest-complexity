# c2 sharing lane: note (M4)

**Question.** Within the same three-step recurrent computation on r1's task, how do tying the transition and tying the readout across steps change held-out composition under r1's 10k recipe? Design, predictions and measures are in [QUESTION.md](QUESTION.md), committed before any measurement. Commands, budget and run manifest are in [README.md](README.md). Every number below comes from `figures/endpoints.csv`, `figures/trajectories.csv`, or the CSVs in `out/`.

**Cells.** All four cells get identical labels (step weights 1 1 1), 384 rows, 10,000 full-batch Adam updates and wd 0.1. All four start as the same function at each init seed.

| cell | transition | readout | parameters |
|---|---|---|---|
| TT | tied | tied | 19,208 |
| TU | tied | per-step | 21,272 |
| UT | per-step | tied | 53,256 |
| UU | per-step | per-step | 55,320 |

TT and TU are r1's original runs. UT and UU are new.

## Observations

Primary figure: [figures/sharing_grid.png](figures/sharing_grid.png).

**Every run fits.** Training accuracy is 1.0 at all three steps by 350–700 updates in all 12 runs, so the held-out differences are not failed optimization.

**Held-out final accuracy at 10,000 updates**, per init seed 51200 / 51201 / 51202, with chance at 0.125:

| cell | all 128 rows | ¬N (42 rows no neighbour copy answers) | N (86 rows) |
|---|---|---|---|
| TT | 1.000 / 0.961 / 1.000 | 1.000 / 0.881 / 1.000 | 1.000 / 1.000 / 1.000 |
| TU | 0.633 / 0.523 / 0.516 | 0.238 / 0.071 / 0.048 | 0.826 / 0.744 / 0.744 |
| UT | 0.836 / 0.680 / 0.625 | 0.595 / 0.333 / 0.190 | 0.953 / 0.849 / 0.837 |
| UU | 0.766 / 0.609 / 0.703 | 0.381 / 0.190 / 0.262 | 0.953 / 0.814 / 0.919 |

**Paired differences in held-out final accuracy**, per seed:

| contrast | 51200 | 51201 | 51202 |
|---|---|---|---|
| tie transition, readout tied (TT − UT) | +0.16 | +0.28 | +0.38 |
| tie transition, readouts per-step (TU − UU) | −0.13 | −0.09 | −0.19 |
| tie readout, transition tied (TT − TU) | +0.37 | +0.44 | +0.48 |
| tie readout, transitions per-step (UT − UU) | +0.07 | +0.07 | −0.08 |
| interaction (TT − TU) − (UT − UU) | +0.30 | +0.37 | +0.56 |

On the ¬N rows the interaction is +0.55, +0.67 and +1.02.

**Trajectories.**
- Only TT met the 0.95-for-five-evaluations threshold: at 7,700, 9,150 and 8,200 updates.
- TU, UT and UU are below it at 10k, and all three were still rising: +0.07 to +0.14 held-out over the last 2,000 updates.
- At 1,000 updates, ¬N is at most 0.02 in every run, while N is 0.33–0.52. ¬N rises later than N in every cell, the two-phase pattern the reviewer found for TT in r1.

**Decoder across position** ([figures/decode_table.png](figures/decode_table.png), probe seed 64300). The table gives the mean off-diagonal entry among positions 1–3, a probe fit on h_i and scored on h_j of held-out programs. At `ckpt-0` it is 0.12–0.14 in every cell, and the `ckpt-0` table is identical across the four cells of a seed.

| cell | mean off-diagonal (1–3) at 10k | held-out h3 decodable at 10k |
|---|---|---|
| TT | 0.95–1.00 | 0.94–0.99 |
| UT | 0.75–0.84 | 0.67–0.87 |
| TU | 0.39–0.48 | 0.51–0.60 |
| UU | 0.32–0.37 | 0.59–0.75 |

**Readout across position** (`out/readout_table.csv`). In TU, readout V3 applied to h2 reads step2 at 0.52–0.58, which reproduces the reviewer. In UU it reads 0.36–0.45.

**Optimization and decay, descriptive** (`out/groups.csv`, `--diag`).
Checkpoints at 4k, 7k and 10k (after the fit) compare the decay step with the Adam-direction step.
- *Tied groups.* The decay step is 0.44–0.94 of the Adam-direction step, and the two point against each other (cosine −0.40 to −0.94). Readouts are closest to balance: 0.71–0.94 and −0.82 to −0.91 in TT.
- *Per-step transition copies, copy 3 most often.* The ratio is 0.10–0.74 and the cosine −0.11 to −0.75. Their Adam steps are less offset by decay.
- *Eps.* The fraction of coordinates with √v̂ < eps stays at 0.7–7.6% globally from 500 updates on, and at most 8.8% in any group. The Adam step is not eps-limited in any cell.
- *Copy drift.* Untied copies drift apart from their identical start. The cosine between transition copies is 0.26–0.41 at 10k, and between readout copies 0.19–0.37.

## Predictions (QUESTION.md) against outcomes

| # | prediction | outcome |
|---|---|---|
| 1 | UT, UU fit by 1,000 updates | held (350–550) |
| 2 | UT < TT on every seed by ≥ 0.10; UT in 0.30–0.90 | held (+0.16 to +0.38; UT 0.63–0.84) |
| 3 | UU ≤ TU on every seed; UU in 0.15–0.55 | **failed**: UU > TU on every seed by 0.09–0.19, and UU is 0.61–0.77 |
| 4 | super-additive interaction | held (+0.30 to +0.56) |
| 5 | cells below 0.70 have ¬N ≤ 0.35 | held (TU 0.05–0.24, UT 0.19 and 0.33, UU 0.19) |
| 6 | off-diagonal decoding TT ≥ 0.8, UU ≤ 0.5, UT and TU between | held (TU only just above UU) |
| 7 | untied-transition cells 10–20% slower per update | held (12–18%) |

## Interpretation of the initial comparison (bounded by this task, recipe and 10k budget)

1. **Only TT composed within 10k.** With both groups tied, held-out accuracy reaches 0.96–1.00, including the rows no neighbour copy answers. Untying either group, or both, gives 0.52–0.84 at 10k, with ¬N 0.05–0.60. All three of those cells were still rising, so these are finite-budget gaps, not plateaus.
2. **The two tyings are strongly non-additive.**
   - Tying the readout matters a lot when the transition is tied (+0.37 to +0.48), and little or inconsistently when the transitions are per-step (−0.08 to +0.07).
   - Tying the transition helps under a shared readout (+0.16 to +0.38), and under per-step readouts it does worse than not tying (−0.09 to −0.19 on every seed).
   - So neither tying, taken alone from UU, moved held-out accuracy up at 10k under this recipe. Only the combination did.
3. **A shared, cross-position decodable state code did not by itself give TT's composition.**
   - UT's shared readout yields a code that a probe fit at one position reads well at the others (0.75–0.84 off-diagonal), far above TU and UU.
   - Yet UT composes only partly (0.63–0.84).
   - Decodability is not use, and an untied model could also keep an invertible position-specific code that the table scores low.
4. **What this does not separate.** The intervention changes several things at once:
   - parameter count (UT and UU have 2.8× TT's);
   - the number of decayed parameters;
   - summed versus per-copy gradients;
   - per-copy data reuse (a tied transition is applied three times per row, an untied copy once).

   The TU < UU result could reflect a shared transition serving three positions whose readouts pull their codes apart, but nothing here tests that; it is speculation. Nothing here shows that sharing is necessary in general, or that TT implements a clean symbolic state machine.

## Limitations

- **Coverage.** One task draw, one split, three paired init seeds. This is exploratory coverage; seed-to-seed spread within the untied cells is as large as 0.21.
- **Budget.** Every value is a 10k-update value under two recipes (width 128, lr 1e-3, wd 0.1 or 0). Longer training is the time lane's; it continues TT and TU at wd 0.1, not UT, UU or any wd 0 run.
- **Probes.** Probes are linear, fit on 384 rows. Held-out h1 and h2 are training-prefix states (C1.1).
- **Shared evidence with the reviewer.** TU is the reviewer's r1 arm, so every contrast involving TU rests on the reviewer's runs.

## Follow-up: the same 2×2 without weight decay

Figure: [figures/followup_wd0.png](figures/followup_wd0.png). Decoder table: [figures/followup_decode_wd0.png](figures/followup_decode_wd0.png).

**Design.**
- Pre-registered in QUESTION.md ("Follow-up") at `d8735fc` and approved by the coordinator before launch.
- All four cells run at wd 0 (`c2-share-wd0-*`), 12 runs, with the recipe otherwise unchanged.
- Each wd 0 run starts from the same weights as its wd 0.1 counterpart and logs identical step-0 metrics (`out/checks.txt`).
- wd 0 is a different recipe, not a control that isolates "regularization": it changes the whole post-fit trajectory, and with it the effective step size. A change in a tying effect between the two recipes says only that the effect depends on decay under this recipe.

**Observations.**
- **Fit.** Every run fits training at all three steps by 350–600 updates.
- **Held-out final at 10,000 updates**, wd 0 against wd 0.1 (per seed 51200 / 51201 / 51202):

  | cell | wd 0: all 128 rows | wd 0: ¬N | wd 0 − wd 0.1 |
  |---|---|---|---|
  | TT | 0.398 / 0.383 / 0.453 | 0.048 / 0.071 / 0.024 | −0.60 / −0.58 / −0.55 |
  | TU | 0.297 / 0.297 / 0.320 | 0.000 / 0.000 / 0.048 | −0.34 / −0.23 / −0.20 |
  | UT | 0.336 / 0.336 / 0.305 | 0.000 / 0.000 / 0.000 | −0.50 / −0.34 / −0.32 |
  | UU | 0.297 / 0.312 / 0.305 | 0.024 / 0.000 / 0.000 | −0.47 / −0.30 / −0.40 |

- **Paired differences within wd 0.**

  | contrast | per seed |
  |---|---|
  | TT − UT | +0.06 / +0.05 / +0.15 |
  | TU − UU | 0.00 / −0.02 / +0.02 |
  | TT − TU | +0.10 / +0.09 / +0.13 |
  | UT − UU | +0.04 / +0.02 / 0.00 |
  | interaction | +0.06 / +0.06 / +0.13 |

  At wd 0.1 the interaction was +0.30 to +0.56.
- **Trajectories.**
  - Held-out accuracy at wd 0 reaches 0.23–0.34 by 1,000 updates and then rises slowly in every run: +0.02 to +0.11 from 1k to 10k (median +0.06).
  - Three runs rose more than 0.03 between 5k and 10k (reviewer CX-S2): TT 51202, UT 51201 and UT 51202.
  - The change over the last 2,000 updates is −0.01 to +0.03, against +0.03 to +0.14 at wd 0.1.
  - Held-out cross-entropy rises meanwhile: from 4.0–6.6 at 1k to 6.1–12.3 at 10k.
  - Above-chance accuracy at wd 0 is almost all on N rows (N 0.43–0.66, ¬N ≤ 0.07).
- **Optimizer.** The slow rise is not an eps stall.
  - Across all 12 runs at 10k, the median √v̂ is 6e-6 to 2e-5, far above eps. From 500 updates on, the fraction of coordinates below eps is 2–9%, as at wd 0.1.
  - Updates do not vanish (update norm 8e-4 to 5e-3 at 10k), and the training margin keeps growing (minimum 8.7–10.5 at 10k).
  - Parameter norms grow instead of shrinking. For example, at seed 51200 the TT transition norm is 29.0 at wd 0 against 19.1 at wd 0.1 (`out/groups_wd0.csv`).
- **Decoding at 10k** (`out/decode_table_wd0.csv`).
  - Mean off-diagonal entry among positions 1–3: TT 0.66–0.71, UT 0.57–0.64, TU 0.31–0.36, UU 0.28–0.33. The order is the same as at wd 0.1, but TT and UT are lower.
  - Held-out h3 is decodable at only 0.30–0.45 in every cell.
  - In TT and UT, probes fit on training h3 read the training-prefix states h1 and h2 at 0.70–1.00. The shared readout keeps training states in a common code without the held-out h3 joining it.

**Predictions F1–F4 against outcomes.**

| # | prediction | outcome |
|---|---|---|
| F1 | all 12 fit by 1,000 | held (350–600) |
| F2 | TT at wd 0 below wd 0.1 by ≥ 0.2 on every seed | held (−0.55 to −0.60) |
| F3 | spread across cells smaller at wd 0 on every seed | held (0.09–0.15, against 0.37–0.48) |
| F4 | TT highest at wd 0 on ≥ 2 of 3 seeds | held (3 of 3; lead over the next-best cell 0.05–0.13, which is UT at 51200/51201 and TU at 51202; reviewer S1) |

**Reading** (bounded by this recipe and 10k).
- Under this recipe, the large tying effects at 10k depend on decay. Without it, every cell stays at 0.30–0.45 held-out and answers almost none of the ¬N rows.
- The interaction shrinks from +0.30 to +0.56 to +0.06 to +0.13.
- TU < UU disappears: the two are within ±0.02.
- A small TT advantage remains on every seed: 0.05–0.13 over the next-best cell.
- This does not show that decay is the mechanism of composition, or that composition is impossible without decay. The wd 0 cells still rise slowly at 10k, so 10k is not a plateau, and how far they would rise is untested here. And wd 0 changes the trajectory as a whole, not one ingredient of it.
