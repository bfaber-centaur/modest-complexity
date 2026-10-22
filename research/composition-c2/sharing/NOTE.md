# c2 sharing lane: note (initial comparison; follow-up pending)

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

## Interpretation (bounded by this task, recipe and 10k budget)

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
- **Budget.** Every value is a 10k-update value under one recipe (width 128, lr 1e-3, wd 0.1). Longer training is the time lane's; it continues TT and TU, not UT or UU.
- **Probes.** Probes are linear, fit on 384 rows. Held-out h1 and h2 are training-prefix states (C1.1).
- **Shared evidence with the reviewer.** TU is the reviewer's r1 arm, so every contrast involving TU rests on the reviewer's runs.

## Follow-up

The follow-up is proposed in QUESTION.md ("Follow-up") and awaits the coordinator's approval. Its indicators were declared in advance. The eps indicator does not hold. The decay/gradient balance indicator holds for the tied groups after the fit, and less closely for per-step transition copies. The proposal is therefore the matched no-decay comparison: all four cells at wd 0.
