# c2 mechanisms: note (M4)

**Question** ([QUESTION.md](QUESTION.md)). In the r1 RNNs, does the hidden state after k ops act as a state that later ops use by its symbol alone? Or does it carry history or position information that changes what they do?

**Scope.**
- **Data.** One task draw and split (`ops51000-ls51100`) and three paired init seeds (51200–51202).
- **Models.** Three recovered conditions, all with one shared transition, mainly at 10,000 updates:
  - tied/inter: shared readout, intermediate supervision;
  - per-step readout/inter: readouts `V1` `V2` `V3`, intermediate supervision;
  - tied/final: shared readout, final-only.
- **No training.** Every number is in `runs/` (IDs below) or `figures/*.csv`.
- **Commands, budget and run manifest:** [README.md](README.md). Adaptive choices: [ADAPTIVE.md](ADAPTIVE.md).

## Observations

**O1. The recovered checkpoints are the r1 runs** (`c2-mech-eval-a1`).
- All 27 files (3 conditions × 3 seeds × 0 / 1k / 10k) match their SHA-256.
- Each reproduces its published r1 metrics line exactly.

**O2. Whole-state interchange is input recombination here** (`c2-mech-interchange-a1`, `checks.json`).
- **Why.** The transition is shared, so replacing h_j of a base row with a donor's h_i gives, bitwise, the final state of the spliced program `donor prefix ++ base suffix`, of length L = 3 + i − j.
- **Consequences.**
  - The donor-depth × recipient-depth matrix is constant along diagonals. That is verified as identical predictions for every (start, program).
  - Same-step swaps are ordinary length-3 evaluation.
  - Same-state and different-state donors give the same suffix success, because the base enters only through its suffix.
- **What this means.** Natural interchange adds no evidence beyond evaluating the network on recombined programs. For tied/inter and tied/final, its L = 1..6 values are the generalization lane's frozen length evaluation, and agreement between the lanes is a cross-check.

**O3. Natural states, by length** ([figures/interchange.png](figures/interchange.png); seeds 51200 / 51201 / 51202).

| condition | L ≤ 2 | L = 3 | L = 4 | L = 5 | L = 6 |
|---|---|---|---|---|---|
| tied/inter | 1.00 | 1.00 / 0.99 / 1.00 | 0.99 / 0.93 / 0.98 | 0.95 / 0.74 / 0.88 | 0.86 / 0.56 / 0.72 |
| per-step readout (`V3`) | 0.12–0.62 | 0.91 / 0.88 / 0.88 | 0.20–0.26 | 0.15–0.18 | 0.13–0.14 |
| tied/final | 0.00–0.25 | 0.82 / 0.83 / 0.83 | 0.12–0.14 | 0.11–0.12 | 0.12–0.13 |

- **L = 0** is the readout applied to the start state h_0. The tied/inter readout was never trained on h_0, yet reads it at 1.00. That was predicted ≤ 0.5.
- **Consistency across same-symbol histories** (C).
  - tied/inter: C is ≥ 0.966 where S ≥ 0.97, and falls with S at long lengths, to 0.43–0.77 at L = 6. Those errors therefore depend on the donor's history and are not a consistent wrong answer.
  - The other two conditions: C is 0.16–0.58 off the diagonal.
- **Held-out histories.** In tied/inter, depth-3 states from held-out programs behave like those from training programs: L = 6 agreement between them is 0.41–0.75, against within-group C of 0.43–0.77.

**O4. Class-mean states do better than natural ones where natural ones fail** ([figures/centroid.png](figures/centroid.png)). c_i(x) is the mean of training-row h_i with symbol x. Placing it at recipient depth j is not an input recombination.
- **tied/inter.** S_cent ≥ 0.939 in all 16 cells of every seed, including L = 6.
- **per-step readout.** The depth-2 mean followed by op 3 is right for all 32 (symbol, op) pairs, at 1.00 in every seed. Natural states score 0.52–0.63 on held-out programs. `V3` reads the depth-2 mean's symbol at 0.88–1.00, against 0.52–0.58 for natural h_2.
- **tied/final.**
  - The depth-2 mean followed by op 3 scores 0.88 / 0.88 / 0.97, against 0.27–0.32 natural on held-out programs.
  - At depth 1 the mean does worse than natural: 0.65–0.77 against 0.82–0.83.
  - Off the diagonal, both stay at chance.
- **Controls.** Permuted-donor and per-unit-marginal states hit the target at 0.00–0.28 in cells with ≥ 32 combinations, near chance of 1/8.
- **What the class-mean gain consists of** (reviewer counterexample CX-M1, `review/REVIEW.md` §4). It covers every cell where S_cent − S ≥ 0.05 and asks how much of the gap two changes to the natural donors recover: (a) keeping only donors nearest their own symbol's training centroid, and (b) rescaling each donor to its centroid's norm.
  - **tied/inter, L = 4–6 (11 cells).** (a) recovers 1–22%, so the gain there is not injecting the right symbol. (b) recovers 25–63%, so a large part of what the class mean fixes is the state's overall scale.
  - **per-step readout and tied/final, held-out depth-3 states, cell (3, 3).** (a) recovers 64–86%, so the gain there is mostly the injected symbol.
  - **The same two arms, cell (2, 2).** (a) recovers 10–59%.

**O5. Follow-up: residual dose** ([figures/followup.png](figures/followup.png), `c2-mech-residual-a1`). The patched state is c(x) + λ·(h − c(x)), on held-out programs. It stays in the ReLU state set (a convex combination of two non-negative states).

| condition, boundary | λ = 1 (natural) | 0.75 | 0.5 | 0.25 | 0 (class mean) |
|---|---|---|---|---|---|
| per-step readout, depth 2 → op 3 | 0.52–0.63 | 0.65–0.74 | 0.82–0.87 | 0.95–0.98 | 1.00 |
| tied/final, depth 2 → op 3 | 0.27–0.32 | 0.39–0.43 | 0.54–0.56 | 0.73–0.77 | 0.88–0.97 |
| tied/inter, depth 3 → 3 more ops (L = 6, all programs) | 0.56–0.86 | 0.70–0.94 | 0.82–0.98 | 0.91–0.99 | 0.94–0.99 |

- **Monotone.** Accuracy rises monotonically in every seed as the residual shrinks.
- **Training programs.** In per-step readout they stay ≥ 0.99 at every λ. In tied/final they *fall* from 1.00 to 0.88–0.97 as λ → 0, so some training rows need their history residual.

**O6. Follow-up: where the residual lives is unresolved** (`c2-mech-residual-a1`, plus the post-hoc descriptive check `c2-mech-energy-a1`). U is the 7-dimensional span of the centered training centroids at depth 2. U is centered, so a pure scale residual (r ∝ c(x)) lies largely outside it. A "harmful residual outside U" would therefore partly restate the scale effect in O4.
- **Keeping one part of the residual hurts.** Keeping only the part inside U, or only the part outside it, does worse on held-out programs than keeping the whole residual, in every condition:
  - tied/inter: 0.60–0.68 inside U and 0.69–0.80 outside, against 0.96–1.00 for the whole residual;
  - per-step readout: 0.41–0.53 inside and 0.31–0.36 outside, against 0.52–0.63;
  - tied/final: 0.30–0.42 inside and 0.13–0.16 outside, against 0.27–0.32.
- **Why the controls do not settle it** (post-hoc check `c2-mech-energy-a1`).
  - The split states leave the ReLU state set: 7–12% (inside) and 16–21% (outside) of their coordinates are negative, against none for natural and class-mean states.
  - U holds 22–31% of the residual energy, against 5–7% for the dimension-matched random subspaces.
- **Held-out subspace interchange** (14,336 pairs per checkpoint). Swapping only the donor's U-coordinates into held-out-program bases moves the answer to the donor's target in:

  | condition | swap U only | random 7-d subspaces |
  |---|---|---|
  | tied/inter | 0.50–0.59 | ≤ 0.02 |
  | per-step readout | 0.35–0.38 | ≤ 0.10 |
  | tied/final | 0.26–0.29 | ≤ 0.13 |

  - Whole-state swaps follow the donor in 0.99–1.00, 0.88–0.91 and 0.82–0.83 of pairs.
  - Swapping everything *except* U follows the donor in only 0.06–0.16.
  - For the residual, random subspaces carry about a fifth of U's share of the energy. The share of the donor–base difference was not measured.

## Interpretation (bounded by these tasks, this recipe and these checkpoints)

1. **tied/inter: a reusable semantic state, to a degree.**
   - **Natural states.** Under whole-state interchange, states from any depth and history act as their symbol for later ops up to L = 4 (S ≥ 0.93).
   - **Class means.** Replacing a state by its training class mean fixes most cross-length failures in tied/inter. Matching only the donor's norm recovers a quarter to two thirds of that gain (CX-M1). The residual h − c(x) is therefore not a purely history-specific direction; a large part of it is overall scale.
   - **Dose.** Shrinking the residual removes the L = 5–6 errors (O5).
   - **Reading.** This supports "reusable semantic state up to L = 4; beyond the trained length, deviations from the class mean, much of them in scale, go with the errors".
   - **Limit.** It does not show that the network computes with an explicitly symbolic state. The class mean is our construction and uses the true symbol.
2. **per-step readout and tied/final: history-dependent computation.**
   - **Position.** Natural states are not reusable across positions. Only the readout index each was trained with reads them, and cross-length success is near chance.
   - **Held-out programs.** Within the trained length, replacing the depth-2 state by its class mean fixes the held-out failures: op 3 then gives the right answer for every (symbol, op) pair in per-step readout and for 88–97% of them in tied/final. Part of that gain is injecting the right symbol: at (2, 2), nearest-centroid donors recover 10–59% of it, and at (3, 3) 64–86% (CX-M1).
   - **Reading, with its limit.** At depth 2 every (symbol, op 3) pair occurs in training, and the class mean is an average of training states that the network maps correctly. So "the transition maps the class mean correctly" is close to true by construction. Nor is it something the network can do itself: the mean is our intervention, and it uses the true symbol. The bounded reading is that on unseen programs these networks' natural states are dominated by variation away from the class mean, which fits r1's finding that their wrong answers are neighbour labels. It does not show that they "know" the symbol and are distracted.
3. **Which part of the residual matters is unresolved.** The linear decomposition is confounded by off-manifold states and by controls that are matched in dimension, not in size.
   - **What the U swap shows.** Swapping the U coordinates moves 0.50–0.59 (tied/inter), 0.35–0.38 (per-step) and 0.26–0.29 (tied/final) of held-out answers to the donor's target. Dimension-matched random subspaces move ≤ 0.13, but they carry less energy, so these rates are not a measured share of the causal effect.
   - **What a negative result cannot show.** A negative subspace result here would not rule out a nonlinear or redundant code.
4. **Strong length-3 accuracy alone does not settle the question,** as the charter says. Tied/inter's held-out length-3 accuracy is near 1, and its natural states still fail at L = 6. The per-step and final-only arms fail at length 3, yet hold a usable class-mean state.

## Limitations

- **Coverage.** One task draw and split, three paired init seeds, and the 10k snapshot (0 and 1k only in `cells.csv`). All conditions were still improving at 10k (r1; the time lane continues them).
- **Interchange.** For these models, natural interchange is input recombination (O2). The centroid and dose interventions are the only non-equivalent ones, and both use the row's true symbol.
- **Readout map.** The per-step-readout results off L = 3 use `V3`, a declared distribution shift. `V1`/`V2` read only their own depth (`S_V1`, `S_V2` in `cells.csv`).
- **Linear tools.** Every subspace and probe here is linear. The D5 decode table (`c2-mech-probe-a1`) is descriptive: held-out h_3 decodes at 0.94–0.99, 0.51–0.60 and 0.29–0.34.
- **Small cells.** Controls in cells with 8–32 combinations are noisy (0.00–0.38).
- **Scored predictions.** Several QUESTION.md predictions failed or were mixed (README table). Notably, the shared readout reads h_0, and history spread is above 1, not below 0.1.

## Literature

The task family is lookup-table composition (Liška, Kruszewski and Baroni 2018, arXiv:1802.06467). This note reproduces nothing from it and claims no novelty; no targeted literature check was done.

## Reviewer handoff

The reviewer checked 817a768 in `review/REVIEW.md` §4 (PR #56). Every number it checked passed. Its wording points are addressed in this version: CX-M1 in O4 and Interpretation 1, M2 in Interpretation 2, M3 in Interpretation 3, and the post-hoc label in O6.

1. **O2 equivalence.** `test_mech.py` (`test_patch_is_recombined_forward`, plus the per-step-transition guard) and `checks.json` for all 27 checkpoints. Is "adds no evidence beyond recombined evaluation" stated correctly and not overstated?
2. **Pair aggregation** (`pairs.csv`). The claim that same-state S equals all-pairs S, the `both_correct` subset, and the held-out-history split.
3. **Centroid and dose semantics.**
   - Centroids are fit on training rows only.
   - At depth 2 every (symbol, op 3) pair occurs in training, so the class-mean result is not an unseen-pair test.
   - λ = 1 reproduces natural predictions exactly (`checks.json` of `c2-mech-residual-a1`).
   - λ = 0 over all rows equals M2's S_cent(2, 2) exactly in all nine checkpoints (README).
4. **O6 reading.** Do the energy and negativity numbers (`energy.csv`) justify "unresolved" rather than "the harmful residual is outside U"?
5. **The per-step readout map** (`V3`, recipient frame), and the reviewer's own role in that arm (it wrote the r1 per-step-readout runs).
6. **Overlap with the generalization lane.** Its frozen length 1–6 numbers for tied/inter and tied/final should equal O3's by construction.
