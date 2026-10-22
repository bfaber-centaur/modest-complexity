# c2 mechanisms: does the hidden state act as a reusable semantic state?

Lane: mechanisms (thread "Hidden-state interchange"), role file [charter/05-mechanisms.txt](../coordinator/charter/05-mechanisms.txt), assignment DECISIONS D7. Written before any measurement in this lane. Adaptive choices made later go in [ADAPTIVE.md](ADAPTIVE.md), not here; the follow-up is appended below as its own section when proposed.

## Question

In the r1 RNNs, is the hidden state after k ops a state that later ops use according to its symbol alone (reusable across histories and positions), or does it carry history or position information that changes what later ops do?

## Inputs (no training)

The recovered r1 checkpoints (DECISIONS D2), verified by SHA-256 against [recovered/RECOVERY.json](../recovered/RECOVERY.json) before use. Task `ops51000-ls51100`, init seeds 51200–51202 (paired across conditions, not new seeds).

| condition | runs | transition | readout | step weights |
|---|---|---|---|---|
| tied/inter | `r1-comp-rnn-inter-s5120{0,1,2}-a1` | shared | shared `V` | 1 1 1 |
| per-step readout/inter | `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1` | shared | `V1`, `V2`, `V3` | 1 1 1 |
| tied/final | `r1-comp-rnn-final-s5120{0,1,2}-a1` | shared | shared `V` | 0 0 1 |

Primary snapshot: `ckpt-10000`. `ckpt-0` and `ckpt-1000` are analyzed with the same code as secondary snapshots. Later states (20k–50k) come only from the time lane's committed checkpoints, for the follow-up.

## What whole-state interchange can and cannot show here (derived before measuring)

Notation: depth i means the state h_i after i ops (h_0 is the start embedding). A **donor** contributes h_i of its prefix (s', o'_1..o'_i). A **base** (recipient) is a length-3 row (s, o_1, o_2, o_3); its h_j is replaced and its suffix o_{j+1}..o_3 is run through the original network. The patched answer is read with the recipient's final readout.

1. **Every condition has one shared transition** (A, B, bh). So the patched final state is exactly the final state of the network run on the recombined program `donor prefix ++ base suffix`, of length L = i + 3 − j, from the donor's start symbol. Whole-state interchange at a step boundary is therefore evaluation on recombined inputs, for same-step (i = j, L = 3) and cross-position (i ≠ j, L = 0..6) swaps alike. It is not an independent causal method on top of input evaluation. `test_mech.py` checks the equivalence bitwise.
2. **Consequence for the matrix.** With the readout fixed by the recipient (below), cells with equal i − j enumerate the same set of (start, program) pairs with the same readout. So the suffix-success matrix S(i, j) is constant along diagonals and is the accuracy on all programs of length L = 3 + i − j. The 16 cells hold 7 distinct values (L = 0..6). This is checked numerically and reported, not counted as a finding.
3. **Overlap with the generalization lane.** For tied/inter and tied/final, S at L = 1..6 is the same quantity as the generalization lane's frozen length 1–6 evaluation. Agreement between lanes is a cross-check, not two pieces of evidence.
4. **The base does not affect the patched answer** except through its suffix. So for a fixed (donor prefix, suffix) the outcome is the same whether the base shares the donor's symbol or not. Under the all-pairs weighting below, same-state success equals all-pairs success exactly. History dependence is therefore measured by **consistency** across donors with the same symbol (below), and by interventions that are *not* input recombinations: semantic centroids and random/magnitude-matched states.
5. **Same-step swaps** (i = j) are evaluation on the 512 length-3 rows: 384 training rows and 128 held-out rows. They repeat ordinary evaluation and are shown only as the diagonal, not as separate evidence.

## Readout map for the per-step-readout condition (declared)

Recipient frame: the suffix runs with step indices j+1..3 (the transition is shared, so only the readout index matters) and the answer is read with `V3`, the readout the base uses for its answer. For i ≠ j, `V3` sees a state after 3 + i − j transitions instead of 3. That is an explicit distribution shift, not a fair default expectation of invariance. As a secondary, descriptive column, the same patched states are also read with `V1` and `V2`.

## Measures

All computed exhaustively; no sampling except the random controls.

**M1. Ordinary evaluation, verified.** For all 27 recovered checkpoints, a forward pass with `c2rnn.forward` must reproduce the published r1 metrics line at that step exactly: per-head train and held-out accuracy, cross-entropy to float equality, and the 128-character `held_final_pred`. Labels are checked against an independent loop over r1's `execute`. Rows are also labelled N / ¬N (`c2task.neighbour_answerable`, 86 / 42 rows).

**M2. Natural interchange, primary.**
- **Pairs.** Predeclared from task structure: every ordered pair (base row, donor row) of the 512 length-3 rows (64 programs × 8 starts, training and held-out). That is 262,144 pairs for each donor depth i ∈ {0,1,2,3} and recipient depth j ∈ {0,1,2,3}. The donor contributes h_i of its own row's prefix.
- **Target.** The oracle suffix applied to the donor's symbol at depth i: P[o_3]∘…∘P[o_{j+1}](x_donor).
- **Suffix success S(i, j).** The fraction of pairs whose patched argmax equals the target. Also reported: the base-retained rate (patched answer = the base's own true answer).
- **Pair categories.**
  - *self*: same row, i = j. Must equal unpatched correctness bitwise.
  - *same prefix*: i = j and the same prefix.
  - *same state, different history*: the donor's symbol at i equals the base's symbol at j, prefix different. The target is then the base's true answer.
  - *different state*: the target differs from the base's answer, because the suffix is a bijection.
- **Subsets.**
  - (a) all pairs;
  - (b) pairs whose base and donor rows both have a correct unpatched final prediction;
  - (c) for i = 3, donor histories from training programs vs held-out programs, which are unseen histories;
  - (d) for i = j, recombined program in training vs held out.
- **History consistency C(i, j).** Group donor prefixes by (depth i, symbol x). For each suffix, take the fraction of pairs of distinct histories in the group whose patched predictions agree, then average over groups and suffixes. Undefined at i = 0 (one history per symbol). C = 1 with S < 1 means consistently wrong (state-like but a wrong suffix computation); C < 1 means the history changes the answer. For i = 3, also report agreement between held-out-history and training-history donors with the same symbol.

**M3. Non-natural interventions (not equivalent to inputs).**
- **Semantic centroid.** c_i(x) is the mean of h_i over the 384 training rows with symbol x at depth i. It is fit on training states only. Patch c_i(x) into recipient depth j and run each suffix. S_cent(i, j) is the fraction equal to the oracle suffix applied to x. Unlike M2, this is not constant along diagonals by construction.
- **Random/magnitude-matched controls,** for every (donor prefix, suffix), generator seed 64301:
  - (R1) the donor's own h_i with its 128 coordinates permuted, which keeps the norm, sparsity and values;
  - (R2) each unit drawn independently from its values over the training-row states at depth i, which keeps the per-unit marginals.
  - Measures: the target-hit rate against the paired real donor's target (chance 1/8) and the base-retained rate.
  - They calibrate "generic disruption": a real donor's effect counts only beyond these.

**M4. Descriptive geometry and decoding.**
- **History spread** at each depth: the within-symbol mean squared distance to c_i(x), over the mean squared distance between centroids. Training rows; plus held-out depth-3 states against training centroids.
- **Cross-depth centroid offset:** ||c_i(x) − c_j(x)|| over the mean between-centroid distance at depth j.
- **The D5 decode table** (`c2probe.decode_table`, unchanged, probe seed 64300), fit on training-program h_i and scored on held-out-program h_j. Held-out h_0..h_2 are training prefixes (r1 C1.1), so only h_3 is an unseen-history state. All of this is labelled descriptive: decodability is not use.

## Predictions (before measuring)

1. M1 reproduces every published number exactly.
2. **Diagonals.** S(i, i) is overall length-3 accuracy:
   - about (384 + 128·acc_held)/512;
   - tied/inter ≥ 0.99;
   - per-step readout 0.87–0.91;
   - tied/final 0.81–0.83.
3. **tied/inter.**
   - L = 1, 2 (i < j, j < 3): ≥ 0.95, because the shared readout was trained on h_1, h_2.
   - L = 0 (`V` on h_0, never trained): ≤ 0.5.
   - L = 4: ≥ 0.8. L = 5, 6: lower than L = 4, anywhere in 0.3–0.95.
   - C ≥ 0.95 wherever S ≥ 0.95.
   - S_cent within 0.05 of natural S at every L ≤ 4.
   - History spread at depths 1–3 under 0.1.
   - This would support a reusable semantic state within the tested lengths.
4. **tied/final.**
   - Off-diagonal S near chance, ≤ 0.25 at every L ≠ 3. h_1 and h_2 are unreadable by `V` (r1: 0.06–0.15), and nothing trained the state for L > 3.
   - C clearly below 1 at depths 1–2 for suffixes of length ≥ 1, meaning history-dependent.
   - S_cent below natural S on the diagonal.
5. **Per-step readout.**
   - j = 3 column: `V3` on h_2 at 0.5–0.6 (reviewer r1: 0.52–0.58), and lower on h_1 and h_0.
   - L = 4–6: between tied/final and tied/inter.
   - On held-out-history depth-3 donors, S and C are lower than on training-history donors.
6. **Controls.** Target-hit is 0.05–0.25 in every cell and condition. Real-donor S in tied/inter exceeds it by ≥ 0.5 wherever S ≥ 0.8.
7. **Snapshots.**
   - ckpt-0: S near chance off the diagonal; train-row diagonal low.
   - ckpt-1000: tied/inter off-diagonal S still low. Its held-out accuracy was 0.34–0.36 and mostly neighbour-like (r1 counterexample 1).

**Bound on any conclusion.** "Reusable semantic state under these interventions" is claimed only for the condition, lengths and snapshot where S, C and S_cent are all high and the controls are near chance. Low S does not show the absence of a semantic code: a position-specific or nonlinear code with a mismatched readout is possible. Finite-training results are not necessity claims.

## First figure (plan)

`figures/interchange.png` + `.svg` + `interchange.csv`.
- **Left 3×3 block** (rows: tied/inter, per-step readout/inter, tied/final; columns: seeds 51200–51202). Each panel is a 4×4 heatmap of S(i, j) at 10k, donor depth on the y-axis and recipient depth on the x-axis, all pairs, annotated with S and, underneath, C.
- **Right column,** one panel per condition: S, S_cent, target-hit of R1 and R2, and C against L = 0..6, one line per seed, with chance at 1/8 marked.
- The correct-unpatched subset and the held-out-history split go in the CSV and a NOTE table. If they differ visibly from all pairs, they become a second panel row, not a separate figure.

## Run IDs, seeds, budget

- **Run IDs:** `c2-mech-eval-a1` (M1), `c2-mech-interchange-a1` (M2–M3 and geometry, all 27 checkpoints) and `c2-mech-probe-a1` (D5 tables, 9 checkpoints at 10k). They run through `run_jobs.py` with registry `mechanisms/registry.jsonl`.
- **Seeds:** probe 64300; controls 64301.
- **Budget** is 15 min elapsed with a job running (D6). The analyses are forward passes over at most 32,768 rows per cell, so the expected total is a few minutes. A short calibration run is reported first. The 80% mark is 12 min.

## Follow-up candidates (one, chosen after M2 with the coordinator)

- **A. Emergence trace.** M2–M3 on the time lane's committed 20k–50k states, together with 0/1k/10k, to trace when cross-length S, C and S_cent rise in each condition. No training.
- **B. Candidate state subspace.** Use the span of the centered training centroids (≤ 7 dimensions, pooled over depths 1–3, fit on training rows). Swap the donor's projection into the base while keeping the base's complement; this is base-dependent and not an input recombination. Compare against matched random subspaces of equal dimension and the complement swap, evaluated on held-out-history and cross-position pairs. A negative result would not rule out a nonlinear or redundant code.

## Literature

Liška, Kruszewski and Baroni (2018), [arXiv:1802.06467](https://arxiv.org/abs/1802.06467), introduce lookup-table composition. With extra supervision, standard RNNs could reach compositional solutions. Without it, a small share of initializations reached partial ones.

The interchange framing follows the general causal-abstraction literature. This lane makes no novelty claim and has not done a targeted literature check.

## Follow-up (proposed after M2, 2026-10-22; approved by the coordinator 04:18Z as written)

**What M2 showed** (README, M2 observations): in all three conditions, replacing a state by its training-row class mean (the semantic centroid) does better than the natural state where the natural state fails:
- tied/inter at L = 5–6;
- the depth-2 → op 3 step on held-out programs, in per-step readout and tied/final.

So the history-specific part of the state goes with the failures. M2 cannot tell which part of that variation matters, or whether the class-mean directions carry the symbol causally on held-out rows.

**Proposal (candidate B, refined).** Use the nine 10k checkpoints. Fit everything on training-row states only: the centroids c_i(x), and U_i, the span of the 7 centered centroids at depth i. Evaluate at boundary i = 2 on all 512 rows: patch, then op 3, then the recipient readout. Score held-out programs and training programs separately. In tied/inter, also evaluate at boundary i = 3 followed by 1–3 more ops (L = 4–6, all programs).

The history residual is r = h_i − c_i(x), with x the row's true symbol. The patched states are:
1. **Dose.** h' = c_i(x) + λ r, for λ ∈ {0, 0.25, 0.5, 0.75, 1}. λ = 1 is natural and λ = 0 is the M2 centroid.
2. **Where the harmful residual lives.**
   - h' = c_i(x) + Π_U r keeps only the residual inside U.
   - h' = c_i(x) + Π_⊥ r keeps only the residual outside U.
   - Matched controls repeat both with five random 7-dimensional subspaces (seed 64302).
3. **Held-out subspace interchange.**
   - h' = h_b + Π_U (h_d − h_b) is applied to held-out-program bases at i = 2. Donors are all depth-2 states with a different symbol.
   - The target is the oracle op 3 on the donor's symbol.
   - Controls: the same with the five random 7-dimensional subspaces, and with the complement Π_⊥.

**Predictions.**
- **tied/inter.** Accuracy rises monotonically as λ → 0. The harmful residual is mostly outside U: keeping Π_⊥ r reproduces the natural errors, and keeping Π_U r does not. U-interchange follows the donor in ≥ 0.9 of held-out pairs; random subspaces do so in ≤ 0.2.
- **per-step readout and tied/final.** The same direction at i = 2, with U-interchange lower (0.4–0.8).
- **Not predicted.** Whether the residual inside U also harms.

**Limits.** The centroid uses the row's true symbol, so a high λ = 0 accuracy is not something the network achieves on its own. A negative subspace result would not rule out a nonlinear or redundant code. U is fit on training rows, and its 7 dimensions are fixed by the 8 classes.

**Cost and IDs.** Forward passes only, under 1 min. Run ID `c2-mech-residual-a1`, seed 64302. No training.

**Alternative (A, if preferred).** Run M2–M3 on the time lane's committed 20k–50k checkpoints, to trace whether cross-length S, C and S_cent rise with training in the per-step-readout and tied/final arms. It depends on those checkpoints existing.
