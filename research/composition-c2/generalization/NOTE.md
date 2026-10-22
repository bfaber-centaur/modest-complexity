# c2 generalization lane: note (M4)

**Question.** r1's RNN with one transition and one readout shared across positions, trained with intermediate supervision (tied/inter), answered 96–100% of 16 held-out length-3 programs at 10k updates, on one permutation draw and one split. Does that success extend to programs longer than any trained, and does it reproduce on new permutation draws and new splits?

Predictions, measures and the draw rule were committed before any evaluation ([QUESTION.md](QUESTION.md), e5a4415); the follow-up was proposed and approved before it ran (99c83c8). Adaptive choices: [ADAPTIVE.md](ADAPTIVE.md). Commands, run manifest and budget: [README.md](README.md). Every number below comes from the committed run outputs in `runs/` or the figure CSVs in `figures/`.

## Design in brief

- **Part A, frozen, no training.** The recovered r1 `ckpt-10000` of tied/inter and tied/final (init seeds 51200–51202), on every program of lengths 1–6 with every start symbol (8·4^L rows; L = 6 is 32,768 rows). Row, program and start-symbol weighting coincide. Labels come from `c2task.rows` and are checked against a separate per-row loop.
- **Part B, new training.** 2 new permutation draws × 2 Latin-square splits × 2 init seeds × {tied/inter, tied/final} = 16 runs of r1's recipe (width 128, full-batch Adam lr 1e-3, decoupled wd 0.1, 10k updates, `c2train.py` unchanged, every flag explicit). The same init seeds (63200, 63201) are used on every split, so init is crossed with task. Each run's `ckpt-10000` then gets Part A's frozen evaluation on its own task.
- **Follow-up, frozen.** Where do the 11 tied/inter models (3 original, 8 new-task) first go wrong at steps 4–6 of length-6 programs?
- **Exposure.** Tied/tied: 19,208 parameters, 384 training rows (48 programs × 8 starts), 10,000 full-batch updates = 3.84M row presentations. inter supervises 3 labels per row, final 1.

## Observations

### Part A: frozen r1 models on longer programs

Figure: [figures/length.png](figures/length.png). Reviewer re-derived every Part A number with independent code (review/REVIEW.md, PR #53: PASS; G1 and G2 below).

**Checks.** Every check in `runs/c2-gen-frozen-orig-a2/checks.json` holds.
- At length 3, each model's 128-character held-out prediction string and its train/held-out final accuracy equal r1's logged step-10000 row.
- Length-1/2 accuracy equals r1's logged training step1/step2 accuracy. These lengths were labelled through intermediate supervision (inter) or never labelled (final), so they are not compositional transfer.
- The step-t accuracy on length-6 rows equals the length-t accuracy exactly (the RNN is causal).
- The oracle loop agrees with the labels on every row.

| final-answer accuracy, all rows | L=1 | L=2 | L=3 all (held-out only) | L=4 | L=5 | L=6 |
|---|---|---|---|---|---|---|
| tied/inter 51200 | 1.000 | 1.000 | 1.000 (1.000) | 0.994 | 0.950 | 0.856 |
| tied/inter 51201 | 1.000 | 1.000 | 0.990 (0.961) | 0.926 | 0.744 | 0.560 |
| tied/inter 51202 | 1.000 | 1.000 | 1.000 (1.000) | 0.983 | 0.875 | 0.718 |
| tied/final 51200 | 0.125 | 0.148 | 0.818 (0.273) | 0.136 | 0.123 | 0.126 |
| tied/final 51201 | 0.094 | 0.148 | 0.828 (0.312) | 0.120 | 0.115 | 0.126 |
| tied/final 51202 | 0.062 | 0.086 | 0.830 (0.320) | 0.136 | 0.118 | 0.124 |

Chance is 0.125. The "all" column at L = 3 is 75% training rows; the held-out value is the like-for-like comparison with L = 4–6 (reviewer G2).

**What the longer programs are** (task facts, `gen_task.py`). The four ops generate all of S8. At L = 4 all 256 products are distinct and none equals a function of length ≤ 3. Collisions start at L = 5 (18 programs in collision groups) and grow at L = 6 (372). At L = 6, 4,086 of 4,096 programs are new functions; only 10 compute a function seen at length ≤ 3, which is too few to compare. So lengths 4–6 are new syntax *and*, almost always, new input-output functions on this draw.

**At 10k, tied/inter keeps answering past the trained depth, and degrades with length.**
- Accuracy is 0.93–0.99 at L = 4 and 0.56–0.86 at L = 6, in the same seed order as at length 3: 51201 is the weakest, and it was also the seed with 5 held-out errors at L = 3.
- The per-step error rate rises with depth in every seed. It is the share of rows still correct after step t−1 that go wrong at step t. For 51200 it is 0.6% at step 4, 4.6% at step 5 and 11.6% at step 6; for 51201, 6.9%, 21.9% and 32.8%.
- **Held-out windows (reviewer G1, confirmed here in `runs/c2-gen-followup-anatomy-a1/windows.csv`).** Programs that contain any held-out length-3 window, at any position, are less accurate in every seed: by 0.01–0.10 at L = 4 and by 0.05–0.14 at L = 5–6. Programs whose length-3 *prefix* is held out are lower too (L = 6: 0.81 vs 0.87, 0.50 vs 0.58, 0.65 vs 0.74).
- **Drift, descriptive.** Hidden-state norm and distance to the training h3 centroid of the true symbol grow beyond step 3. At step 6 they are 1.02–1.17× and 1.03–1.19× their training-h3 medians.
- **After a first error at step 4 or 5,** the next step is right again in 27–38% of rows. In 38–41% it applies the next op correctly to the model's own wrong answer, and in 21–34% it does neither.

**tied/final does not extend.** Accuracy is 0.11–0.14 at every length ≥ 4, near chance, and its untrained readout at h1, h2 is near chance at L = 1–2. Its states drift further: at step 6 the norm is 1.4–1.6× and the centroid distance 1.5–1.7× their training-h3 medians.

**At 50k updates the length decline is gone for tied/inter (A5, `runs/c2-gen-frozen-50k-a1`).** These are the time lane's exact continuations of the same three runs: same recipe, length-3 training only, full Adam state, ckpt-10000 → 50,000. Their `ckpt-50000` came from branch `claude/project-thread-xb3nfz` at 543ed95; hashes match the time lane's manifests, and every check holds against their own logs.
- tied/inter answers **every row of every length 1–6 correctly in all three seeds** (1.000 on 32,768 rows at L = 6). At step 6 the hidden-state norm and centroid distance are 0.99–1.00× their training-h3 medians, against 1.02–1.19× at 10k.
- tied/final's length-3 held-out accuracy rose to 0.56 / 0.70 / 0.66 (¬N 0.12 / 0.26 / 0.14), but lengths 4–6 stay at 0.10–0.15.
- The reviewer evaluated the time lane's intermediate checkpoints (G6, PR #59): the decline was already gone by 20k, with L = 6 at 0.9996 / 0.986 / 1.000. That is before the time lane's Adam-eps regime begins at about 25k.
- So the 10k decline, the held-out-window effect and the follow-up's error anatomy describe the 10k models. They are finite-time properties of this recipe, not limits on these architectures.

**Reference systems** (labelled references, not evidence of discovery).
- The hard-coded oracle and r1's P2 are 1.0 at every length. P2 is an MLP trained on the 32 transitions and iterated with argmax feedback.
- Each tied/inter network's own one-step map, re-encoded through a discrete symbol between steps, gets all 32 transitions right and is therefore 1.0 at every length. tied/final's own map gets 2–4 of 32 right.
- So the tied/inter models know every single step. What declines with length is the iteration of their continuous state.

### Part B: replication on four new tasks

Figure: [figures/replication.png](figures/replication.png) (summary `figures/replication.csv`).

**Tasks** (`tasks.json`, drawn by the rule committed before drawing). Seeds 63000 and 63001 were accepted. Split seeds 63100–63103 were all accepted, and none was rejected. Each task, checked separately:
- coverage holds: 32 transitions at every position, every op 12× per position;
- no train/held-out row overlap;
- no held-out program is equivalent to any trained program of length ≤ 3 (all 84 products are distinct);
- the 48 training final tables determine the four permutations uniquely (the reviewer's r1 enumeration, re-run per split).

The tasks differ in ways the original did not predict:

| task | ¬N rows (of 128) | 3-of-4-slot neighbour vote | best partial-slot lookup | held-out repetition pattern |
|---|---|---|---|---|
| ops51000-ls51100 (original) | 42 | 0.077 | 0.125 | 6 distinct, 3/3/3, 1 all-same |
| ops63000-ls63100 | 33 | 0.115 | 0.125 | 8 distinct, 2/2/2, 2 all-same |
| ops63000-ls63101 | 32 | 0.100 | 0.133 | 6 distinct, 3/3/3, 1 all-same |
| ops63001-ls63102 | 24 | 0.412 | 0.217 | 4 distinct, 4/4/4, 0 all-same |
| ops63001-ls63103 | 24 | 0.412 | 0.217 | 6 distinct, 3/3/3, 1 all-same |

In draw 63001 all four ops send symbol 2 to symbol 1. The draw rule allows this, and it lets rows passing through symbol 2 be answered from neighbours, which raises the neighbour vote to 0.41.

**All 16 runs fit** their training set (final accuracy 1.0 by 350–600 updates) and none was capped or failed.

| held-out final at 10k (¬N) | init 63200 inter | init 63200 final | init 63201 inter | init 63201 final |
|---|---|---|---|---|
| ops63000-ls63100 | 1.000 (1.00) | 0.484 (0.00) | 1.000 (1.00) | 0.500 (0.00) |
| ops63000-ls63101 | 1.000 (1.00) | 0.422 (0.00) | 1.000 (1.00) | 0.453 (0.06) |
| ops63001-ls63102 | 1.000 (1.00) | 0.547 (0.00) | 0.977 (0.88) | 0.555 (0.00) |
| ops63001-ls63103 | 1.000 (1.00) | 0.547 (0.00) | 0.992 (0.96) | 0.492 (0.00) |

- tied/inter > tied/final in all 8 (split, init) pairs. tied/inter reaches 0.98–1.00 held-out, with 0.88–1.00 on ¬N rows (no one-op-slot training neighbour answers them). tied/final answers ¬N rows at most 2 of 32.
- tied/final's held-out accuracy is higher on the new tasks (0.42–0.56) than on the original (0.27–0.32). It all comes from N rows.
- The timing repeats: tied/inter's ¬N accuracy is 0.00–0.08 at 2k updates and first reaches 0.2 at 3.4k–5.1k, well after the training set is fit (as on the original task, where it is 0.02–0.05 at 2k and reaches 0.2 at 3.5k–5.5k).
- Length curves at 10k (frozen, bottom row of the figure): tied/inter is 0.93–1.00 at L = 4 and 0.58–0.88 at L = 6. tied/final is 0.11–0.17 at every L ≥ 4.
- **Task versus init.** Within a draw, the same init gives similar length curves on both splits. Init 63201 is better at L = 6 on draw 63000 (0.85–0.88 vs 0.62–0.65); the two inits are about equal on draw 63001 (0.58–0.62). Three seeds on the original span 0.56–0.86. With 2 inits per split, this shows that both task and init move the L = 6 number; it does not apportion the variance.

### Follow-up: where the first errors fall (10k models)

Figure: [figures/followup.png](figures/followup.png) (`figures/followup.csv`; per-transition table `runs/c2-gen-followup-anatomy-a1/anatomy.csv`). This describes inputs, meaning which transitions fail first. It is not a mechanism.

- **Not concentrated on a few transitions (F1 failed in 11 of 11).** The 4 most frequent of the 32 (state, op) transitions carry 0.20–0.29 of first errors at steps 4–6. That is above the diffuse 0.125 but far below the predicted ≥ 0.5. The top 8 carry 0.36–0.50.
- **The same transitions fail at steps 5 and 6 (F2 held in 11 of 11).** Spearman ρ between per-transition first-error rates at steps 5 and 6 is 0.62–0.96. Between steps 4 and 5, where step 4 has enough errors, it is 0.08–0.80.
- **Held-out windows are over-represented at the error step, less than predicted (F3: 3 of 11 met the 0.375 threshold).** In 0.31–0.41 of first errors the last three ops form a held-out program, against 0.25 among rows at risk, in every model.
- **Wrong answers lean toward another op applied to the right state.** 0.62–0.71 of first wrong answers equal `P[k'](x)` for the true state x and some other op k′. Uniform wrong answers would give at most 3/7 = 0.43. That is an upper bound, because two other ops can send x to the same symbol; averaged over (state, op) on the original draw the baseline is 0.339 (reviewer G5, PR #59), so the lean is if anything understated.

## Reading

**Length.**
- Under r1's recipe at 10k updates, the shared-transition, shared-readout model trained with intermediate labels keeps executing programs one to three ops longer than any it was trained on. Its accuracy declines with each added op; no length shows a cliff.
- Continued training on length-3 programs alone removed that decline on the original task: at 50k updates all three seeds are exact through length 6. Length generalization kept improving after held-out length-3 accuracy had reached 0.96–1.00, as held-out length-3 accuracy itself kept improving after the training set was fit. This is tested only on the original task and only through length 6; the new-task runs were not continued.
- At 10k the decline is broad. First errors spread over many transitions (top four of 32 carry 20–29%), the same transitions keep failing as length grows, and the per-step error rate rises with depth. This matches gradual degradation of the iterated continuous state more than a few broken transitions. It does not identify the cause. The drift measures are descriptive, and the mechanisms lane owns whether the state is reusable under intervention.
- At 10k, "a reusable step function" overstates the result (reviewer G1). What the 10k data support is narrower: the model extends past depth 3, with lower accuracy on programs that contain a held-out 3-op window. The whole decline is not explained by those windows, though: programs without one also fall to 0.64–0.93 at L = 6. At 50k the window effect has no room to appear (every row is correct through L = 6).
- Every 10k tied/inter model maps each (symbol, op) correctly when its state is re-encoded between steps, and P2 iterates perfectly. Those references show that the 10k length decline is a property of the continuous iteration these networks had learned by then, not of the one-step knowledge. They are references, not evidence of what the network discovered.
- tied/final shows no transfer beyond length 3 under this recipe, at 10k or at 50k, even as its length-3 held-out accuracy rises. Its readout was trained on h3 only, and the states it reaches beyond step 3 are further from the trained region. This is a finite-budget observation, not a demonstration that final-only training cannot produce it.

**Task robustness.**
- tied/inter's length-3 success, including the ¬N rows a one-slot neighbour copy cannot answer, reproduced on both new draws and all four splits. The contrast with tied/final held in every pair. So the original result survives task-draw and split changes, not only initialization changes, for these 2 draws and 4 splits.
- tied/final is better on the new tasks than on the original. All of that gain is on neighbour-answerable rows, and most of it on draw 63001, where a shared fixed transition makes neighbour answers more often right. That is a property of the draw, not a failure of the replication. It also shows that a held-out accuracy of about 0.5 can arise with no ¬N success at all.

**Relation to the literature.** Liška, Kruszewski & Baroni (2018, arXiv:1802.06467) set up a related lookup-table composition domain: 8 random bijections of 3-bit strings and their 64 pairwise compositions, learned by a character-level LSTM. Their main test was zero-shot generalization to withheld *inputs* of trained compositions. About 2% of 50k random initializations exceeded 80%, and in an additional experiment (data not shown) unseen compositions did not generalize. Here the held-out units are whole programs, and the architecture builds the composition structure in (one transition applied per op, one shared readout). So these results are an extension under stronger structural assumptions, not a reproduction of theirs. No novelty is claimed; no targeted literature check beyond that paper was made.

## Predictions, scored

| prediction | outcome |
|---|---|
| A0 checks (length 3 exact, L1–2 equal logged step accuracies, references 1.0) | held |
| A1 tied/inter L=4 ≥ 0.9 in ≥ 2 of 3 | held (3 of 3: 0.99, 0.93, 0.98) |
| A2 tied/inter non-increasing from L=3 to 6 | held |
| A3 tied/inter L=6 ≥ 0.6 in ≥ 2 of 3 | held (0.86, 0.72; 51201 0.56) |
| A4 tied/final L=4–6 in [0.05, 0.25] | held (0.11–0.14) |
| A5 tied/inter 50k ≥ 10k at every L ≥ 4; tied/final 50k ≤ 0.3 at L ≥ 4 | held (inter 1.000 at every length; final 0.10–0.15) |
| B1 all 16 fit | held |
| B2 inter > final in 8 of 8 pairs | held |
| B3 inter ≥ 0.8 in ≥ 6 of 8; final ≤ 0.5 in 8 of 8 | inter part held (8 of 8); final part failed in 3 of 8 (0.547, 0.555, 0.547, all draw 63001) |
| B4 inter ¬N ≥ 0.7 in ≥ 6 of 8; final ¬N ≤ 0.1 in 8 of 8 | held (8 of 8; 8 of 8) |
| B5 new-task length curves follow the original pattern | held |
| F1 top-4 transitions ≥ 50% of first errors in every model | failed (0 of 11; 0.20–0.29) |
| F2 step-5 vs step-6 ρ > 0.5 in ≥ 9 of 11 | held (11 of 11) |
| F3 held-out-window share ≥ 0.375 in ≥ 9 of 11 | failed (3 of 11; 0.31–0.41 vs 0.25 base) |

## Limitations

- **One recipe, two budgets.** Part B and the follow-up are 10k-update numbers under r1's recipe; only the original task's six runs were also evaluated at 50k, where tied/inter's length decline disappeared. A decline at one budget does not show that longer programs are out of reach, and a plateau would not show necessity.
- **Small task sample.** There are 2 new draws and 4 splits, with 2 inits per split, and the 3 original seeds are on one task. This is exploratory coverage, not statistical power. Draw 63001 has an unusual shared transition.
- **Lengths 4–6 only.** At 10k, errors were still rising at L = 6; at 50k there were none through L = 6. No claim is made beyond L = 6.
- **Function classes.** Almost every longer program on the original draw computes a new function, so "new syntax, seen function" is untested here (10 programs). The new draws have more such programs at L = 6 (up to 209); they were counted, not analysed.
- **Drift measures** are medians of distances to training-h3 centroids. They describe where states go; they do not show why errors occur.
- **The untied-readout arm** was not evaluated at longer lengths. It is not this lane's assignment, and it needs a declared step map.

## Reviewer handoff

1. `runs/c2-gen-frozen-rep-a1/checks.json`: the 16 replication checkpoints' length-3 prediction strings and accuracies against their own logs, and the causality identity.
2. `tasks.json` against QUESTION.md's draw rule: the order of seeds tried, the accept reasons, and the per-split identifiability count (`identifiable_from_final_tables` = 1 for all four).
3. Part B registry and plan: plan entries committed (3bc4f20) before launch (04:12Z), with explicit recipe flags (C1). Every run is complete; the calibration checkpoint is listed by hash and not committed.
4. The follow-up's at-risk definition and window indexing in `gen_followup.py` (`held_window`, `first`). The F1–F3 verdicts are in `runs/c2-gen-followup-anatomy-a1/summary.json`.
5. The replication figure's ¬N trajectories for the original task, recomputed here from r1's logged prediction strings (`gen_figures.py trajectories`).
6. A5 inputs: the time lane's `ckpt-50000` taken from `claude/project-thread-xb3nfz` at 543ed95 by `git archive` into `runs/c2-inputs/` (outside Git). Hashes are in `runs/c2-gen-frozen-50k-a1/checks.json` and match the time lane's MANIFEST.json. Re-run from `research/composition-c2/time/runs/` once that lane is merged.
7. Wording: "extends past depth 3" at 10k, not "reusable step function" (G1); the 50k result is scoped to the original task and L ≤ 6. The draw-63001 caveat on B3.

## Next question

How far past length 6 do the 50k tied/inter models stay exact, and do the new-task runs reach the same state with continued training? Both are frozen or continuation measurements under the same recipe; neither was run here.
