# c2 generalization lane: question, predictions, measures

Role prompt: [../coordinator/charter/04-generalization.txt](../coordinator/charter/04-generalization.txt). Assignment, seeds (63000–63999), run-ID prefix (`c2-gen-`), budget (25 min) and gates: [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md) D4, D6, D7. Written and committed before any model was evaluated or trained by this lane. The only numbers below that were computed first are task facts (no model involved), from `gen_task.py --task-id ops51000-ls51100`; they are marked *task fact*.

## Question

r1's shared-transition, shared-readout RNN with intermediate supervision (tied/inter) answered 96–100% of 16 held-out length-3 programs at 10k updates, on one permutation draw and one split. Is that success

- **A. reusable iteration:** does the frozen network still execute programs longer than any it was trained on (lengths 4–6)?
- **B. robust to the task:** does the tied/inter versus tied/final contrast reproduce on new permutation draws and new held-out splits, not only on new initializations?

A failure on longer programs does not erase successful length-3 generalization; it limits the claim of reusable iteration. Success or failure here is bounded by the tested draws, splits, recipe (r1's: width 128, full-batch Adam lr 1e-3, decoupled wd 0.1) and 10k-update budget.

## Part A: frozen evaluation at lengths 1–6 (no training)

**Models.** The recovered r1 checkpoints, verified against `recovered/RECOVERY.json` by SHA-256 before use:
- `r1-comp-rnn-inter-s5120{0,1,2}-a1/ckpt-10000.npz` (tied/tied, step weights 1 1 1);
- `r1-comp-rnn-final-s5120{0,1,2}-a1/ckpt-10000.npz` (tied/tied, step weights 0 0 1);
- later, the time lane's `ckpt-50000` of the same six runs, once committed (same script, same measures).

Both groups are tied, so no step map is needed; the same `A, B, bh, V, bv` are applied at every step (`c2rnn.forward`). The untied-readout arm is not part of this lane's assignment and is not evaluated.

**Rows and weighting.** Exhaustive at every length: all 4^L programs × all 8 start symbols (L = 6: 32,768 rows), labels from `c2task.rows`. Every row has weight 1. Because every program appears with all 8 starts, row weighting equals program weighting and equals input (start-symbol) weighting. A secondary per-function weighting averages first over programs sharing a product, then over distinct products; it equals row weighting at L ≤ 4 (no collisions, below) and is reported at L = 5, 6. No sampling.

**Causality identity (a check).** The RNN is causal, so the readout at step t of any length-6 row depends only on its first t ops. Accuracy at length L over all length-L programs therefore equals step-L accuracy over all length-6 programs (each length-L program occurs 4^(6−L) times as a prefix). The script computes both and asserts equality.

**What each length can and cannot show.**

| length | exposure in training | what accuracy there measures |
|---|---|---|
| 1, 2 | inter: every (s, a) and (s, a, b) prefix is a training-row prefix, labelled through the shared readout. final: never labelled. | inter: training accuracy of step1/step2, by construction (C1.1). Not compositional transfer. final: the untrained readout at h1, h2 (r1: 0.06–0.15). |
| 3 | 48 training programs labelled at h3; 16 held out | Reproduction of r1, broken down into train / held-out and, within held-out, N / ¬N (¬N = 42 rows no one-op-slot training neighbour answers). |
| 4–6 | never seen: the transition was never applied to h3, the readout never read h4–h6 | New syntax. Transfer of the iterated map beyond the trained depth, with cross-position hidden states that may be out of the training distribution (measured, below, as part of the question). |

**Function classes at longer lengths** (*task fact*, original draw; `gen_task.py`). The four ops generate all of S8 (order 40,320). All 84 products of lengths 1–3 are distinct.

| length | programs | distinct products | programs in collision groups | product equals a length-1/2 function | … a training length-3 function | … a held-out length-3 function | novel function |
|---|---|---|---|---|---|---|---|
| 4 | 256 | 256 | 0 | 0 | 0 | 0 | 256 |
| 5 | 1,024 | 1,015 | 18 | 0 | 0 | 0 | 1,024 |
| 6 | 4,096 | 3,907 | 372 | 8 | 1 | 1 | 4,086 |

So on this draw, lengths 4–6 are almost entirely new syntax *and* new input-output functions. "New syntax, seen function" is only 10 programs at length 6; their accuracy is reported but is too small a set to support a comparison. Collision-freedom at length 3 does not carry over: equivalent programs appear from length 5.

**Measures** (per checkpoint, written to CSV):
1. Final accuracy at each L = 1..6, all rows; at L = 3 also training programs, held-out programs, held-out N, held-out ¬N, and the 128-character held-out prediction string compared with r1's published `held_final_pred` at step 10000.
2. At L = 4..6, accuracy by the class of the length-3 prefix (a training or a held-out program) and by function class (table above), with counts.
3. Per-step errors on the length-6 evaluation: accuracy at each step t = 1..6 (equals measure 1 by the identity); the distribution of the first wrong step; and, at the step after a first error, whether the prediction is the true symbol (recovery), the op applied to the model's own wrong prediction (`P[op](ŷ_t)`, "follows its own error"), or something else.
4. Cross-position drift (descriptive): for each step t on length-6 rows, the median ‖h_t‖ relative to the median ‖h_3‖ of training rows, and the median distance from h_t to the training-row h_3 centroid of its true symbol, relative to the median such distance among training h_3. Values well above 1 mean the state has left the region the readout was trained on.

**Reference systems** (labelled as references, not evidence about what the network discovered):
- *Hard-coded oracle*: the true permutations applied L times by a plain per-row loop over `c2task.r1.execute`, which is independent of the vectorised `c2task.rows`. It is 1.0 by definition; agreement on every row is a label check.
- *Learned one-step transition applied repeatedly*: r1's positive control P2 (`r1-comp-interp-s51200-a1/ckpt-10000.npz`, an MLP trained on the 32 transitions, SHA-256 `bedf8ed3…7defd` checked against r1's published manifest), applied L times with argmax feedback, as r1's `train.py` does for 3 steps. The coordinator committed it to `recovered/` for this purpose (8fca55f, listed in RECOVERY.json); it is read from there. (Amended after e5a4415, before any evaluation: the first version read it from the project mount.)
- *The network's own one-step map, re-encoded*: per checkpoint, the readout of one RNN step from the start embedding, `argmax V·T(start(s), op)`, iterated with a discrete symbol between steps. It separates "knows each step" from "iterates its continuous state". For tied/inter it is 1.0 at every length by construction when step1 training accuracy is 1.0; for tied/final it reflects the untrained h1 readout.

**Predictions** (each stated before evaluation):
- **A0 (checks).** L = 3 reproduces r1 exactly: held-out 1.000 / 0.961 / 1.000 (inter) and 0.273 / 0.312 / 0.320 (final), identical prediction strings, training 1.0. L = 1, 2 equal r1's logged train step1/step2 accuracies (inter 1.0; final 0.125 / 0.094 / 0.062 and 0.148 / 0.148 / 0.086). Oracle and P2 are 1.0 at every length; the inter re-encoded map is 1.0.
- **A1.** tied/inter, L = 4: ≥ 0.9 in at least 2 of 3 seeds.
- **A2.** tied/inter accuracy does not increase from L = 3 to L = 6 in any seed.
- **A3.** tied/inter, L = 6: ≥ 0.6 in at least 2 of 3 seeds. If L = 4 is below 0.5 instead, h3 is not a usable transition input and reusable iteration beyond the trained depth fails under this recipe.
- **A4.** tied/final, L = 4–6: between 0.05 and 0.25 in every seed. Its readout is trained only on h3, and r1 found no shared code at h1, h2, so h4–h6 should not be readable.
- **A5 (50k, when available).** tied/inter at 50k is at least its 10k accuracy at every L ≥ 4. tied/final at 50k stays ≤ 0.3 at L ≥ 4 even if its L = 3 held-out accuracy has risen.
- Measures 3–4 are descriptive; no prediction.

## Part B: replication on new tasks (new training; waits for the reviewer's pin PASS)

**Design.** 2 new permutation draws × 2 held-out splits per draw × 2 init seeds per split × {tied/inter (1 1 1), tied/final (0 0 1)} = 16 runs, 10k updates, `c2train.py` unchanged (r1 recipe). The same two init seeds, 63200 and 63201, are used on every split, so initialization is crossed with task: within a split, inter and final start from the identical function; across splits, the same initial weights meet different tasks. This keeps task-draw, split and init variation distinguishable. These are new tasks, recorded separately from the original (`ops51000-ls51100`).

**Draw and split rule, fixed now, before any new draw is made** (`gen_task.py --draw`):
- Draw seeds are tried in order 63000, 63001, … Seed s is accepted iff (i) r1's `draw_ops(s)` accepts s itself (no identity op, no two ops equal; a seed r1 would skip is rejected, not renumbered), and (ii) all 84 products of programs of lengths 1–3 are pairwise distinct. (ii) excludes commuting pairs, involution shortcuts and any equivalence between a held-out program and a trained program of length ≤ 3, for every split. The first two accepted seeds are the draws.
- Split seeds are tried in order 63100, 63101, … across both draws (the first draw takes the first two accepted, the second draw the next two). A split is r1's `split(seed)`, a uniformly drawn 4×4 Latin square. It is rejected if its square equals an already accepted square of the same draw, or if some (state, op) transition is missing at some position (impossible for a Latin-square split; checked anyway).
- Every tried seed, accepted or rejected, is logged with its reason in `tasks.json`.

**Reported per task, not used for acceptance:** group order and op parities, cycle types, commuting pairs, involutions; coverage counts; r1's baselines; the ¬N count and whether every one-slot neighbour is a training row; whether the 48 training final tables determine the four permutations uniquely (reviewer's r1 enumeration, re-run per split; not assumed); held-out repetition pattern; length-4–6 collision and function-class counts.

**Run IDs and commands.** `c2-gen-{inter,final}-ops{draw}-ls{split}-s{init}-a1`, each `c2train.py --task-id ops{draw}-ls{split} --step-weights … --init-seed … --updates 10000 --checkpoint-steps 0 --diag`, through `run_jobs.py --workers 2`, plan entries committed before launch. One calibration run first (`c2-gen-calib-inter-s900130-a1`, 2,000 updates) to measure ms/update on this host.

**Measures.** As r1 and c2train log them: training and held-out final accuracy every 50 updates; held-out N / ¬N; training fit step. At 10k, per task: held-out final, ¬N, train. Then Part A's frozen length 1–6 evaluation applied to each run's `ckpt-10000` on its own task (measures 1–3; no reference systems, since P2 exists only for the original task). Report every run and seed; no pooling of task draws with inits.

**Predictions.**
- **B1.** Every run fits its training set (final accuracy 1.0) by 10k.
- **B2.** tied/inter > tied/final in held-out final accuracy at 10k in all 8 (split, init) pairs.
- **B3.** tied/inter held-out ≥ 0.8 in at least 6 of 8 runs; tied/final ≤ 0.5 in all 8.
- **B4.** tied/inter ¬N accuracy ≥ 0.7 in at least 6 of 8 runs; tied/final ¬N ≤ 0.1 in all 8.
- **B5.** The length curves of the new-task tied/inter runs follow whichever of A1–A3 holds on the original task (same qualitative pattern at L = 4–6).

A result robust to task variation needs B2–B4 to hold on both draws and both splits of each, not on average. If a split or a draw fails while its sister succeeds, that is reported as task dependence, not averaged away.

## First figure plan

`figures/length.png` (+ `.svg`, source `length.csv`, regeneration command in README): accuracy (all rows, exhaustive) against program length 1–6. Two panels with shared axes: tied/inter and tied/final. One line per init seed at 10k (and a second line style per seed at 50k when available). Trained length 3 shaded; lengths 1–2 marked "exposed through intermediate supervision (inter only)". At L = 3, open markers for held-out-only accuracy. Reference systems drawn as labelled grey lines: oracle, P2 iterated, own one-step map re-encoded. Chance 0.125 dashed. The exported PNG is inspected before it is cited.

## Budget and order

25 min elapsed (D6). Planned: Part A, seconds; calibration, under a minute; 16 runs × 10k at the coordinator's 7.5 ms/update ≈ 75 s each, ≈ 10 min with two workers; frozen evaluation of the 16 runs, under a minute; the remainder for one follow-up. 80% (20 min) triggers a note to the coordinator.

Order: this file → Part A (frozen; allowed once this file is pushed) → M1/M2 → pin PASS relayed → calibration → plan entries committed → Part B → one follow-up, proposed in a "Follow-up" section appended here and approved by the coordinator → NOTE.md.

## Adaptive choices

Recorded separately in [ADAPTIVE.md](ADAPTIVE.md) as they happen.

## Follow-up (proposed 2026-10-22 ~04:25Z, after Part A and Part B, before any follow-up measurement; awaiting coordinator approval)

**Question.** Where does reusable iteration first fail? In all 11 tied/inter models (3 original, 8 new-task) accuracy is ≥ 0.93 at length 4 and falls to 0.56–0.88 by length 6, and the per-step error rate rises with depth. Are the first errors at steps 4–6 concentrated on a few (state, op) transitions, i.e. specific transitions that break once the state comes from a later position, or spread across transitions as diffuse accumulation?

**Design (frozen, no training, no new runs of the network beyond forward passes).** The 11 tied/inter `ckpt-10000` files already evaluated. Exhaustive length-6 rows on each model's own task. For every row whose first wrong step t* is 4, 5 or 6, record the transition at t*: the true state before step t* and the op at t*. In exhaustive rows the state before any step is uniform over the 8 symbols and the op uniform over 4 (each prefix is a permutation applied to all starts), so with diffuse errors each of the 32 transitions carries 1/32 of first errors.

**Measures.**
1. Concentration: share of a model's first errors (steps 4–6 pooled) carried by its 1, 4 and 8 most frequent transitions (diffuse expectation 1/32, 4/32, 8/32), with the full 32-entry table per model.
2. Consistency across depth: Spearman correlation, over the 32 transitions, between the first-error rate at step 5 and at step 6 (step 4 has few errors in some models; reported where it has at least 20).
3. Wrong answers at the first error: share equal to another op applied to the true state, `P[k'](x)` with k' ≠ k (3 of the 7 wrong symbols, so 3/7 if errors were uniform over wrong symbols). Descriptive.

**Predictions.**
- **F1.** In every one of the 11 models, the 4 most frequent transitions carry ≥ 50% of first errors at steps 4–6.
- **F2.** The step-5 and step-6 first-error rates over the 32 transitions correlate positively (Spearman ρ > 0.5) in at least 9 of 11 models.
- If F1 fails broadly (top-4 share near 4/32), the decline is diffuse accumulation rather than a few broken transitions.

Run ID `c2-gen-followup-anatomy-a1` (`gen_followup.py`), elapsed cost about 1 minute. This stays a frozen evaluation; it does not test whether the hidden state is a reusable semantic state (the mechanisms lane's interchange question) and does not fit anything on longer programs.
