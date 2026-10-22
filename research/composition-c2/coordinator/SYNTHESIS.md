# Composition round c2: synthesis

**Status.** DRAFT. Waiting on the time lane's follow-up F1 (tt-final 50k → 100k) and its NOTE, and on the reviewer's last checks (listed under "Open items"). Every number below is quoted from a lane NOTE or from `review/REVIEW.md`; nothing here is a new measurement.

**Scope of every claim.**
- One recurrent architecture family: width 128, ReLU, one start embedding, a transition (A, B, bh) and a readout (V, bv), each either shared across the three steps ("tied") or one copy per step.
- r1's recipe: full-batch Adam, lr 1e-3, eps 1e-8, decoupled weight decay 0.1 on every parameter, 384 training rows (48 length-3 programs × 8 starts).
- Mostly one task draw and split (`ops51000-ls51100`, 128 held-out rows, of which 42 are ¬N: no one-op-slot training neighbour answers them), three paired init seeds (51200–51202). The generalization lane adds 2 draws × 2 splits × 2 inits.
- Budgets of 10k updates (3.84M row presentations) and 50k (19.2M), and 100k for tt-final only.
- Three seeds are exploratory coverage, not statistical power. A finite plateau or gap shows neither necessity nor impossibility. A linear probe that decodes a symbol does not show that the network uses it.

Lane notes: [time](../time/NOTE.md), [sharing](../sharing/NOTE.md), [generalization](../generalization/NOTE.md), [mechanisms](../mechanisms/NOTE.md). Review: [review/REVIEW.md](../review/REVIEW.md).

## Short answers

| question | answer under this recipe and these budgets |
|---|---|
| (a) which gaps close with time | The per-step-readout gap closes by 50k. The final-only gap does not close by 50k (F1 pending). tied/inter's length decline is gone by 20k. |
| (b) which sharing choices matter within the budget | At 10k only tied transition + tied readout composes, and the two tyings interact strongly. The effect depends on weight decay, and for the readout it is a speed difference by 50k. |
| (c) does success transfer beyond the original table | Yes, within what was tested: to lengths 4–6 on the original draw (exact through length 6 at 50k) and to 2 new draws and 4 new splits at length 3. |
| (d) what natural-state interventions support | In tied/inter, a state that acts as its symbol for later ops up to length 4, with deviations from the class mean (much of them scale) going with longer-length errors. The other arms' states are history-dependent. Whole-state interchange is input recombination here. |

## (a) Which gaps close with time

The time lane continued the nine recovered r1 runs from their exact 10k state (weights, both Adam moments, step count) to 50k, with the recipe unchanged. Joins with r1's records are exact (reviewer §2).

| arm (r1 10k held-out) | first of five evaluations ≥ 0.95 | ¬N crossing | held-out at 50k |
|---|---|---|---|
| tied/inter (0.96–1.00) | 7,700 / 9,150 / 8,200 | 8,650 / 11,200 / 9,000 | 1.000 in all three |
| per-step readout/inter (0.52–0.63) | 24,900 / 32,300 / 33,550 | 30,250 / 34,100 / 39,350 | 1.000, ¬N 1.000 in all three |
| tied/final (0.27–0.32) | censored (> 50,000) | censored | 0.56 / 0.70 / 0.66, ¬N 0.12 / 0.26 / 0.14 |

- **Per-step readouts: closes.** The 10k gap between tied and per-step readouts is a difference in time to criterion under this recipe, 3–4× longer, not an endpoint difference.
- **Final-only: not closed at 50k, still moving.** On held-out rows still wrong at 50k, the logit gap shrank 15–28% from 40k to 50k, and held-out cross-entropy kept falling (reviewer CX-T1). So "unresolved" is the right reading, not "persists". F1 extends these three runs to 100k: *pending*.
- **Length: tied/inter's decline closes.** At 10k, tied/inter falls to 0.56–0.86 at length 6. At 50k all three seeds answer every row of every length 1–6. The decline was already gone at 20k (reviewer G6). tied/final stays at 0.10–0.15 beyond length 3 at 50k.
- **Two qualifiers on "time" (reviewer T1, CX-T2).**
  - In every arm, median Adam √v̂ falls below eps at 24.1–25.1k, and every coordinate follows by 27.8–29.7k. Past that point the update behaves like momentum gradient descent at an effective rate lr/eps, balanced against decay. Two of the three per-step-readout crossings and all of its ¬N crossings fall in that regime, as does 56–70% of tied/final's 10k→50k gain, and all of F1. So "closes with time" includes a change of optimizer regime. It is not a continuation of the early Adam dynamics. tied/inter's length decline was gone before the regime began.
  - The per-step-readout catch-up partly re-ties the readouts. ‖V_k − V_3‖/‖V_3‖ fell from about 1.1 to 0.65–1.05, and V3 reads h1 at 0.81 / 0.97 / 0.84 at 50k.
- **Not continued:** the untied-transition cells (UT, UU), any wd 0 run, the new-task runs and r1's MLPs.

## (b) Which sharing choices matter within the budget

The sharing lane ran the 2×2 of transition and readout tying, all with intermediate labels (step weights 1 1 1), from one shared function at init. Every run fits training by 350–700 updates.

| held-out at 10k, wd 0.1 | all 128 rows | ¬N | parameters |
|---|---|---|---|
| TT (tied, tied) | 1.000 / 0.961 / 1.000 | 1.000 / 0.881 / 1.000 | 19,208 |
| TU (tied, per-step) | 0.633 / 0.523 / 0.516 | 0.24 / 0.07 / 0.05 | 21,272 |
| UT (per-step, tied) | 0.836 / 0.680 / 0.625 | 0.60 / 0.33 / 0.19 | 53,256 |
| UU (per-step, per-step) | 0.766 / 0.609 / 0.703 | 0.38 / 0.19 / 0.26 | 55,320 |

- **At 10k, only the combination composes.** Tying the readout helps a lot under a tied transition (+0.37 to +0.48) and little or inconsistently under per-step transitions (−0.08 to +0.07). Tying the transition helps under a tied readout (+0.16 to +0.38) and hurts under per-step readouts (TU − UU = −0.09 to −0.19 on every seed). The interaction is +0.30 to +0.56. TU, UT and UU were all still rising at 10k.
- **It depends on decay.** At wd 0, with everything else unchanged, every cell stays at 0.30–0.45 held-out and answers almost no ¬N row (≤ 0.07). The interaction shrinks to +0.06 to +0.13, and TU − UU vanishes. TT keeps a small lead on every seed (0.05–0.13). The wd 0 runs rise slowly through 10k while their logits grow, so 10k is not their plateau (reviewer CX-S2). wd 0 is a different recipe, not a control that isolates regularization.
- **With time, the readout choice becomes speed.** Under wd 0.1, TU reaches 1.000 by 50k (see (a)). Whether UT and UU also close was not tested.
- **What the comparison changes at once.** Parameter count (UT and UU have 2.8× TT's), the number of decayed parameters, summed against per-copy gradients, and how often each copy is used per row (a tied transition three times, a copy once). The sharing result is about these packages, not about tying alone.
- **Decodability is not composition.** UT's shared readout gives a state code that a probe fit at one position reads at the others (0.75–0.84 off-diagonal), well above TU and UU, yet UT composes only partly.

## (c) Does success transfer beyond the original finite table

**Longer programs, same draw (frozen evaluation).** On this draw, lengths 4–6 are almost all new input-output functions as well as new syntax: at length 6, 4,086 of 4,096 programs compute a function not seen at length ≤ 3.

| tied/inter | L = 4 | L = 5 | L = 6 |
|---|---|---|---|
| 10k | 0.93–0.99 | 0.74–0.95 | 0.56–0.86 |
| 20k (reviewer G6) | | | 0.9996 / 0.986 / 1.000 |
| 50k | 1.000 | 1.000 | 1.000 |

- At 10k, accuracy is lower on programs that contain a held-out length-3 window at any position (by 0.01–0.14, reviewer G1). First errors are spread over many transitions, not a few broken ones (the top 4 of 32 carry 0.20–0.29).
- Each 10k tied/inter model maps every (symbol, op) pair correctly when its state is re-encoded through a discrete symbol between steps. What declined with length at 10k was the iteration of the continuous state, not one-step knowledge.
- tied/final does not extend at 10k or 50k (0.10–0.15 beyond length 3).

**New draws and splits (16 new 10k runs).** 2 new permutation draws × 2 splits × 2 inits, each checked for coverage, no train/held-out overlap and identifiability.
- tied/inter reaches 0.98–1.00 held-out at length 3, with 0.88–1.00 on ¬N rows. tied/inter beats tied/final in 8 of 8 pairs. tied/final answers at most 2 of 32 ¬N rows.
- tied/final's held-out accuracy is higher on the new tasks (0.42–0.56), all of it on neighbour-answerable rows, and most on draw 63001, where all four ops send symbol 2 to symbol 1. So held-out accuracy near 0.5 can arise with no ¬N success.
- At 10k, the new-task length curves follow the original pattern (tied/inter 0.58–0.88 at L = 6). The new-task runs were not continued.

**Bounds.** Lengths up to 6 only; 2 new draws; one recipe. Liška, Kruszewski and Baroni (2018, arXiv:1802.06467) studied a related lookup-table composition domain with an LSTM and found little zero-shot composition. These models build the composition structure in (one transition per op, a shared readout), so the result here is an extension under stronger structural assumptions, not a reproduction. No novelty is claimed, and no targeted literature check beyond that paper was made.

## (d) What natural-state interventions support

All interventions are on the recovered 10k checkpoints of tied/inter, per-step-readout/inter and tied/final.

- **Whole-state interchange is input recombination here.** With a shared transition, putting a donor's h_i in place of a base row's h_j gives, bitwise, the network's output on the spliced program (donor prefix + base suffix). So natural interchange adds no evidence beyond evaluating recombined programs. Its numbers match the generalization lane's frozen length evaluation, as they must.
- **tied/inter: a reusable state, to a degree.** States from any depth and history act as their symbol for later ops up to length 4 (S ≥ 0.93), and errors at lengths 5–6 depend on the donor's history. Replacing a state by its training class mean fixes most long-length failures (S_cent ≥ 0.939 in every cell), and shrinking the residual h − c(x) improves accuracy monotonically. Matching only the donor's norm to the class mean's recovers 25–63% of the class-mean gain, against 1–22% for choosing donors nearest their own symbol's centroid (reviewer CX-M1). So much of what the class mean fixes is the state's overall scale. This fits the drift the generalization lane measured at 10k, and the 50k states sit back at their training norm (0.99–1.00×).
- **Per-step readout and tied/final: history-dependent.** Natural states are read only at the position each readout was trained on, and cross-length success is near chance. Replacing the depth-2 state by its class mean makes op 3 right for every (symbol, op) pair in per-step-readout and for 88–97% in tied/final. That is close to true by construction: every such pair occurs in training, and the class mean uses the true symbol. The bounded reading is that these networks' natural states on unseen programs are dominated by variation away from the class mean.
- **Where in the state the residual lives is unresolved.** Swapping the 7-dimensional centroid-span coordinates moves 0.26–0.59 of held-out answers to the donor's target, against ≤ 0.13 for random 7-dimensional subspaces. But those controls carry less energy, and the split states leave the ReLU state set, so these rates are not a measured share of the causal effect.
- **Not done:** interventions on the 50k states, which are exact through length 6.

## Three kinds of evidence, kept apart

- **Architectural inductive bias.** Every RNN here shares its transition across steps; r1's MLPs reached 14–25% held-out. Within the RNNs, the sharing grid shows that at 10k the tied-transition-plus-tied-readout package composes and the others do not yet, with large parameter-count and decay confounds. With time, the per-step readout catches up under wd 0.1, and the untied-transition cells were not continued. The structural assumptions that make composition easy here are also why the result extends rather than reproduces Liška et al.
- **Extra labels.** Intermediate labels (step weights 1 1 1) against final-only (0 0 1), with the same tied architecture: inter composes within 10k on the original task (crossings at 7.7–9.2k) and on all 8 new-task pairs; final-only has not by 50k on the original, transfers nothing past length 3, and answers few ¬N rows. Two cautions. Every held-out intermediate prefix occurs in training, so intermediate-head accuracy is not an unseen-prefix test. And inter gives three labelled positions per row against one, so "labels" also means more gradient signal per row.
- **Mechanistic evidence.** It comes only from the mechanisms lane's interventions on 10k checkpoints. Those support "a reusable state up to length 4 in tied/inter" and "history-dependent states in the other two arms". Whole-state interchange reduces to recombined inputs, and the class-mean interventions use the true symbol. Decoding tables (sharing, mechanisms) are descriptive throughout. Nothing here shows that any network implements an explicit symbolic state machine.

## Next questions (at most three)

1. **Does the late catch-up need the eps regime?** Per-step-readout crossings and most of tied/final's rise happen after Adam's √v̂ falls below eps. Re-running 25k → 50k under a recipe that stays out of that regime would say whether "closes with time" belongs to the optimizer phase or to training length.
2. **Are the untied-transition gaps speed or endpoint?** Continue UT and UU (and one wd 0 cell) from their exact 10k states to 50k, as the time lane did for TU. That separates which sharing differences are time-to-criterion from which persist.
3. **What do the 50k tied/inter states look like?** They are exact through length 6. How far past length 6 do they stay exact, do the new-task runs reach the same state when continued, and do the mechanisms lane's class-mean and residual measures change at 50k?

## Runs and compute

No paid compute; all runs used the session containers. An update took 3.7–7.5 ms depending on host (D6), with peak RSS about 45 MiB. Elapsed time counts wall time with at least one job running.

| lane | runs | failed or capped | elapsed / budget | PR | merged |
|---|---|---|---|---|---|
| coordinator | 7 (3 resume checks, 4 calibrations) | 0 | about 3.5 / 10 min | | |
| time | 12 + F1 (3) | 0 | 12.2 + F1 / 40 min | #55 | *pending* |
| sharing | 28 (4 calibrations, 12 at wd 0.1, 12 at wd 0) | 0 | 14.4 / 20 min | #54 | 52e4f7d |
| generalization | 22 | 1 failed (D10), kept | 6.4 / 25 min | #51 | 344f8e2 |
| mechanisms | 6 (no training) | 0 | about 1.8 / 15 min | #52 | 31f0fcc |
| reviewer | | | 3.8+ / 25 min | #50, #53, #56, #57, #58, … | |

Committed checkpoints: the 28 recovered r1 files (`recovered/`), and the time lane's 36 checkpoints at 20k–50k. Other lanes list local-only checkpoints by hash in their manifests.

## Open items before this is final

- Time: F1 results and NOTE (with T1, CX-T1, CX-T2); reviewer check of F1 and the NOTE; merge PR #55.
- Reviewer: generalization follow-up anatomy numbers; recheck of the mechanisms NOTE revision (0fe8624); sharing wd 0 decode and group tables (optional; they rest on local-only checkpoints).
