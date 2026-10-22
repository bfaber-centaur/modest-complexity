# e3 mechanisms: reading of continuation packet 1

Written 2026-10-22 after the nine runs and the figures, and revised the same day after audit investigation 3 (`d27530a`), which recomputed packet 1. The observations are in [README.md](README.md). This file is interpretation and is provisional. The follow-up 1 section has not been audited.

## Which outcome of QUESTION.md

None of the listed outcomes fits alone: **transient strengthening, then erasure of the accuracy gain, but not of the ranks**.

- **Strengthening first.** For about 70–100 updates training adds 0.25–0.30 of held-out accuracy on top of what the edit gave (common pairs 0.43–0.58 at update 0 → 0.74–0.86 at update 100). That happens while train accuracy recovers from the edit's damage. This rise is training's doing, separate from the update-0 jump the edit produced.
- **Then erasure, in accuracy.** By about 1000 updates, under the same objective, the accuracy gain is mostly gone. By 5000 updates the branch is at chance on held-out pairs. The ranks keep a trace: at 20,000, B's correct label still outranks 39–45% of the operand-associated labels, against 14–18% for A and C, and this has not changed since 5,000.
- **Not sham-equivalent.** The rise appears only in B; C never leaves chance. By update 20,000 the two edited branches agree on accuracy, both at chance. They still differ in rank measures and in frequency structure. Neither is at an end state there: both are still fitting, with train cross-entropy about 10–15× A's.
- **Seed dependence.** None across the three checkpoints.

My stated expectation was partial erasure, keeping perhaps a few tenths. That was wrong in two ways: training first increased the gain, and then erased all of it. My expectation for C, slower train recovery and chance held-out accuracy, held.

## What the trajectory suggests (not tested here)

- **The objective does not hold the filtered network's held-out answers as argmax.** With 299 pairs and weight decay 1, training keeps going after train accuracy reaches 1. Train cross-entropy falls by four orders of magnitude, and held-out accuracy falls with it. In QUESTION.md's terms this is erasure of the answers: the edit supplied them, and training removes them as winners. It does not remove the whole trace. a+b keeps a rank above most unrelated labels and above 39–45% of operand labels, but loses the argmax to the strongest operand label.
- **Two drivers are confounded.** Erasure takes about 500–1100 updates. That matches the decay's time constant, 1/(lr·wd) = 1000 updates, and it also matches the phase where cross-entropy drives up the train margins. This packet cannot separate the two.
- **Concentration is not sufficient for held-out accuracy.**
  - At 20,000 updates (still fitting), 90–92% of B's units still have their update-0 dominant frequency, and input concentration is 0.59–0.61, against 0.40–0.45 for A. Yet held-out accuracy is at chance.
  - Whatever the erasure removes is not the per-unit frequency choice itself. It may be the phases, the alignment across units or the readout; none of these was measured.
  - B also keeps a residual rank preference. The larger part is over operand-associated labels: P(a+b over S) is 0.39–0.45 against 0.14–0.18. The smaller part is over unrelated labels: P(a+b over N) is 0.87 against about 0.7.
- **Erasure is not a return to the source network.** B's argmax agreement with A ends at 0.42–0.50, the level at which independent A runs agree (0.49; audit investigation 3), and its weight distance from A grows while it is erased. B ends as a different fitted network that again prefers operand labels (P(S over N) 0.92–0.93). "The memorizer" should not name a particular network: the training set, more than the initialization, decides which wrong label wins.
- **The early rise is not a recovered algorithm.** It shows that a short stretch of training from the filtered point increases held-out accuracy. It does not show that a hidden computation was exposed:
  - e2 found that random features with a fitted readout also filter to partial answers;
  - here the gain does not persist under the objective that produced the source.
- **Two alternatives for the rise remain open** (audit investigation 3):
  - Fitting the 299 pairs through frequency-concentrated features may itself extend to held-out pairs, as fitted readouts on filtered random features did in e2. On that reading the source network contributes nothing beyond the concentrated features.
  - The rise may be tied to the restart's step size: fresh Adam's first steps are about ±lr per coordinate. Neither packet separates this; a replay at a ten-times smaller lr, or with plain gradient descent, would. It was not run in this lane.
- **The peak is an observation, not a procedure.** Stopping at the held-out peak would select on held-out data. A label-free rule, stopping at the first update with train accuracy 1, lands near the peak. The audit checked this post hoc at one-update resolution: at updates 69, 45 and 33, common accuracy is 0.735, 0.778 and 0.839, within 0.008–0.033 of the peaks. That is three runs, checked after the fact; it is not a tested procedure.

## Proposed adaptive follow-up (not claimed, not run)

**What erases the gain: decay or the cross-entropy drive?**

- **Runs.** Continue the same B and C start files with weight decay 0, keeping everything else identical: fresh Adam, 20,000 updates, the same cadence. That is 6 runs, about 2 minutes elapsed with two workers, within the 10-minute follow-up allowance.
- **If decay drives the erasure,** B at wd 0 should keep most of its peak held-out accuracy far beyond 1000 updates.
- **If the cross-entropy drive is enough,** B at wd 0 should still lose the gain, perhaps on a different time course, while its train cross-entropy keeps falling.
- **C at wd 0** checks that any retention is specific to the dominant-frequency edit.
- **Comparison.** These are the same start files and pairs, so each wd-0 branch is compared with its wd-1 branch seed by seed.

Optimizer-state retention, which keeps the source's Adam moments, stays a separate possible follow-up. A shows that the reset alone does not move the converged memorizer, so it ranks lower.

## Follow-up 1 (weight decay 0): reading

**Answer to the follow-up question.** Within 20,000 updates, decay is necessary for the erasure seen in packet 1. The cross-entropy drive alone erodes B's held-out accuracy only slowly.

**Why decay matters, not harder fitting.**
- Without decay, the same start files kept 0.56–0.62 on the common pairs at update 20,000, against 0.01–0.03 with decay. This held even though they fit the training pairs about 1000× harder by cross-entropy.
- So in packet 1 the gain was not lost because training kept fitting the 299 pairs.
- One reading is that decay shrinks the weights the filter left. With wd 1 they keep about e^-5 of their size by update 5,000, and gradient-built weights replace them. At wd 0 the filtered structure stays: input concentration 0.85–0.87, every unit on its update-0 frequency, P(a+b over S) about 0.9. Training adds to that structure instead of replacing it.
- This reading was not tested directly, for example by measuring how much of each run's weights still lies along the start weights.

**The cross-entropy drive still contributes.**
- B at wd 0 loses 0.20–0.25 from its peak over 20,000 updates, and the loss is still continuing.
- "Both contribute, with decay dominating on this time scale" is the more careful summary. My stated expectation (QUESTION.md) matched: the same early rise, clearly more retention, and a slow drift down.

**Specific to the dominant-frequency edit.**
- C at wd 0 stays at chance and loses even the rank preference it had, so the retention belongs to B's edit.
- The wd-0 runs keep the same optimizer reset, so they say nothing about whether the early rise depends on fresh Adam's step size.
- The no-decay comparison does not change packet 1's caution. Retention under one objective is not evidence of a hidden algorithm in the failed runs. e2 showed the filter gives partial answers even on random features with a fitted readout.

**An apparent tension, unresolved.**
- e2 found that runs trained from scratch without decay also fail at 30%.
- Here, training without decay preserves the filtered network's answers.
- Together these say the objective at wd 0 does not find this solution from initialization, but it does not remove the solution once the edit puts the network there. With decay (packet 1), the objective does remove it. Neither result says why the original runs failed.

**What remains open.** Not proposed, since the lane's follow-up is used:
- whether B at wd 0 keeps declining past 20,000 updates;
- whether an intermediate decay gives an intermediate rate;
- whether optimizer-state retention changes anything.
