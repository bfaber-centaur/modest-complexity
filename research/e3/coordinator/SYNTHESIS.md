# e3 synthesis

Read the figures in [INDEX.md](INDEX.md) first. Everything here is exploratory and covers one width (128), one recipe (full-batch AdamW, lr 1e-3, weight decay 1 unless stated), p = 31, and two or three init seeds per condition.

## What e3 found

1. **The data-size ladder survives changing the training examples.** Three new data seeds, each a nested 30/40/45/50% ladder, give the same ordering and the same accuracy bands as the e2 ladder:

   | share | accuracy |
   |---|---|
   | 30% | 0.000–0.004 |
   | 40% | 0.01–0.09 |
   | 45% | 0.40–0.55 |
   | 50% | 0.88–0.91 |

   In this sample the location of the step varies between data seeds no more than between inits. The one ladder-specific hint, 41002 at 40% on a graded margin, did not survive the audit's pre-stated test (observation; audited).
2. **The intermediate shares are settled, not slow.** Six 45% runs continued from their exact optimizer state from 50k to 150k updates, and four 40% runs from 50k to 100k. None of them moved: accuracy changed by at most ±0.013 and losses and norms were stationary. Packet 1's "still rising" was a reading of two sparse checkpoints and is withdrawn. So, at this recipe, the 50k ladder describes where runs end up. Init changes that end point at 45%: 41001's 0.14 gap persists (observation; audit of the continuation pending).
3. **After the filter edit, training first improves the edited network, then erases its answers, and weight decay drives the erasure.**
   - **The edit.** The per-unit dominant-frequency filter takes failed 30% networks from about 0 to 0.43–0.58 held-out accuracy. The input-energy-matched sham does not.
   - **Training first adds.** Continuing on the original objective adds another 0.26–0.31 within about 100 updates, peaking just after training accuracy returns to 1. The rise survives a ten-times-smaller learning rate and plain gradient descent (audit probe), so it comes from fitting from the edited point, not from the optimizer restart.
   - **Then training removes.** With weight decay 1 the gain is gone in accuracy by 5k updates, though a rank trace remains. The result is a different fitted network that again prefers labels seen with an operand, not a return to the original.
   - **Without decay** the same start files keep 0.56–0.62 after 20k updates.

   The edit's gain is therefore not something this training objective holds at 30% (intervention; audited).
4. **The operand-label preference is not growing.** Measured scale-free as P(S over N), it is highest at first fit, flat or slowly falling after. It is the same without decay and falls by only 0.05–0.06 from 30% to 50% in every ladder. What changes with share is the correct label's standing: P(a+b over S) rises from 0.14 to 0.98 between 30% and 50% (observation; audited).

## Strongest contrary evidence

- **Against "the filter exposes a hidden algorithm":** the evidence is the rise-and-erase result together with e2's controls.
  - The edit's gain is matched by no-decay and random-feature networks (e2).
  - The objective that trained the failed network removes the gain within a few thousand updates.
  - The improvement is an intervention on the network, not evidence that the network contained the answer.
- **Against "data identity sets the step location":** the init × training-set interaction (audit investigation 5).
  - The leading init flips between ladders at 40%.
  - On the 53 pairs untrained in all four ladders, 41002 leads for neither init.
- **Against "the 45% runs only need more time":** the flat 50k–150k continuations.

## Remaining questions (at most three)

1. **What sets the settled accuracy at an intermediate share: the training set, the init, or their interaction?** Two inits per ladder cannot separate them, and the 45% outcome differs by init without moving with more training. Cheapest test: inits 30002 and 30003 at 40% and 45% on two ladders (audit's suggestion), scored on the ladder pairs and on the 53 pairs no ladder trains.
2. **Why does weight decay remove the edited network's held-out answers?** The candidate explanation is that under decay, at 30% data, the operand-label memorizer is the cheaper solution. Norms are consistent with it, but it is untested. Cheapest test, with no new runs: compare which weight components shrink between B's peak and 2k updates at wd 1 and wd 0, at equal training fit.
3. **Is the settled ladder a property of the decay strength?** Both lanes found weight decay deciding the end state: the ladder runs settle, and decay erases the edit's gain. A single 45% ladder rung at a different weight decay (or learning rate), run to settling, would show whether the step location moves with the recipe.

Not addressed in e3: widths other than 128, any composition or reuse claim, and optimizer-state retention after an edit.
