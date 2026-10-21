# e2 synthesis

Read the figures in [INDEX.md](INDEX.md) first. Everything below is exploratory: one data seed, width 128 unless stated, three initialization seeds per condition, and held-out pairs that have guided exploration since e1.

## What e2 established

1. **The 30%/50% contrast is not an evaluation artifact.** The training sets nest, and on the 483 pairs that neither trains on the 50% runs answer 0.88–0.91 while the 30% runs answer 0–0.004 (independently recomputed).
2. **Along a nested ladder the change is graded, except in accuracy.** At 40% and 45% the runs sit between 30% and 50% on every margin, with no seed overlap and no regime split. Accuracy is steep (0.02 at 40%, 0.34–0.54 at 45%) because a correct answer needs the a+b label to beat the labels seen with an operand. Whether that is "more data" or "these particular added pairs" is still open.
3. **The held-out preference in failed runs is real and specific to a+b, but it does not predict success.** The restricted rank survives four mask-preserving nulls and separates the shares at 1k updates. But the failed runs reach by 50k the level the successful runs pass at 2k. A given level marks partial structure, not an eventual answer.
4. **Per-unit frequency structure is real but is not shown to be the computation.** Failed and successful runs have aligned per-unit a+b−c terms of the same kind. Frequency groups are not privileged for function. A label-free per-unit filter lifts the failed runs to 0.43–0.58. It also lifts runs trained **without weight decay** to 0.33–0.47, which show no held-out preference at all, and a network with untrained inputs and a readout fitted to a+b to 0.14–0.30. The filter mostly measures what fitting a+b leaves behind, not a computation the failed runs built.

## Strongest contrary evidence

The no-decay counterexample (INDEX §3; D10). It moved the round's most striking positive result, "failed runs hide a generalizing computation", to a fit-induced reading: any network that fits a+b on these pairs leaves phase relations that a one-frequency-per-unit projection turns into partial answers. The mechanisms lane reproduced it with its own nulls and revised its interpretation; the audit reproduced it independently and added a stronger version (untrained inputs plus a fitted readout reach 0.14–0.30 under the filter). Second strongest: the restricted rank in failed runs reaches values the successful runs pass early, so it is not an early signature of success.

## Open questions (at most three)

1. **Amount or identity of training pairs?** Does the 30%→50% change depend on how many pairs per class are added or on which ones? Test: a second data seed (or a different ordering of added groups) at the same nested shares, scored on a fixed common set.
2. **Is the slower growth of the operand-label gap part of finding the answer, or dilution?** Test: shuffled-label runs at the same nested shares (dynamics lane's proposal). If their operand-label preference also stops growing with share, the gap is dilution.
3. **Is per-unit frequency structure used by training, or only readable after it?** Test: a training-time intervention in a 30% run (penalize each unit's non-dominant components, with and without weight decay), scored by the unedited model (mechanisms lane's proposal). The audit's cheaper variant is scaling only the groups' aligned component in a 30% checkpoint against a sham at equal logit change.

Not addressed in e2: codec size and norm comparisons (the e1 rescaling caveat stands); widths other than 128 at shares above 30%; any claim of composition or reusable computation.
