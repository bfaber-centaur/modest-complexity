# e2 dynamics: question log

Session `e2-dy-cloud-1` (thread "Width and training-time map"), base `48ea420` on `swarm/exploration` (contains the e2 base `6aaa3a2`; shared code `fe284a5` unchanged). Newest entries last; earlier entries are not edited.

## Q1 (2026-10-21, written before computing anything below)

### Already seen when this was written

This session produced the e1 size/time map and the 30%-vs-50% comparison (PR #11), so the following were seen before the question was written: held-out accuracy and cross-entropy for every 30% run and the three 50% width-128 runs; the share of held-out argmax predictions inside the operand-associated set L(a,b) and the share of wrong predictions inside it, at published checkpoints; the mean softmax mass on L and on the correct label at those checkpoints; the common-set accuracy and cross-entropy of the six width-128 runs (coordinator INDEX, e2); and the reported restricted-rank numbers (50% runs at 0.06-0.12 by step 5,000; 30% width-128 runs at 0.27-0.31 at 50,000; width 512 at 30% plateaus near 0.18).

Not yet seen: any per-pair margin of the correct label against the strongest wrong label inside versus outside L, on any run; how those margins move between checkpoints; anything computed with a label partition shared by both shares.

### Question

On the 483 pairs that neither share trained on, what separates a run that ranks the correct label above labels unrelated to its operands from a run that actually answers correctly? Concretely: before the 50% runs start answering held-out pairs, do they already beat the 30% runs on the unrelated labels, or do the two shares differ only against the operand-associated labels?

### Label partition used for both shares

The 30% training set is a subset of the 50% one (D6), so for each common pair L30(a,b) is a subset of L50(a,b). The 30 wrong labels of a pair split into three sets that are the same for every run:

- S = L30: labels seen with operand a or b in the 30% training set (so also at 50%);
- D = L50 minus L30: labels seen with a or b only through the 179 pairs the 50% share adds;
- N = the rest: labels never seen with a or b in either training set.

Each run's own operand-associated set stays as the charter asks (S for a 30% run, S ∪ D for a 50% run). Margins against S and against N compare the two shares on identical label sets; D is where their own masks differ. For addition the correct label is in none of the three (checked in code, not assumed).

### Two explanations

- E1, shared relational ordering, different shortcut competition: both shares learn early to score the correct label above N; the shares differ in whether the correct label then overtakes S (and D). Held-out success at 50% is the in-set margin crossing zero on pairs whose N-margin was already positive.
- E2, the ordering differs from the start: the 50% runs score the correct label above N sooner and more often than the 30% runs, already before 50% accuracy rises; the correct label overtakes N and S at about the same time pair by pair, so "beats unrelated labels but loses to operand labels" is not a stage that precedes correct answers.

### Cheap observation

Published checkpoints only, own forward pass. For every common pair and checkpoint, the margin of the correct label over the strongest label in S, in D and in N (logit differences, not raw logit shifts), the fraction of pairs with each margin positive, and accuracy and cross-entropy. Set sizes |S|, |D|, |N| per pair. A 30% width comparison on the e1 audit split (no common-set issue there, one mask per run) as the second panel group.

Expected under E1: N-margin positive for a similar fraction of pairs in both shares at steps 1,000-5,000; S-margin negative in both then; at 50% the S-margin turns positive near the accuracy onset (3,900-5,500 by seed) on pairs whose N-margin was already positive one checkpoint earlier; at 30% the S-margin stays negative.

Expected under E2: the N-margin fraction is clearly higher at 50% than at 30% before the onset; pairs that become correct at 50% did not have a positive N-margin at the preceding checkpoint more often than other pairs.

### Failure modes of the diagnostic

- A maximum over more labels is larger, so a margin against a bigger set is lower by construction. Comparing shares on the same S and N removes this between shares, but not between S and N within a run (sizes differ). Fractions are reported next to the no-preference value for each pair's set size.
- Published checkpoints are sparse: 1,000, 2,000, 5,000, 10,000 for most runs, and the 30% run `e1-st-add-w128-i30002-a1` publishes only 300 and 50,000. The 50% onsets fall between 2,000 and 5,000, so "one checkpoint earlier" may be 3,000 updates earlier. This session still holds that run's and `e1-st-add-w128-i30001-a1`'s every-1,000 checkpoints locally; any of them used here gets published with its hash checked against the existing e1 manifest.
- Logit scale grows with the parameter norm, so margin sizes are not comparable across steps; signs and fractions are.
