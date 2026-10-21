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

### Outcome (2026-10-21, after packet 1 in README.md)

Neither explanation fits as written. Against E1: by 2,000 updates, before any 50% run answers held-out pairs, the 50% runs beat every N label on 0.29-0.39 of common pairs and the 30% runs on 0.14-0.16, so the shares do not share the early ordering against unrelated labels. Against E2: at 5,000 and 10,000 updates 57-71% of common pairs in the 50% runs beat every label outside their own set while losing to one inside, and those pairs become correct by the next checkpoint more often than other wrong pairs. Not predicted by either: at 50% the gap between the strongest S label and the strongest N label stays at 3.7-4.8 logits throughout, while at 30% it doubles to about 8; up to 1,000 updates the two shares are indistinguishable on every measure except the D labels.

## Q2 (2026-10-21, cycle 2, written before any run below exists)

### Already seen

Packet 1 (README.md), including the 1,000-2,000-update separation and the S − N gap behavior. Nothing at a training share between 30% and 50%.

### Question

With the data seed fixed, training sets nest at every share: each class has 16 unordered pairs in a seeded order, and the training set takes the first 5 (30%), 6 (40%), 7 (45%) or 8 (50%). The 483 common pairs are untrained at all four. As the training set grows by one unordered pair per class at a time, does behavior on the common pairs change gradually, or does each run follow either the 30% pattern or the 50% pattern?

### Two explanations

- E3, graded: the rise of the correct label against N labels and the growth of the S − N gap both change smoothly with the number of training groups; 40% and 45% runs sit between the 30% and 50% runs on both measures and on common-set accuracy at 50,000, with seed spread smaller than the share effect.
- E4, two patterns: each run settles within the first few thousand updates into one of the two packet-1 patterns: margin over N flat with a growing gap (30%), or margin over N rising with the gap near 4 (50%), possibly with a later onset. Intermediate shares change which pattern a run takes, so seeds at the same share can split.

### Separating observation

Six runs, width 128, init seeds 30000-30002, at 40% and 45% (plan entries in REQUEST.md); every other setting as the e1 and e1f2 runs. Per run, on the same 483 pairs and the same S, D, N sets as packet 1 (the 40% and 45% sets lie between the 30% and 50% ones, so D splits into labels a run has and has not seen with an operand; `margins.py` gets a per-run own-set definition): margin over N, S − N gap, "beats every N", accuracy at 1,000-10,000 (every 1,000), 20,000 and 50,000 updates.

Expected under E3: at 2,000 and 10,000 updates the median margin over N and the S − N gap of the 40% and 45% runs fall between the 30% and 50% ranges, ordered by share, and accuracy at 50,000 is intermediate for every seed.

Expected under E4: each run's (margin over N, gap) at 10,000 lies near the 30% cluster (about −1.5, 5.8) or the 50% cluster (about 2.4-3.3, 3.8), and accuracy at 50,000 is near 0 or above 0.5; at least one share may split by seed.

A run whose margin over N rises while its gap still grows would fit neither and would show that the two measures can come apart at a fixed width (at 30%, width 512 already shows this pattern weakly).

### Failure modes

- Three seeds per share cannot tell a narrow graded transition from a switch; a split by seed is informative, an absence of one is weak.
- A 40% run whose correct-answer onset is later than 50,000 updates would look like the 30% pattern on accuracy; the margin and gap trajectories are what distinguish it, and runs still changing at 50,000 are reported as such.
- Amount and identity of training data still change together: the added groups are fixed by the seeded order (at 40% each class adds the group that was first in the e1 calibration split, at 45% both e1 calibration groups). A second data seed would be needed to separate them and is not part of this cycle.
