# e3 mechanisms: question log

Written 2026-10-22 at about 00:20 UTC on base `4783a47` (e2 closed, D11), before any e3 run or any held-out evaluation of the edits below. This is a prospective exploration log, not a preregistration. Session label to be assigned by the e3 coordinator. Nothing below is launched until the coordinator pins the e3 setup, assigns run IDs, and the auditor has checked the continuation code.

One computation was made before writing this: a label-free feasibility check of the sham's matching rule, on the three 50k checkpoints' weights only (energies per unit and frequency; no pairs, labels or accuracies). It is described under "Matching rule".

## What e2 already showed (and what it did not)

- The per-unit dominant-frequency filter (`research/e2/mechanisms/unitfilter.py`, `filter_units` with `dominant(w)[0]`) lifts the three failed 30% width-128 runs from 0.000–0.004 to 0.43–0.58 accuracy on the 483 common never-trained pairs, with train accuracy falling from 1.000 to 0.69–0.79.
- The same filter lifts runs trained without weight decay to 0.33–0.47, and untrained random features with a readout fitted to a+b to 0.14–0.30 (e2 audit cycle 3). Its held-out accuracy is mostly what fitting a+b leaves behind under a one-frequency-per-unit projection, not a computation the failed runs built (e2 SYNTHESIS §4).
- So the edit is an intervention that produces a different network: the input projection flips 28–35% of (pair, unit) ReLU gates (audit G17). Nothing here says what training does to that network afterwards.

## Question

After the filter improves a failed model, does continued training on the original training set and objective preserve, strengthen, or erase the improvement? And is whatever happens specific to keeping each unit's dominant frequency, or does an equally large label-free edit that keeps other frequencies behave the same way?

## Branches

For each of the three failed checkpoints at 50,000 updates, `e1-co-add-w128-i30000-a1`, `e1-st-add-w128-i30001-a1` and `e1-st-add-w128-i30002-a1` (e1 split `aa8f7504…1f24`, 299 training pairs, weight decay 1):

- **A, unchanged.** The checkpoint's U, V, W, byte-identical.
- **B, filter.** `filter_units(w, dominant(w)[0])` exactly as in e2 packet 2: every live unit's U and V columns and W row projected onto its dominant input frequency k_j, with the DC of U and V kept.
- **C, matched sham.** Every unit keeps only non-dominant frequencies (DC of U and V kept, the same frequency set for U, V and W), chosen by the label-free rule below with a recorded seed (37000, 37001, 37002 for the three checkpoints).

All nine branches then train for 20,000 updates with the original recipe: full-batch Adam on the same 299 training pairs and a+b targets, lr 0.001, decoupled weight decay 1, cross-entropy. Adam moments are zeroed and the step count restarts at 1 in **every** branch, including A, so the comparison is recovery after a common optimizer reset, not uninterrupted training. Checkpoint age (50,000) and continuation updates (0–20,000) are recorded separately.

## Matching rule (C against B)

B removes all of each unit's non-dominant energy. A sham that drops the dominant frequency cannot match B unit by unit: in about a third of units (37–46 of 128) the dominant frequency holds more input energy than all the others together. The rule therefore matches in aggregate:

1. For each unit, draw a random order of its 14 non-dominant frequencies (seeded as above).
2. For a global factor λ, each unit keeps frequencies in that order until its kept input (U and V, non-DC) energy reaches λ times its dominant-frequency energy, stopping at whichever of the last two totals is closer.
3. λ is set by bisection so that the network's total kept input energy equals B's (equivalently, the same total input displacement).

No pair, label or accuracy enters. Feasibility on the weights alone: λ = 1.8–2.1, total kept input energy within 0.5% of B's, relative input displacement 0.74–0.76 for both B and C. The median unit keeps 12–14 of its 14 non-dominant frequencies, so for most units C is close to "remove only the dominant frequency".

Known mismatches, all to be reported rather than hidden:
- Output weights: W's relative displacement is about 0.71–0.75 for C against 0.78–0.80 for B, because the rule matches input energy only.
- Unit by unit: about a third of the units cannot be matched individually, so per-unit displacement differs even though the total is matched.
- Gates: B and C will flip different numbers of ReLU gates.
- Immediate damage: C is expected to lose much more train accuracy than B.

Reported for every branch right after the edit and before any update: relative displacement per matrix, kept energy per matrix, gate flip fraction over all 961 pairs, RMS pair-centred logit change, train loss and accuracy, and common-set accuracy.

## Measurements

At the original checkpoint, right after the edit (continuation update 0), and during continuation:
- training loss and accuracy;
- accuracy on the 483 common pairs and on the run's own cal + audit;
- the normalized margins and ranks agreed with the auditor for e3 (provisionally: margin over unrelated labels in units of the wrong-label logit spread, and the restricted rank);
- per-unit frequency concentration and input/output agreement (`units.py`);
- distance from the edited starting weights, and from the original checkpoint.

Cadence (proposal for the shared code): evaluation every 10 updates for all 20,000 (cheap), and checkpoints at 0, 10, 30, 100, 300, 1,000, 2,000, 5,000, 10,000, 15,000 and 20,000. That is dense enough early to see a benefit lost within tens of updates.

## What each outcome would mean

- **A after the reset.** Branch A is the reference for everything else. If the reset alone moves A (fresh moments give near-unit-size Adam steps at first), B and C must be read against A, not against the 50k checkpoint.
- **Erasure.** B's common-set accuracy falls back toward A's as its train accuracy recovers to 1. Then the objective, with these 299 pairs and decay, does not hold the filtered network's held-out answers. The edit supplied them; training removes them.
- **Persistence.** B keeps most of its held-out accuracy while train accuracy recovers. Then the filtered network sits where fitting and those answers coexist under this objective. That says nothing yet about a hidden algorithm: e2 showed fitted readouts on random features also filter to partial answers.
- **Strengthening.** B rises above its post-edit level, perhaps toward the 50% runs. Then the edit moves the network onto a path that training continues, which the unedited run never found in 50,000 updates.
- **Sham-equivalent.** C ends where B ends (in held-out accuracy and frequency structure). Then the trajectory reflects a large label-free perturbation plus a reset, not keeping the dominant frequency.
- **Seed dependence.** Three checkpoints that differ in direction would limit any statement to "can happen".

My expectation, stated before running: training accuracy recovers in B within a few hundred updates, and B's held-out accuracy falls part of the way back. Retaining a few tenths seems plausible; full erasure would not surprise me. I expect C to recover train accuracy more slowly and to stay near chance on held-out pairs. I hold both expectations loosely.

## Failure modes of this design

- The reset is itself an intervention. Fresh Adam moments with bias correction give large first steps, and A tests only part of that, since A starts with zero gradient-noise history at a point it had converged to.
- B and C are not matched on gates, output displacement or immediate damage. A difference between them may come from those mismatches and not from which frequencies were kept.
- Three checkpoints from one split and one width. Their evaluation pairs are shared, so pair-level results are not independent replicates.
- The filter was not tuned, and the sham seeds are fixed here before any held-out evaluation. Neither may be changed after seeing held-out accuracy.
