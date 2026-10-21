# e2 audit, cycle 4: question (written 2026-10-21, about 20:15 UTC, before any of the computations below)

Session `e2-au-cloud-1`, base `dfc9577` (swarm/exploration after the e2 index and synthesis). Target: dynamics packet 2 (`research/e2/dynamics/`, merged at `7f4490b`): the six runs at 40% and 45% training share, the margins table, and INTERPRETATIONS.md 2 ("a race between two graded quantities") with its dilution alternative. Also: whether `research/e2/coordinator/SYNTHESIS.md` overstates the audit's verdicts.

## Already seen when this was written

- The packet 2 README section and table, INTERPRETATIONS.md 1–2, QUESTION.md Q2 with its outcome, the synthesis and the index claim table. Not the figures or the CSV rows.
- For one run (`e2-dy-add-w128-i30000-tf40-a1`): config, runtime, manifest header and the last logged rows (own audit accuracy 0.0228 at 50k; the 40% audit split is the common set). The runtime records `git_status: ?? launch.log`, which is why the manifest says the producing tree was dirty.
- `git diff fe284a5 dbfcf21` is empty for the shared code and environment files.
- Nothing computed yet.

## Questions

1. **Runs.** Are the six runs what was claimed and do their published files reproduce?
2. **Numbers.** Do the packet's margins, gaps, orderings, first-crossing steps and pair transitions reproduce from code written from the README definitions?
3. **The race.** For each pair the margin over S equals the margin over N minus the S − N gap, so "a pair is correct only if the margin over N exceeds the gap" is arithmetic. What carries evidence is how the two terms move with share. Logit scale changes across runs and steps, and the gap is a difference of maxima over sets of different size (S about 16 labels, N about 7), so part of it grows with the spread of the logits whatever the network prefers. Does "the gap grows more slowly, and the margin over N rises faster, at higher share" survive in units of each pair's logit spread, at matched training fit?
4. **Dilution.** Is a slower-growing gap at higher share a generic property of a memorizer given more pairs per operand? A training-free memorizer can be built at all four nested shares: the step-0 U, V of each init seed with W fitted in closed form to one-hot a+b on that share's training pairs (cycle 3's ridge control, λ = 1e-6 and 1e-2 of the mean Gram eigenvalue). It answers almost nothing held out at 30% (cycle 3: 0.00–0.01).

## Diagnostic

- **A, runs.** Claims at the launch commit `dbfcf21`; registry events; config against the claim; splits regenerated with the audit's independent code (`research/audit/independent.py`), hashes against the plan, nesting 30 ⊂ 40 ⊂ 45 ⊂ 50 and the common set untrained at every share; manifest hashes and checkpoint schedule; source hashes against `dbfcf21`; logged metrics recomputed at all 16 published checkpoints; independent AdamW replays from an independent initialization through first fit to 5k (early) and from 40k to 50k (late), all six runs. Tolerances as in e1: recomputation 1e-12 relative with identical accuracies; short replays 1e-8; replays longer than about 10k updates are judged by logged accuracy and cross-entropy (rounding divergence, e1 AUDIT F11).
- **B, numbers.** S, D, N rebuilt from the frozen 30% and 50% splits; margins at every published checkpoint of the 12 width-128 runs; compared with `margins_boundary.csv` and the README table and bullets.
- **C, scale.** Per pair, divide margin over N and gap S − N by the standard deviation of the pair's 30 wrong-label logits; medians over the 483 pairs; at 50k and interpolated (log training cross-entropy) at 1e-4. Report also a set-size null for the gap: the expected max over |S| minus max over |N| when the pair's wrong-label logits are assigned to S, D and N at random (200 draws per pair, seed 36200).
- **D, dilution.** The ridge memorizer at 30, 40, 45 and 50% for three init seeds and both λ, scored unedited on the common pairs with the same quantities, raw and in σ units.

## Decision rules, fixed now

- **C.** "The gap grows more slowly at higher share" survives scale removal if, at 50k and at matched cross-entropy 1e-4, the σ-unit gap is ordered 30 > 40 > 45 > 50 with no overlap between neighbouring shares' seed ranges. It is contradicted if the 30% and 50% ranges overlap or the order reverses; anything else is ambiguous. The same rule, reversed in order, for the margin over N.
- **D.** Dilution as a memorizer property is supported if the memorizer's σ-unit gap falls from 30% to 50% (mean over seeds, at either λ) by at least half the relative fall of the trained runs' σ-unit gap at matched cross-entropy 1e-4. It is contradicted for this memorizer if the gap does not fall (change ≥ 0, or within the seed spread) at both λ. Anything else is ambiguous.

## Failure modes

- The ridge readout is not a trained network with decay. A positive D shows that a gap falling with share needs no answer-finding; a negative D speaks only for this memorizer.
- If the ridge memorizer answers held-out pairs at 50%, it is not a pure memorizer; its accuracy is reported.
- Medians over pairs, one data seed, one order of added pairs, three init seeds. Neighbouring shares differ by 60 or 58 training pairs.
- σ-units remove a common scale per pair, not every scale effect: the logit distribution's shape can still differ between runs.
