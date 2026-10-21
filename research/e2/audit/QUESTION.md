# e2 audit question (written 2026-10-21, before the comparison below was computed)

Session `e2-au-cloud-1`, branch `claude/project-thread-id1tmm`, base `6aaa3a2` (shared code `fe284a5`).

## Already seen when this was written

From the e1 audit (`research/audit/AUDIT.md` Parts 1–4) I have seen the restricted-rank statistic, which is the rank of a+b among labels never seen with either operand in training:

- by width at 30% (0.47–0.51 at width 16 down to 0.18 at width 512);
- its null in the no-decay and shuffled-label runs;
- its trajectory in the 200k run and in the three 50% runs (0.06–0.12 at step 5k, 0.002 at 50k).

I also know that the 30% and 50% training sets nest, and the common-set scores.

I have not computed the logit rank of any label other than a+b, nor anything indexed by the offset between a label and a+b.

## Question

Is the restricted-rank signal specific to the sum label a+b? Or is a+b one of several labels, defined relative to (a, b), that a model ranks high among the same absent labels?

## Two explanations

A. **Sum-specific.** The held-out logits single out a+b. Any other label at a fixed offset, a+b+k with k ≠ 0, sits at no preference (mean normalized rank 0.5) when ranked among the same absent labels.

B. **Smooth or periodic structure.** The held-out logits vary smoothly with the offset c − (a+b), as a partial Fourier or "clock" representation would make them, for example Σ_ω α_ω cos(ω(a+b−c)). Then a+b is the peak of a profile, and other offsets are also above or below 0.5 in a pattern set by the frequencies in use.

A third possibility, C, is that the statistic is an artifact of how the absent set is built. It is largely excluded already: step-0 checkpoints and all three shuffled-label seeds give 0.5. The offset profile would show C as many offsets far from 0.5 in trained addition models with no relation to k.

## Cheap observation that separates A and B

The offset profile: for each run's final checkpoint, the mean normalized rank of the label a+b+k among the pair's absent labels, for k = 0…30. It uses only pairs where a+b+k is itself absent, with the same mask and normalization as `research/audit/heldout_rank.py`. For each k, the k = 0 value is also reported on the same subset of pairs, so offsets are compared on matched pairs.

- Runs: the 30% addition runs at every width, the no-decay and shuffled-label controls, and the three 50% runs.
- Expected under A: a single dip at k = 0 and a flat profile near 0.5 elsewhere.
- Expected under B: the dip at k = 0 plus a structured profile over k, symmetric in ±k for a cosine readout, with a shape that changes with width and share.

## Failure modes of the diagnostic

- **Subset selection.** For k ≠ 0 only pairs whose a+b+k is absent enter the average, and whether it is absent depends on which neighbouring pairs were trained. So different k use different, structured subsets of pairs. The matched k = 0 value on each subset controls for this, but not perfectly.
- **Thin subsets.** For the 50% split few labels are absent per pair (median 8), so some offsets have few usable pairs. Counts are reported with the profile.
- **Not a mechanism.** A smooth profile would show structure in the ranking among absent labels, not which algorithm the network implements, and not a deployable predictor.

## Added after the profile was computed (2026-10-21, ~20:00 UTC)

A failure mode not listed above: a readout spread evenly over many frequencies, Σ_ω cos(ω(a+b−c)) with similar α_ω, is itself a sharp single peak at c = a+b (a Dirichlet kernel). So a single dip at k = 0 with a flat profile elsewhere does not exclude B. The profile can only show B when a few frequencies dominate. The result is therefore read as "no offset other than k = 0 is preferred", not as "the readout is not periodic".
