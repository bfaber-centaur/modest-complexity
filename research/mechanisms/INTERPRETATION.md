# Mechanism lane: interpretations (read [README.md](README.md) figures first)

These are candidate explanations, not conclusions. All evidence so far comes from one seed and from models that do not generalize.

## Packet 1. Retention/removal on non-generalizing checkpoints

Candidate: these checkpoints store their training fit diffusely across all 15 input frequencies, roughly in proportion to energy, rather than in a small subset of frequencies. The evidence:

- The spectrum stays about as flat as at initialization.
- Train accuracy grows gradually with the energy retained.
- Random K-subsets do almost as well as the top-K, even though they carry less energy.

The pattern repeats at widths 32, 128 and 512 and at the first-fit step 300. The width-512 and step-300 checkpoints show the clearest advantage of top-K over random subsets, which is still modest: 0.53 versus at most 0.49, and 0.32 versus at most 0.31, at K=4.

Held-out loss rises as more frequencies are kept. That fits the coordinator's account of below-chance held-out accuracy: whatever the input weights encode about the training pairs pushes held-out pairs further from their labels, and removing input structure moves predictions back toward the DC-only baseline.

Strongest alternative: input-embedding Fourier energy is the wrong coordinate system for these models. A memorized table can be spread over frequencies for reasons unrelated to how it is computed. Filtering U and V while keeping W fixed also changes which hidden units are active, so the curves partly measure disruption of ReLU gating rather than removal of a component. A per-unit or output-side (W) analysis could look different.

What would distinguish them:

- The same curves on a generalizing checkpoint. A Fourier account predicts a sharp rise at small K there, one that random subsets do not reproduce.
- An energy-matched null in place of random frequency subsets.

Do not read too much into the small above-chance held-out accuracy at small retained K (at most 0.072). Random subsets reach 0.105, so this is within the spread of arbitrary filtering of a memorizing model.

## Packet 2. Rescaling sensitivity of raw proxies

Candidate: on these checkpoints, the input-energy frequency ranking is not a property of the computed function. Gains of 2 to 23% on 2 to 33 units change the top-4 set, and ±10% random gains change it in most draws, with no change in any logit.

Because the spectra are nearly flat, small gauge changes are enough. The fragility is partly a consequence of flatness and may be much smaller for a model with concentrated frequencies. In the synthetic network, lifting k=7 (built with 0.9 of k=3's input gain) to rank 1 took a gain of 1.116.

The codec is less fragile than the ranking but still not a property of the function. Identical functions cost 95,496 or 190,728 bits at width 128, and the code can get longer (final checkpoints) or shorter (step 300) under rescaling. The pruning order is invariant, so all of this sensitivity enters through per-matrix quantization ranges. With only four quantization levels, the codec jumps by factors of two, and a checkpoint near the q=8 fidelity threshold flips easily.

Strongest alternative: these rescalings are off the set of points the optimizer actually reaches, so in practice the sensitivity may not matter. Packet 3 tests this directly.

## Packet 3. Balanced gauge

Observation summary: the convention works as designed. Every rescaled copy normalizes to the same weights, so the ranking and codec of the balanced copy are rescaling-invariant.

On these trained checkpoints, balancing changes little. The top-4 set is unchanged at all four checkpoints, and the codec is unchanged except at step 300, where it halves. That fits the packet-2 alternative: the trained weights sit near, but not at, the minimum-norm gauge. Every unit has its input norm somewhat larger than its output norm. This is the direction expected from initialization, where input weights (SD .2 over 62 entries) start much larger than output weights (SD 1/sqrt(width) over 31 entries). It also fits the gap shrinking with training (step 300 versus 50k) and with width.

What the convention does not do: it does not turn energy into a measure of functional importance. In the synthetic network, four groups with identical logit contributions get balanced energy shares from 0.11 to 0.53, because balanced energy tracks per-unit weight magnitude, including the parts of units' contributions that cancel each other. A function-sensitive ranking, such as the loss change when a frequency is removed alone, would answer a different question from any weight-energy ranking under any gauge.

Strongest alternative to "balanced proxies are the ones to report": the balanced gauge is one convention among several (for example, unit-norm inputs). Its codec numbers are as conventional as the raw ones, just reproducible. The choice should be stated with every codec or ranking number, not treated as removing the measurement question.

Missing check: everything here on a generalizing checkpoint.

## Follow-up. Held-out label preference under frequency interventions

Candidate: the held-out preference for the true label among absent labels is spread across input frequencies, as the training fit is. Each frequency on its own already carries some of it: one retained frequency moves the rank from 0.50 to about 0.38. Removing frequencies wears it down gradually, and at width 512 the top-energy frequencies are not special. At width 128 they carry somewhat more than random frequencies, so "spread" holds more strictly at the larger width.

The norm-matched null dissociates the two readouts. Random-direction perturbations of the same size hurt the preference more than removing whole frequency components, but hurt the training fit much less. One reading: the training fit lives in large logit margins, around 14 nats, which tolerate unstructured noise. The preference lives in small differences among labels that all sit far below the present labels, so it depends on the coherent, fine-grained structure that removing a frequency leaves intact in the remaining components.

The rotated null adds energy on average while removal subtracts it, so "more damage to the preference" could have been a larger change to hidden pre-activations rather than anything about frequency structure. The random-subspace null rules that out. It is a projection of the same rank that removes slightly less energy than top-K removal, and it damages the preference even more than the rotated null (w512, K=8: rank 0.42 to 0.43, against 0.35 to 0.37 rotated and 0.32 top-K). Its damage to the training fit matches top-K removal. The picture across the four interventions:

- Training fit tracks how much energy is removed, roughly whatever the directions. The rotated null, a change of the same norm that adds energy on average, harms it much less.
- The preference tracks whether the change is aligned with frequencies. Removing whole frequencies, top-energy or random, preserves it best. Changes in random directions damage it, whether they add or remove energy.

Candidate reading: frequency projections commute with cyclic shifts of the tokens and random subspace projections do not. If the preference comes from translation structure (the model treats a held-out pair somewhat like shifted training pairs with the same sum), then a shift-commuting removal keeps that structure intact in the frequencies that remain. A generic removal distorts each token's embedding by a different amount and corrupts it in every frequency at once. The training fit, a lookup of individual pairs, has no reason to care.

Strongest alternative: the Fourier basis is special here for a reason other than translation. Frequency projections also commute with reflection of the tokens, and they keep cos and sin of a frequency together. The removals also differ in which hidden units they switch off. A perturbation that commutes with shifts but removes nothing, such as random per-frequency phase rotations applied identically to U and V, would test translation directly. A translation reading predicts that it spares the preference relative to a non-commuting perturbation of the same size.

Do not read this as evidence that the model "knows" addition. The preference is a ranking among labels that the argmax never selects.

## Follow-up. The 200k-update continuation

Observation summary: from 50k to 200k updates every readout in this lane stays where it was, even though the weights move about 17% at constant norm. A reading consistent with this: after the fit, weight decay and the loss move the model along a set of near-equivalent memorizing solutions. The movement is not concentrated in any frequency, so it neither builds Fourier structure nor erodes the partial preference. Longer training alone does not turn this run's preference into held-out accuracy, as the coordinator's accuracy curves already show. The contrast these packets have been missing is a checkpoint that does generalize.
