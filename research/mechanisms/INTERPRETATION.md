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
