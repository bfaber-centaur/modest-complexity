# Mechanism lane: interpretations (read [README.md](README.md) figures first)

Candidate explanations, not conclusions. One seed, one width, one checkpoint.

## Packet 1. Retention/removal on the non-generalizing width-128 checkpoint

Candidate: this checkpoint stores its training fit diffusely across all 15 input frequencies, roughly in proportion to their energy, rather than in a small frequency subset. The spectrum stays close to the flat spectrum of the random initialization, train accuracy grows gradually with the energy retained, and random K-subsets of similar size do almost as well as the top-K. The held-out loss rising as more frequencies are kept fits the coordinator's account of below-chance held-out accuracy: whatever the input weights encode about the training pairs pushes held-out pairs further from their labels, and removing input structure moves predictions back toward the DC-only baseline (CE 3.54).

Strongest alternative: input-embedding Fourier energy is simply the wrong coordinate system for this model. A memorized table can be spread over frequencies for reasons unrelated to how it is computed, and filtering U and V while keeping W fixed also changes which hidden units are active, so the curves partly measure ReLU gating disruption rather than the removal of a "component". A per-unit or output-side (W) analysis could look different.

What would distinguish them: the same curves on a generalizing checkpoint (if the size/time lane or a follow-up produces one), where a Fourier account predicts a sharp rise at small K that random subsets do not reproduce; and an energy-matched null (removing a random perturbation of the same norm and rank) instead of random frequency subsets.

Do not read: the small held-out accuracy above chance at retention K=1 to 3 (0.064 to 0.072, about 35 to 39 of 544 pairs). Random subsets reach 0.105, so this is within the spread of arbitrary filtering of a memorizing model.

## Packet 2. Rescaling sensitivity

Candidate: on this checkpoint, the input-energy frequency ranking is not a property of the computed function. A 4% gain on seven units changes the top-4 set, and random gains of about ±10% (σ=0.1) change it in most draws, with no change in any logit. Because the spectrum is nearly flat, small gauge changes are enough; the ranking's fragility here is partly a consequence of the flat spectrum and may be much smaller for a model with concentrated frequencies (in the synthetic network, lifting k=7, built with 0.9 of k=3's input gain, to rank 1 took a 1.116 gain).

The codec is less fragile than the ranking (unchanged at σ=0.1) but still not a function property: identical functions can cost 95,496 or 190,728 bits. The pruning order is invariant, so all of the sensitivity enters through per-matrix quantization ranges.

Strongest alternative: these rescalings are off the set of points the optimizer actually reaches. Every random draw increased the parameter norm, consistent with weight decay settling near a norm-minimizing balance between each unit's input and output weights. If proxies are only ever compared at that balanced gauge, the sensitivity shown here may not matter in practice. That is a normalization convention, which is the next analysis; it would fix this one symmetry, not every equivalent representation.

Missing check: whether the codec can get shorter under some rescaling (no random draw shortened it, which also fits the balanced-gauge account), and the same audit on a generalizing checkpoint.
