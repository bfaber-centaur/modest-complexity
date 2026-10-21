# e2 mechanisms: interpretation (read [README.md](README.md) first)

Candidate explanations, not conclusions. The evidence covers nine runs at width 128 (six on a+b over one pair of nested splits, three with shuffled labels), and nothing here is audited.

## Packet 2

Candidate (H1, now with a functional test): the failed 30% runs hold a partial addition computation in their weights. In each unit, the component at the unit's own dominant frequency computes a+b well enough to answer about half of the never-trained pairs: 0.43–0.58, or 0.64–0.72 when the input DC is dropped as well. The rest of the same units' weights masks it completely, so unedited accuracy is 0.000–0.004. Packet 1 showed the aligned terms exist and point the right way. This shows they can carry correct answers once the competing part is removed, which was the coordinator's criterion: the edited 30% models answer correctly, and the nulls stay at or near chance (all null draws ≤ 0.106).

Why this is not a masked readout: the edit uses only each unit's own weights and the Fourier basis of Z_31, never labels, pairs or a+b−c. Whether the kept components add up at a+b depends only on their learned phases. Scrambling those phases, with the same frequencies and energies, gives chance. So do the same edit at initialization and on networks fit to shuffled labels.

The 50% runs point the same way:
- The edit raises their held-out accuracy from 0.88–0.91 to 1.000 at 50k.
- At step 1k it reads 0.59–0.67 while the unedited model reads 0.000.
- In the successful runs, then, the frequency-structured computation is present long before the argmax shows it, and their remaining errors at 50k come from the other components.

This resembles hidden-progress measures in the grokking literature, such as the restricted loss of Nanda et al. 2023. The difference is that the restriction here is read from the weights, not from the frequencies of a+b−c in the logits.

What does not hold:
- Upweighting frequency-aligned units does nothing in the 30% runs. The masking part sits inside units, as their other-frequency components and DC, not in a separate set of misaligned units. This fits packet 1, where groups were not special under ablation.
- Neither half of the filter suffices: filtering only the inputs or only the outputs gives 0.13–0.21.

Strongest alternative: fit-induced interpolation. Weights that fit c = a+b on the training pairs, once projected onto one Fourier mode per unit, may interpolate to held-out pairs because the projection is a strong prior on Z_31. On this reading, the filter supplies the inductive bias and the training fit supplies the phases; nothing in training "computes" addition on unseen pairs.
- The shuffled-label control does not rule this out. Those runs never became frequency-concentrated (0.17, the initialization level), so they show only that the filter does not create a+b from structureless weights.
- Packet 1's alternative survives in this form.
- It may not be fully separable from H1. A structure in the weights that answers unseen pairs correctly is a generalizing computation, whatever produced it. What stays open is whether the trained network itself ever uses it.

Next tests:
1. Residual. Split the 30% runs' unedited logits into the filtered model and the rest, and check whether the rest favours operand-associated labels on held-out pairs. That would connect this to the e1 restricted-rank finding.
2. Across the e1 map. The filter is weights-only and cheap, so it can be applied to every published e1 checkpoint (widths 32–512) to see where hidden structure exists without accuracy.
3. Retrospective tracing, from packet 1.
4. A training intervention, which needs a central claim: continue the 30% runs with a penalty on each unit's non-dominant components, or with the projection applied periodically. Does held-out accuracy then appear in the unedited model? This is the test that separates H1 from fit-induced interpolation.

## Packet 1

Candidate (H1 in QUESTION.md, the measurable part): the failed 30% runs build the same kind of term as the successful 50% runs. Their units grouped by frequency k add a cosine of a+b−c at frequency k that is largest at the correct label, on pairs that neither split trains on. In the failed runs that term is about five times weaker and is buried under a held-out contribution that is not a function of a+b−c: about 99% of the variance, against about 73% in the successful runs.

This matches the e1 finding that the restricted rank improves while argmax accuracy stays at zero. The aligned terms raise the correct label among the labels never seen with either operand. They do not overcome whatever lifts the operand-associated labels.

What does not hold:
- Grouping by frequency does not identify especially important units for the function's accuracy. At equal logit change, removing a group costs about as much as removing a matched random set, in both shares. One reading: each frequency's group contributes about 1/15 of a sum, and a random set removes a similar total spread over frequencies.
- The "pizza" form predicted from the architecture explains held-out group contributions only partly. The groups reach shares of 0.09–0.46 against 0.81 for the ideal term.

Strongest alternative: the aligned terms in the 30% runs could be a side effect of fitting training labels that equal a+b, not a computation that generalizes. A unit whose input and output weights share frequency k contributes a frequency-k term in a+b−c on any pair, held-out or not, once its weights fit training pairs with c = a+b. The measurement shows the term exists on held-out pairs and points the right way. It does not show the term was learned as a computation rather than inherited from the weight structure. The 50% runs make the same point with more force: their terms are there before accuracy rises (step 1k–5k).

What would distinguish them: whether these terms can carry correct answers when the competing contribution is removed.
1. Project the logits onto the a+b−c-structured part of each group, or ablate the non-equivariant part, in the 30% runs, and see whether held-out accuracy appears.
2. Retrospective tracing: whether the 50k groups of the 50% runs are the same units, with the same frequencies, as the aligned groups at 1k–5k.
