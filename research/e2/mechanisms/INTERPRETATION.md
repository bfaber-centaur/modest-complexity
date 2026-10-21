# e2 mechanisms: interpretation (read [README.md](README.md) first)

Candidate explanations, not conclusions. The evidence covers six runs at width 128 over one pair of nested splits, and nothing here is audited.

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
