# e2 mechanisms: interpretation (read [README.md](README.md) first)

Candidate explanations, not conclusions. The evidence covers twelve runs at width 128: six on a+b with weight decay over one pair of nested splits, three on a+b without decay and three with shuffled labels, both on the 30% split. Packet 1 was audited in e2 audit cycle 2; packet 2 and its close-out are not audited.

## Where the lane ends (close of cycle 2, revises packet 2)

Revised reading: the per-unit filter does not show that the decayed 30% runs hold a computation of their own. The no-decay runs fit the same a+b labels with much weaker per-unit structure (concentration 0.22–0.23 against 0.40–0.44), and they give 0.33–0.47 under the same filter against 0.43–0.58. Their nulls stay as low. At every checkpoint, both groups keep the same fraction of their retained train accuracy on held-out pairs (0.57–0.78 against 0.62–0.76). The filter's held-out accuracy in the failed runs is therefore mostly what any network fit to a+b on these pairs gives under this projection. That is the fit-induced alternative from packet 2, and it now has direct support: the projection supplies the prior, and fitting a+b supplies the phases.

What survives:
- Weight decay changes the weights in measurable ways that this edit does not turn into extra held-out accuracy. It roughly doubles per-unit concentration and the energy the filter keeps. It halves the size of the edit. It makes each half of the filter work on its own (0.13–0.21 against 0.03–0.08).
- From the audit (cycle 2): with decay, the aligned term is about five times larger, group-specific and aligned in 45/45 groups. Without decay it is aligned in 34/45, with no group-specific share and a null restricted rank. None of these is shown to be functional.
- The 50% runs' filtered models stand nearer the diagonal of held-out against train accuracy (0.83–1.00). They also read 0.59–0.67 at step 1k while unedited they read 0.000. They have more training pairs, so this does not isolate a mechanism either.
- Packet 1's H1 ("the 30% runs build the same kind of term as the 50% runs, only weaker") holds descriptively. But a weaker version of that term also appears without decay, where nothing else points towards generalization. Its presence and sign are not evidence of progress towards the 50% computation.

What packet 2 still shows: the edit is label-free, and its result depends on learned phases. Scrambled phases, step 0 and shuffled labels give chance. Any a+b fit on these pairs leaves phase relations that a one-frequency-per-unit projection turns into partial held-out accuracy. That is a property of fitting this task with this architecture, not of the failed runs specifically.

Best next question (recorded only, not run in e2; no claim drafted): does penalizing each unit's non-dominant components *during training* make held-out accuracy appear in the unedited 30% model, and does it do so differently with and without weight decay? This is the test that could separate structure training actually uses from structure a projection reads into any fit.



## Packet 2 (as first written; revised above by the no-decay comparison)

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
