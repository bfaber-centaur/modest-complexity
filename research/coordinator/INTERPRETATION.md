# e1 interpretations (read the index figures first)

These are candidate explanations, not conclusions. Each names its strongest alternative.

## Preview 1: fit without generalization, held-out accuracy below chance

Candidate: within 50,000 updates this recipe memorizes the 299 training pairs at every width tested and does not generalize. Held-out accuracy near zero, below the 1/31 chance level, is what a memorizing model with additive input embeddings would be expected to produce on modular addition: for any held-out pair (a, b), its label a+b never occurs among the training pairs of row a or column b (each row of the addition table is a permutation of the labels), so fitting the training rows and columns pushes the correct class down. This property holds for any split of modular addition, not just this grouped one.

Strongest alternative: a label, split or evaluation defect that makes held-out predictions systematically wrong. This is the first item for the independent audit (split identity, labels, accuracy recomputed with independent code). A second alternative is that generalization is delayed beyond 50,000 updates rather than absent; the horizon censors timing, and these runs are not evidence that this recipe cannot generalize.

Missing checks: two more initialization seeds at each width and the other three widths (size/time lane); the no-decay and shuffled-label controls; whether the training fraction (30% of pairs, about 10 per class) is the binding constraint. Changing the recipe would be a separately labeled comparison, not a rescue of these runs.

## Follow-up 1: the preference saturates instead of turning into accuracy

Candidate: under this recipe and split, width 512 reaches a stationary state by about 50,000 updates (flat train loss, flat norm) in which a partial relational preference for the true label coexists with argmax predictions that always go to labels seen with an operand in training. Within 200,000 updates nothing suggests a delayed transition: the preference plateaus, and held-out loss only drifts down by 0.3 nats.

Strongest alternative: a transition could still occur on a much longer time scale, driven by slow drift that these summaries do not resolve; the held-out loss is still decreasing slowly at 200,000. A second alternative is that the binding constraint is the data (299 training pairs, about 10 per class) rather than width or time, which a horizon extension cannot test.

## Follow-up 2: the training share, not width or horizon, separates the two regimes here

Candidate: in this recipe the 30% split leaves too few training pairs per class for the structured solution to overtake memorization within the horizons tried, while at 50% the same width and seeds memorize first and then generalize gradually over tens of thousands of updates. The e1 rank preference would then be the same structure seen below the threshold: in the 50% runs it appears before held-out accuracy moves, and accuracy follows as predictions leave the labels seen with an operand.

Strongest alternative: the 50% split differs from the 30% split in more than size (a different partition, a smaller audit split of 364 pairs, about 16 rather than 10 training pairs per class), so the contrast is one comparison of two partitions, not a dose-response curve. A second alternative is that the same transition happens at 30% on a time scale much longer than 200,000 updates; follow-up 1 makes that look unlikely at width 512 but does not rule it out.

Missing checks: an independent audit of e1f2; intermediate training shares (for example 35-45%) to locate the threshold; widths other than 128 at 50%; a second partition at 50% (different data seed).
