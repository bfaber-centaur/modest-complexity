# e1 interpretations (read the index figures first)

These are candidate explanations, not conclusions. Each names its strongest alternative.

## Preview 1: fit without generalization, held-out accuracy below chance

Candidate: within 50,000 updates this recipe memorizes the 299 training pairs at every width tested and does not generalize. Held-out accuracy near zero, below the 1/31 chance level, is what a memorizing model with additive input embeddings would be expected to produce on modular addition: for any held-out pair (a, b), its label a+b never occurs among the training pairs of row a or column b (each row of the addition table is a permutation of the labels), so fitting the training rows and columns pushes the correct class down. This property holds for any split of modular addition, not just this grouped one.

Strongest alternative: a label, split or evaluation defect that makes held-out predictions systematically wrong. This is the first item for the independent audit (split identity, labels, accuracy recomputed with independent code). A second alternative is that generalization is delayed beyond 50,000 updates rather than absent; the horizon censors timing, and these runs are not evidence that this recipe cannot generalize.

Missing checks: two more initialization seeds at each width and the other three widths (size/time lane); the no-decay and shuffled-label controls; whether the training fraction (30% of pairs, about 10 per class) is the binding constraint. Changing the recipe would be a separately labeled comparison, not a rescue of these runs.
