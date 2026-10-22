# Role: R2 investigator, shared subcomputations

Directory `research/regimes/shared/`. Run IDs `r1-share-…`. Seeds 52000–52999 (PLAN §1). Label `r1-share-cloud-1`. Budget 25 min.

## Bobby's brief (verbatim)

> Question: Does learning one task help another because the model reuses a shared computation?
>
> Starting task family:
> z = x1 XOR x2
> y1 = z XOR x3
> y2 = z AND x4
>
> Use ten binary inputs, including irrelevant bits. Exhaustive evaluation has 1,024 inputs. Inspect label imbalance and include appropriate baselines; modify the task family if necessary, documenting why before the main runs.
>
> Design:
> - Compare joint training and separate training.
> - Compare acquisition of y2 after y1 pretraining against fresh initialization.
> - Include a task pair without the designated shared intermediate.
> - Include frozen random-feature and frozen pretrained-feature comparisons where useful.
> - Report both parameter budgets and per-task training exposure.
> - Avoid treating more supervision or more updates as evidence of reuse.
>
> Measurements:
> - New-task learning curves versus number of examples and updates.
> - Dependence on irrelevant inputs.
> - Candidate shared representations and selective interventions.
>
> First figure: transfer learning curves for related and unrelated pretraining.
>
> Transfer alone does not prove that the model computes our chosen z.
>
> Starting literature: https://arxiv.org/abs/2303.11873

Read the paper's abstract, and its body if it bears on a concrete design choice. Record what you read and what you could not retrieve.

## Coordinator's notes

These are things to consider, not requirements. Your QUESTION.md decides, and any task change is documented there before the main runs.

**Imbalance**
- y1 is balanced over the domain.
- y2 = z AND x4 is 1 on a quarter of inputs, so "always 0" scores 0.75. y2 also carries no information about z when x4 = 0.
- Report balanced accuracy and cross-entropy against a majority baseline, or change y2 to a balanced function that still needs z. Decide before the main runs.

**The designated intermediate is not forced**
- y1 = x1⊕x2⊕x3 is symmetric. A network can compute it through x1⊕x3, x2⊕x3, or no pairwise intermediate at all, so y1 pretraining need not build a z = x1⊕x2 representation.
- y2 = (x1⊕x2)∧x4 does need x1⊕x2 when x4 = 1, but can be computed as (x1∧x4)⊕(x2∧x4).
- So "shared z" is a hypothesis about the network, not a property of the tasks. That is exactly why transfer alone does not show reuse.

**Unrelated pair**
- For example, pretrain on y1' = x5⊕x6⊕x7: the same function class on disjoint bits, so matched difficulty and no shared intermediate.
- A partially related variant shares input bits but not z, for example x1⊕x3⊕x5.
- Both kinds of pretraining should get the same updates and examples. Then "pretraining at all" (exposure, norm growth, optimizer warm state) is not mistaken for reuse.
- Reset the optimizer at the switch in every arm.

**Learning curves**
- Train the new task on n training inputs, for example from 16 to 512. Evaluate on all held-out inputs and on the full 1,024.
- With 7 irrelevant bits there are only 8 relevant patterns for y2, so small n may miss some patterns entirely. Report pattern coverage per n.
- Draw training subsets nested across n, and paired across arms.

**Parameter budgets.** Separate training with two full networks has about twice the trunk parameters of joint training. Say which one you equalize, widths or parameters, and report both.

**Frozen features.** Fit a fixed-budget readout for y2 (logistic or ridge, same examples, no test tuning) on:
- y1-pretrained features;
- unrelated-pretrained features;
- untrained features;
- a row-shuffled-feature null.

**Irrelevant-input dependence.** For each input bit, the fraction of the 1,024 inputs whose prediction flips when that bit alone flips, before and after transfer.

**Selective intervention, if a z-like direction exists**
- Interchange or flip it.
- Check that y1 flips, that y2 flips only where x4 = 1, and that y2 does not change where x4 = 0.
- Use a direction decoding a different XOR, such as x1⊕x3, as the control.

**Outcomes worth distinguishing in advance**
- No transfer.
- Transfer from related and unrelated pretraining alike: exposure, not reuse.
- Related-only transfer without a selective z intervention: transfer through something other than our z.
- Related-only transfer with a selective z intervention.

## First figure

New-task (y2) learning curves after related pretraining, unrelated pretraining and fresh initialization, against updates and against training-set size n. Draw:
- three paired seeds as individual traces;
- train next to held-out performance;
- the majority and balanced baselines;
- capped and failed runs explicitly.
