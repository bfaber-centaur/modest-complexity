# Size/time lane e1: interpretations

Read [README.md](README.md) and its figures first. Each entry is one candidate explanation, its strongest alternative, and an experiment that would separate them. None was predicted before the runs.

## 1. Fit without generalization

Candidate: under this fixed recipe and split (299 training pairs, 31% of the table), every width memorizes the training table and none finds a generalizing solution within 50,000 updates. Width buys faster fitting, lower train loss and larger train margins (packet 1), not held-out accuracy. At width 128 the shuffled-label table is fitted at the same speed and along nearly the same loss curve as addition, which is what pure memorization would look like.

Alternative: generalization is delayed beyond the horizon. Delayed generalization in modular arithmetic is known to slow sharply as the training fraction falls, and 31% is low. For widths >= 128 the runs reach an apparently stationary state by about 20,000-25,000 updates (norm near 60, adaptive and decay displacements equal and opposite with cosine near -1, train loss flat; [updates_add.png](figures/trajectories/updates_add.png), [norm_add.png](figures/trajectories/norm_add.png)), which argues against an imminent change but does not exclude a slow one. Width 64 is still approaching that state at 50,000 and width 32 is further from it.

Separating experiment: the coordinator's 200,000-update width-512 extension, and README follow-up 1 (width 64, whose trajectory is still changing at 50,000). Follow-up 2 (larger training fraction) asks whether the same recipe generalizes when given more of the table.

## 2. Held-out loss at comparable training loss

Candidate: at the same training loss, wider networks are less confidently wrong on held-out pairs, perhaps because the memorized solution is spread across more units so that held-out logits are less extreme.

Alternative: the ordering is elapsed training, not width. Narrower networks need more steps to reach the same training loss, and held-out loss grows with steps in every addition run. The table in packet 2 cannot separate the two because width and time to reach a given loss co-vary almost perfectly. The parameter norm at the crossing is similar across widths, so a raw norm difference does not explain it, but the norm is not a scale-free measure.

Separating check: at matched training loss, compare the scale of held-out logits with that of training logits per width (from published checkpoints), and compare runs that reach the same loss at similar steps. With a fixed recipe the second needs a learning-rate or schedule change, which would be a separately labeled comparison.

Decay: removing decay (add0) leaves fitting time unchanged at width 128, drives train loss to 1e-9 and raises held-out loss to about 2.4 times its decayed value by step 50,000. In this map decay limits how confidently wrong the network becomes; it does not produce generalization within the horizon.

## 3. Below-chance held-out accuracy

Candidate: a held-out prediction is made from what the network learned about each operand separately. The first layer is additive (`U[a] + V[b]`), so a held-out pair reuses the embeddings of operands seen in training pairs, and the network answers with labels it associated with operand a or operand b. For addition such a label is never correct, so accuracy falls below chance; for the shuffled table the correct label is in that set 44% of the time and accuracy returns to about chance (packet 3).

Alternative: this is a property of any memorizer that interpolates between training pairs sharing an operand (for example a nearest-neighbour rule on operands), combined with the split design (pairs and transposes in the same split), and says little about this architecture or about width. The test is coarse: L covers about half the labels, and "prediction in L" does not show which operand drives a prediction or how.

Separating experiment (mechanism lane): on a published late checkpoint, replace operand a's embedding with that of another value and check whether the held-out prediction moves to a label seen with the remaining operand b, with whole-operand and no-op controls. A second split in which a pair and its transpose can fall in different splits would show whether the transposed training pair then supplies the correct label.

## Not interpreted

Widths 16-64 show more audit-accuracy flicker above zero late in training than wider networks ([overview.png](figures/trajectories/overview.png), zoomed panel); with at most 5 of 544 pairs correct this is left as an observation. Width-16 train-accuracy drops after first fit (2-25 logged steps below .99) coincide with a minimum train margin near zero; whether they are optimizer noise at low margin or competition between memorized pairs was not checked.
