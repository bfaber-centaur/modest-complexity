# Role: R3 investigator, training history and recovery

Directory `research/regimes/history/`. Run IDs `r1-hist-…`. Seeds 53000–53999 (PLAN §1). Label `r1-hist-cloud-1`. Budget 25 min.

## Bobby's brief (verbatim)

> Question: What survives learning another task, and what determines recovery?
>
> Design:
> - Train task A, then task B, then return to A.
> - Branch B-duration conditions from identical A checkpoints.
> - Compare related and unrelated B tasks.
> - Include continued-A and fresh-A controls.
> - Keep task identity explicit when both tasks are intended to coexist.
> - Treat conflicting mappings without task identifiers as a distinct experiment.
> - Initially reset optimizer state consistently at phase boundaries.
> - Preserving optimizer state is a possible follow-up, not an uncontrolled difference.
>
> Measurements:
> - A and B performance throughout all phases.
> - Updates and examples needed to reacquire A.
> - Recovery using a newly trained readout on frozen features, evaluated on held-out examples.
> - Functional changes over the complete input domain.
>
> First figure: forgetting and recovery curves aligned at task switches.
>
> Failure to recover does not establish complete erasure. Faster relearning alone does not identify what was retained.
>
> Starting literature: https://arxiv.org/abs/1612.00796

Read the paper's abstract, and its body if it bears on a concrete design choice. Record what you read and what you could not retrieve.

## Coordinator's notes

These are things to consider, not requirements. Your QUESTION.md decides.

**Task choice**
- Use a small binary domain, for example 10 bits (1,024 inputs). A must be learnable from a training subset, so that held-out evaluation and the frozen-feature readout mean something.
- "Related" B should share a stated subfunction with A. "Unrelated" B uses disjoint bits with matched difficulty, checked by learning each from scratch.
- Settle both before the main runs and give the reason for each.

**Task identity**
- Separate heads, one A head and one B head on a shared trunk, keep identity explicit without an input flag. A task-ID input bit with one shared head is the other option.
- With separate heads and wd = 0, B training never touches the A head. Any loss of A is then trunk drift, and you can state that exactly.
- The no-identifier conflicting-mapping experiment is a separate labelled experiment: one shared head, with B for example equal to NOT A on the same inputs. Give it its own run IDs, and do not pool it with the coexisting-task runs.

**Phase protocol**
1. Train A for a fixed T_A and save one checkpoint per init seed.
2. Branch every B duration d (for example 0 and three durations spread over an order of magnitude or two), and every control, from that identical file. Record its SHA-256 in each branch's config.
3. Reset Adam (`nets.adam_state`) at every boundary in every arm, including continued-A, so the reset is not itself a difference between arms. Keep wd fixed and stated.
4. Return to A for a fixed T_R and log both tasks throughout.

Branches from one A checkpoint are dependent. The three seeds are three separate A runs.

**Controls**
- Continued-A: A for d updates in place of B, then the same return phase.
- Fresh-A: A from a new initialization. Its updates and examples to criterion are the "no savings" reference.
- Use the same criterion everywhere (for example held-out A accuracy ≥ 0.95, or the end-of-A level), defined before the runs. A run that never reaches it is censored, not assigned a time.

**Measuring what survived**
- **Relearning cost:** updates, and examples × updates, until the criterion is regained.
- **Frozen-feature readout:** freeze the trunk at the end of B, fit a new A readout with a fixed budget on A's training inputs, and evaluate on held-out inputs. Compare with the same readout on end-of-A features, on untrained features and on a row-shuffled null.
- **Functional change:** over all 1,024 inputs, the fraction whose A prediction changed between end-of-A and end-of-B and after return, plus a logit-level distance.

"Not recoverable by a readout" does not mean erased, and faster relearning does not say what was kept. Write down which observation would separate "features retained, readout misaligned" from "features overwritten" before you run.

**Outcomes worth distinguishing in advance**
- A survives B in the trunk. A readout recovers it, and relearning is fast.
- A is functionally lost but a readout recovers it, so the features survive.
- Neither, but relearning is still faster than fresh-A.
- Related and unrelated B differ in the direction opposite to intuition.

## First figure

A and B accuracy, train and held-out, over the three phases, aligned at the A→B and B→A switches. Draw:
- one panel per B condition (related or unrelated × duration);
- continued-A and fresh-A references;
- three seeds as individual traces;
- censored and capped runs marked.
