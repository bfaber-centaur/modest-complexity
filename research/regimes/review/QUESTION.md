# r1 independent review: what I check, in what order, and with what budget

Reviewer for the r1 regime expedition ([../PLAN.md](../PLAN.md), role [../coordinator/prompts/05-review.md](../coordinator/prompts/05-review.md)). Run IDs `r1-rev-…`, seeds 55000–55999, label `r1-rev-cloud-1`. Results go in [REVIEW.md](REVIEW.md), one section per regime plus one for shared code. I never edit a lane's files. I report, and I may propose a patch on this branch.

## Question

For each regime, is the leading interpretation supported by correct oracles and splits, correct losses and metrics, matched exposure across controls, interventions that change only what they claim to, correct checkpoint and optimizer handling, and figures that show what the note says? What is the best counterexample I can find to it?

## Order of work

1. **Shared code** (`common/nets.py`, `run_jobs.py`). [check_common.py](check_common.py) rewrites the loss from its definition and checks every gradient entry against central differences of that rewrite, not of `nets.loss_grad`. The case has three hidden layers, three heads, rows unlabeled per head, a fully unlabeled head and a row unlabeled everywhere. It runs Adam with decoupled decay through a freeze/unfreeze schedule against a separate loop. It checks that `adam_state` is a full reset, that a checkpoint round-trips and continues bitwise, and the runner's complete/failed/capped/reuse semantics on a throwaway plan. Done first; results in REVIEW.md §0.
2. **Each lane's M0 and M1, as they arrive.** Before reading a lane's code, I write my own oracle from the brief and its QUESTION.md and enumerate the domain. I compare labels, split membership and hashes with the lane's published data. Then I check the leakage conditions the brief names:
   - R1: equivalent programs, permutation identities, position coverage;
   - R2: label balance, pattern coverage per n;
   - R3: identical A checkpoints across branches, by contents as well as file hash;
   - R4: base-level splitting, nested exceptions, realized reliability.
   These are reported fast, because the lanes build on them.
3. **Each lane's M2 figure and M4 note.**
   - Recompute one seed's key numbers from published artifacts with my own code.
   - Tabulate exposure across arms: updates, examples, labels per example, parameters, optimizer resets.
   - State what each intervention holds fixed and what it changes.
   - Map each NOTE.md claim to the figure or table it cites.
4. **Counterexamples**, at least one per regime (below). Each prediction is written here before the run or analysis, and a failed attempt is recorded as a result.

## Counterexamples planned before seeing any result

These are written before any lane result exists, so they target the leading readings the briefs anticipate. I will sharpen each one once the lane's QUESTION.md fixes its design, and record that revision under "Revisions" before running.

- **R1.** If intermediate supervision is reported to help held-out compositions, the reading to beat is that the help comes from extra loss terms, not from the information in the labels. A second reading to beat is that held-out "success" is concentrated on items a partial-program or nearest-program heuristic already answers.
  - Counterexample: break held-out accuracy down by whether the answer equals a heuristic's answer. *Prediction:* if the composition reading is right, accuracy on heuristic-disagreeing items is well above the heuristic baseline on those same items, which is 0 by construction. A cheap shuffled-intermediate-label run, if the lane did not run one, is the other attempt.
  - If both conditions fail, the reading to beat is "optimization, not generalization". I check training accuracy at the cap.
- **R2.** If related pretraining is reported to speed y2, the reading to beat is that y1 pretraining merely grows input weights on x1, x2, x3 (relevance, not z).
  - Counterexample: pretrain on x1⊕x3⊕x5, which shares input bits but not z, with matched updates. Alternatively, analyze whether a z direction's intervention is selective (y2 changes only where x4 = 1).
  - *Prediction under reuse of z:* the shared-bits-no-z arm transfers less than the related arm.
- **R3.** If a frozen-feature readout is reported to recover A after B, the reading to beat is that A is readable from any features of that width, including random ones.
  - Counterexample: compare with the untrained-feature and row-shuffled readouts at the same budget.
  - If faster relearning is reported, the reading to beat is generic savings from any prior training, through norm or scale. I check against unrelated B and continued-A, and if needed a fresh-A with weight norms matched to the end-of-B network.
  - *Prediction under retention:* related-B relearning beats the norm-matched fresh-A.
- **R4.** If the rule is reported to "take over" below some reliability, the reading to beat is shortcut plus memorized exceptions.
  - Counterexample: compare training accuracy on exception bases with reversed-environment accuracy on held-out bases, and look at rule-bit flip rates on held-out bases.
  - *Prediction if the rule was learned:* reversed accuracy is near train accuracy, and rule-bit flips change about all predictions.
  - Secondly, flip rates are off-distribution at ρ = 1 (s and y never disagree in training). I check that "shortcut reliance" agrees with independent-environment accuracy.

## Budget plan (20 min of numerical execution, PLAN §4)

| item | planned | notes |
|---|---|---|
| shared-code checks | < 0.5 min | code checks only; [check_common.py](check_common.py) took 3.5 s including a 2 s sleep |
| oracle and split checks, four lanes | ≈ 1 min | enumeration, no training |
| recomputation from published artifacts | ≈ 2 min | analysis only |
| counterexample runs, ≤ 2.5 min per regime | ≈ 10 min | through `run_jobs.py`, plan entries pushed first |
| contingency | ≈ 6 min | stop and ask the coordinator before exceeding 20 |

Stopping rules:
- No reviewer run reproduces a lane's grid. A counterexample uses the fewest seeds that can falsify. One seed is enough to show that a claimed universal fails; three paired seeds are needed before I call a pattern.
- A run that caps is kept, reported and not silently retried.
- If a counterexample needs more than 2.5 min, I write it up as a proposed follow-up for the lane instead.

## Status values

- **PASS:** checked with independent evidence.
- **CONCERN:** a real issue that limits a claim but does not invalidate the runs.
- **BLOCK:** the affected runs, metrics or claims cannot be used as stated.
- **NOT YET CHECKED.**

Each status names what it affects.

## Revisions

(Counterexample refinements are recorded here, dated, before they are run.)
