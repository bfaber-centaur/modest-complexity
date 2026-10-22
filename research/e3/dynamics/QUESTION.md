# e3 dynamics: question log

Session "Width and training-time map" thread (e1 size/time, e2 dynamics). Branch `claude/project-thread-j8mgkm`. The e3 base, seeds and run IDs are not pinned yet; this entry is written before any e3 setup or computation. Newest entries last; earlier entries are not edited.

## Q1 (2026-10-22, before computing)

### What is already known (e2, one data seed 31000)

- Along the nested ladder 30% ⊂ 40% ⊂ 45% ⊂ 50% (299, 359, 417, 478 training pairs; 5-8 of each class's 16 unordered groups), common-pair accuracy at 50,000 updates is 0.000-0.004, 0.023-0.025, 0.34-0.54 (still rising) and 0.88-0.91, three init seeds per share.
- The underlying margin of a+b over labels never seen with either operand rises with share without seed overlap from 4,000 updates on, in logits and in units of the logit spread (e2 audit cycle 4). Accuracy is steep because that one graded margin has to clear a lead of about two logit standard deviations held by operand-associated labels; the lead changes little with share or training.
- The 40% and 45% sets add only e1 calibration groups, in one seeded order, so amount and identity of added pairs moved together (e2 open question 1).

### Question

Three new data seeds each give a different nested ladder at the same four nominal settings. Does the e2 ordering travel to them, does the transition (the setting where held-out accuracy at 50,000 rises from near zero to most pairs) move, and in this small sample does the choice of data seed move results more than the choice of init seed?

### Expected patterns and what each would mean

- Ordering and location both travel: in every ladder accuracy at 50,000 rises with training count, near zero at 30%, high at 50%, with the steep step between 40% and 50%, and the spread between ladders at a setting is no larger than the spread between the two init seeds within a ladder. Then the e2 ladder was not a property of seed 31000's particular groups, at this resolution.
- Ordering travels, location moves: within every ladder more training pairs never hurt, but the step sits at a different setting in some ladder (for example a 40% run that answers most pairs, or a 50% run that does not by 50,000), and ladder-to-ladder spread at a setting exceeds init spread. Then which pairs are trained on matters at a fixed count, and the follow-up should look at what differs between those training sets.
- Ordering fails inside a ladder: a larger nested set does worse than a smaller one at 50,000 for both init seeds. Because the sets nest, this would point at specific added pairs or at timing (a run still rising at 50,000), and would need trajectories, not only end points, to read.
- The graded margin travels while accuracy does not (or the reverse): accuracy is a threshold readout, so a modest shift of the margin between ladders could move accuracy a lot near the step. Ladders are compared on both before calling a location shift.

### What gets measured

Per ladder, on that ladder's pairs never trained at 50% (each ladder its own set, shown separately; pairs are not independent replicates and ladders do not pool): accuracy and cross-entropy against actual training-pair count and over training; the margin of a+b over labels never seen with an operand, in units of each pair's wrong-label logit spread; the restricted rank of a+b among labels absent from that run's own operand-label sets; and the scale-free operand-label preference the auditor is defining from e2 artifacts (fraction of operand-associated versus unrelated label pairs ordered in favor of the operand-associated label). Definitions follow the auditor's where they exist; raw logit differences are not read as growing preference.

### Failure modes

- Two init seeds per cell: init spread is barely estimated, so "data seed matters more than init" can only be a statement about this sample.
- 50,000 updates may cut runs that are still rising (as 45% was in e2); trajectories are reported with end points, and "not yet" is kept separate from "fails".
- Nominal shares are not exact counts: the diagonal group of each class holds one ordered pair instead of two, so each ladder's actual training counts differ by a few pairs; the x axis uses actual counts.
- Each ladder's evaluation set differs and contains about 480 pairs whose labels and operands overlap heavily; pair-level fractions are descriptions of one ladder, not samples from a population.
- Data seed changes which groups are trained and in which order they are added together, so a location shift says the training-set identity matters without saying which property of it.
