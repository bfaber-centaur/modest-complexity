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

## Q1 outcome (2026-10-22, after the 24 initial runs)

Pointer only; the evidence is in [README.md](README.md). The first expected pattern matched at this resolution: in every ladder and init, more training pairs never hurt, and the bands at each setting match the e2 ladder. Ladder-to-ladder spread in accuracy is no larger than the init spread. The one exception is the graded margin at 40%, where data seed 41002 is ahead on both inits. The 45% runs are still rising at 50,000 updates. Not yet audited.

## Q2 (2026-10-22, follow-up proposal, before any follow-up run)

Written after the coordinator's review of packet 1 (DECISIONS D14), before any follow-up computation.

### Why these runs

The ladders agree, so the follow-up continues intermediate runs. Two things in packet 1 are unresolved by a 50,000-update horizon:

1. **Every 45% run is still rising.** Accuracy went up 0.067 to 0.112 between updates 30,000 and 50,000. So the 50,000-update step location, and the 0.142 init difference in 41001, may be timing rather than outcome.
2. **The only ladder effect is at 40%, and it is graded.** 41002 leads on the σ-margin for both inits (mean 0.65 σ, against 0.17 σ for 41001, the lowest), while accuracy is near zero everywhere and the margins are still rising.

### Selection (all chosen before running; none on held-out results beyond those stated)

- **45%:** all six e3 runs (three ladders × two inits), continued from update 50,000 to 150,000. All are selected so that no rising or flat run is picked out.
- **40%:** the ladders with the highest and the lowest mean σ-margin at 50,000 (41002 and 41001), both inits, continued from 50,000 to 100,000. This is four runs.
- **Not continued:** 30% and 50% (one is at chance and one is near its ceiling), 41003 40% (between the two), and the 31000 runs (produced at earlier commits; they would not fit the 25-minute allowance).

### Expected patterns and what each would mean

- **45% approaches the 50% level (0.88 to 0.91) by 150,000.** The 50,000-update step location is largely a horizon effect, and any statement of location has to name its horizon.
- **45% flattens clearly below that level.** For example, a change of less than 0.03 between 100,000 and 150,000 while still below 0.8. Then at these counts the training-set size limits the accuracy reached in this horizon, not only its timing.
- **The 45% init differences shrink.** For example, 41001 goes from 0.142 to below 0.05. Then init mostly shifts timing. If the differences persist, init changes the outcome at this count.
- **At 40%, 41002 starts rising in accuracy by 100,000 while 41001 stays near zero.** The margin lead is a timing lead tied to which pairs are trained. If both stay near zero while their margins rise in parallel, the lead stays graded within this horizon. If both rise, the step keeps moving toward fewer pairs with training.

### Measurement

The same `ladder.py` measures on the same eval sets. Checkpoints are kept every 1,000 updates, and the shared definitions are unchanged.

### Failure modes

- The longer horizon is still finite.
- Two inits per ladder.
- Rising margins at 40% may need far longer than 100,000 updates to show in accuracy.
- A continuation from saved Adam moments must equal an uninterrupted run. That needs a test before it is trusted.
