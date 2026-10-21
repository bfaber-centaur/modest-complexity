# e2 audit, cycle 2: question (written 2026-10-21, about 19:53 UTC, before any of the computations below)

Session `e2-au-cloud-1`, base `8b11c2c` (swarm/exploration after PR #16). Target: mechanisms packet 1 (`research/e2/mechanisms/`, PR #15, merged and not yet audited) and the extra onset checkpoints of DECISIONS D9.

## Already seen when this was written

- **Packet 1.** I have read its README, INTERPRETATION, QUESTION, `groups.py` and figure, but not the rows of `groups.csv`. Its table and observations say:
  - In the 30% runs, frequency groups explain 0.012–0.017 of their held-out contribution by a function of d = a+b−c, against 0.004–0.005 for matched random sets.
  - All 45 of their own-frequency terms have phase within ±π/2 of the correct label, with median amplitude 0.12–0.15 logits.
  - At step 0, 34 of 75 groups are within ±π/2, with amplitude about 0.0004.
  - In the 50% runs, removing a group costs about as much as removing a random set at equal logit change.
- **My cycle 1.** The restricted rank is null for the no-decay (`add0`) and shuffled-label (`shuf`) width-128 runs, and the shuffled labels equal a+b on 4% of pairs.
- **Not yet seen.** I have computed nothing per unit or per group for any run, nothing for the `add0` or `shuf` runs beyond the restricted rank, and nothing on the D9 checkpoints.

## Question

The packet's own strongest alternative says the aligned terms in the 30% runs are a side effect of fitting training labels equal to a+b, not the start of a computation that generalizes. Is the alignment produced by fitting a+b labels, or does it need what separates the decayed runs?

## Three explanations

A. **Fitting a+b labels suffices.** The `add0` runs train on the same pairs with the same labels and fit them (train accuracy 1). Under A their groups also carry own-frequency terms aligned with a+b on the common pairs, at amplitudes like the decayed 30% runs.

B. **Alignment needs weight decay**, the condition under which the restricted rank appears. Under B the `add0` groups sit at the step-0 level: about half within ±π/2, and amplitudes far below the decayed runs.

C. **Readout artifact.** Grouping by dominant frequency plus the d-function makes any trained network look aligned. Under C the `shuf` groups (same pairs, labels not a+b) are also aligned with a+b.

## Diagnostic

1. Reimplement the group readout independently from its written definition:
   - live units;
   - dominant non-DC input frequency from U and V;
   - the group's contribution, centred over labels for each pair;
   - the within-d mean on the 483 common pairs;
   - the phase and amplitude of its own-frequency cosine.
2. Check the reimplementation against `groups.csv` on the six addition runs at 50k.
3. Apply it to the three `add0` and three `shuf` width-128 runs at 50k and at step 0.
4. Decision rule, fixed now. For each condition, count groups with |phase| < π/2 against 50% (step 0: 34/75), and compare median amplitude with step 0 and with the decayed 30% runs.

## Also checked (software, not interpretation)

- Whether the random sets are matched as claimed:
  - removing hidden units cannot change other units' gates;
  - the count and norm match, and how many of the 16 draws were found per group;
  - group ablations against the random-set trend at equal logit change, quantified rather than read off the scatter;
  - the logit change measured on the common pairs only, not all 961.
- Whether grouping uses only the analysed checkpoint.
- The 1k–10k share separation of the restricted rank, redone with the D9 checkpoints: three seeds per share at 1k spacing. The cycle-1 figure had two 30% seeds at 1k, 2k and 5k only.

## Failure modes

- In `add0` the units may be weakly concentrated, so its groups are less well defined than in the decayed runs. Amplitude is the readout that does not depend on how concentrated a group is.
- The shuffled labels equal a+b on 4% of training pairs, a small pull towards alignment.
- Groups within one run share the same logits, so the 45 phases per condition are not independent observations. Counts are reported per seed.
- A sharper version of the packet's proposed decisive test, projecting the logits onto the a+b−c-structured part and asking whether accuracy appears, is decided by construction. A function of d alone whose largest value is at d = 0 predicts a+b for every pair. That test is not run.
