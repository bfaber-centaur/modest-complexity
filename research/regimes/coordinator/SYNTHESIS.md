# r1 regime expedition: synthesis

Coordinator's account of the four investigations. Every number below is taken from a lane's NOTE.md or from the reviewer's REVIEW.md, and those cite the committed run files. This page adds no new measurements. Plan: [../PLAN.md](../PLAN.md). Decisions: [DECISIONS.md](DECISIONS.md). Milestone log: [MILESTONES.md](MILESTONES.md).

## Index

| regime | question | primary figure | note | runs (registry) | lane CPU | review |
|---|---|---|---|---|---|---|
| R1 composing small programs | When does learning primitives support unseen compositions? | [heldout.png](../composition/figures/heldout.png) | [NOTE](../composition/NOTE.md) | 20, all complete | 4.8 of 25 min | no BLOCK; C1.1, C1.2 fixed |
| R2 shared subcomputations | Does y1 → y2 transfer come from reusing z = x1⊕x2? | [transfer.png](../shared/figures/transfer.png) | [NOTE](../shared/NOTE.md) | 29, all complete | 4.1 of 25 min | no BLOCK; C2.1–C2.6, NOTE fixed |
| R3 training history and recovery | What of A survives B, and what sets recovery? | [recovery_curves.png](../history/figures/recovery_curves.png) | [NOTE](../history/NOTE.md) | 75, all complete | 4.9 of 25 min | no BLOCK or CONCERN |
| R4 competing predictive routes | Which of rule and shortcut does training select, and does it change? | [routes_over_training.png](../routes/figures/routes_over_training.png) | [NOTE](../routes/NOTE.md) | 36: 34 complete, 2 failed (container restart, rerun as a2) | 14.4 of 25 min | no BLOCK; C4.1, C4.3 fixed |
| independent review | All seven checklist items, per regime | — | [REVIEW.md](../review/REVIEW.md) | 20, all complete | 2.9 of 20 min | — |

**Coordinator compute.** About 7 of 12 min, all of it calibration (DECISIONS D4).

Each lane ran:
- its initial comparison on three paired initialization seeds;
- exactly one adaptive follow-up, whose prediction was committed before its first run (REVIEW.md checks the order for R1, R3 and R4. For R2, the coordinator checked it: commit `1d9d240` at 02:52:09 precedes the first follow-up launch at 02:52:17. These are commit times, not push times).

No lane hit its budget, and no run was capped. R4's two failed runs were killed by a container restart; they are kept, and their reruns match them bitwise up to the kill. Each lane used **one** task draw and **one** split, so dataset randomness was recorded but not sampled.

The recipes differ on purpose (DECISIONS D7):
- R1: wd 0.1;
- R2: lr 3e-3;
- all four: full-batch Adam, small ReLU networks.

Timing therefore does not compare across regimes.

## Side by side: what each experiment did and did not establish

### R1, composing small programs

- **Setup.** 4 random permutations of 8 symbols; 48 of 64 length-3 programs trained, 16 held out by a Latin square. The tested single-op/drop-one heuristics and proper-subset lookups score ≤ 0.125; this is not a bound on all non-compositional strategies.
- **Initial comparison.** A 20→128→128 MLP fits every training row by about 250 updates. Held-out final accuracy at 10k updates:

  | condition | held-out final accuracy |
  |---|---|
  | final-only | 0.14–0.17 |
  | intermediate-state supervision | 0.20–0.25 (paired +0.03 to +0.11) |
  | shuffled-intermediate control | 0.10–0.13 |

- **What the errors look like.** With intermediate supervision the step heads are right on held-out rows (step2 0.98–1.00) while the final answer is not. Every held-out prefix also occurs in training, so this shows only that an unseen op 3 leaves the step heads intact; it is not generalization of the intermediate computation (reviewer C1.1). 97–100% of wrong answers are a one-slot training neighbour's label.
- **Follow-up.** A shared-transition RNN with one shared readout. With intermediate supervision it reaches 1.00, 0.96 and 1.00, rising slowly after the fit. Final-only reaches 0.27–0.32, still rising at 10k.

**Established**
- In this MLP and recipe, neither supervision gives composition within 10k updates. This holds although the final-answer tables alone fix all four permutations; the reviewer checked all 8! assignments.
- A network that applies one transition across positions, with intermediate states supervised through one shared readout, composes zero-shot.
- Linear probes decode step2 in the MLP (0.55–0.99 depending on layer and condition). But swapping the probe subspace yields the composed counterfactual in only 1–16% of pairs, against ≤ 1% for random subspaces. The shuffled-label control does 8–10% at hidden 2, so much of that effect does not need true intermediate labels (C1.2).
- **The shared readout carries much of the RNN's effect.** The reviewer's untied-readout arm, with a shared transition and per-step readouts, reaches 0.52–0.63 held-out, paired with 0.96–1.00 for tied, and is still rising at 10k.

**Not established**
- Whether the untied and final-only RNNs are slower or different: both were still rising at 10k.
- Any asymptote: three conditions were still rising at 10k.
- That the decoded state is the network's route to its answer.
- Anything beyond one permutation draw and one split.

### R2, shared subcomputations

- **Setup.** y2 = z∧x4, scored on the x4 = 1 half, where y2 = z and every z-blind rule scores 0.5. Pretraining arms (2,000 updates on a 512-input pool):
  - rel (y1);
  - part (x2⊕x3⊕x4);
  - unrel (x5⊕x6⊕x7);
  - same (y2 itself);
  - the follow-up split, (x1⊕x3⊕x5, x2⊕x6⊕x7).

  Each was then trained on y2 for 2,000 updates on nested n = 8–512, by fine-tuning or by a frozen-feature readout.
- **Frozen readout.** y1 features give a linear y2 readout of 0.98–1.00 at n ≥ 128. They beat part and split by 0.14–0.38 on every replicate at n = 32–128.
- **Fine-tuning.**
  - part ≈ rel.
  - split ≈ fresh.
  - unrel transfers negatively (−0.09 to −0.10 at n = 64).
  - The reviewer's gate-only arm, x3⊕x4⊕x5, also shows no transfer (C2.5).

**Established**
- Frozen-feature transfer is specific to y1 relative to matched-exposure controls.
- y1 pretraining makes every pairwise XOR of x1, x2, x3 linearly available, and z is not privileged among them. Readouts of (x1⊕x3)∧x4 and (x2⊕x3)∧x4 score like y2 (C2.4).
- Fine-tuned transfer is not z-specific, and it is not explained by the gate bit alone, by the shared pair alone, or by the irrelevant-bit count alone.

**Not established**
- That the network computes or uses z as its intermediate. Mean-shift steering could not test this, and the note says why.
- What drives fine-tuned transfer. The remaining candidate is "two of y2's three bits inside one pretraining parity", which is untested.
- How much of the fine-tuning result is phase-2 disturbance: at n ≤ 32 fine-tuning degrades even the same-task control (C2.3).
- Small-n results: pattern coverage differs by replicate there (C2.1).

### R3, training history and recovery

- **Setup.**
  - Tasks: A = x0⊕x1⊕x2; related B = x0⊕x1⊕x3; unrelated B = x4⊕x5⊕x6.
  - Network: a 10→128 trunk with separate heads, wd 0, and the optimizer reset at every switch.
  - Schedule: B branches of d = 100, 1,000 or 10,000 updates from one hashed A checkpoint per seed. The reviewer replayed all 12 B phases exactly.
- **Forgetting.** Once B is learned (d ≥ 1,000), held-out A through the bitwise-unchanged A head falls to 0.69–0.86. Continued-A keeps ≥ 0.995 despite substantial trunk drift.
- **Frozen readout.** On the end-of-B trunk it recovers A at 0.91–1.00. On a B-only trunk it gets 0.49–0.79. The reviewer's never-trained-parity counterexample reads 0.50–0.63.
- **Relearning cost.**

  | starting point | updates to relearn A |
  |---|---|
  | kept A head (initial comparison) | 37–560 |
  | reset head on the end-of-B trunk (follow-up) | 190–410 |
  | reset head on a B-only trunk | 460–1,100 |
  | fresh A | 540–700 |

**Established**
- Loss of A through its own head coexists with A-specific, linearly decodable structure in the trunk.
- Savings on return are carried partly by the trunk, not only by the preserved head. The kept head was faster than the reset head in 10 of 12 pairs.
- Forgetting tracks B's acquisition, not the size of the drift.

**Not established**
- Which trunk directions carry A through B.
- Whether related and unrelated B differ: there was no consistent order at three seeds.
- Anything beyond one split, 3-bit-parity tasks and one recipe.

### R4, competing predictive routes

- **Setup.** A 3-bit parity rule against a shortcut bit with training reliability ρ.
  - Exceptions are nested and paired across ρ.
  - The split is by irrelevant context, so no base crosses it.
  - Network: 10-128-2, full-batch Adam, T = 20k updates.
- **Initial comparison.**
  - At every ρ > 1/2 the shortcut comes first: its flip rate reaches 0.95 by 22–41 updates.
  - The training set, exceptions included, is fit by 366–603 updates.
  - The rule then takes over after the fit: reversed-environment accuracy crosses 0.5 at 546–737 updates for ρ = 3/4 and at 2,989–4,036 for 7/8.

  | ρ | outcome at T = 20k |
  |---|---|
  | 15/16 | mixed (reversed accuracy 0.25–0.42); which route wins varies by individual held-out base, more with its context than its rule pattern (reviewer C4.1) |
  | 31/32 and 1 | shortcut |

- **Follow-up.** 10× longer training moves 15/16 only to an even split (0.40–0.63; 1 of 3 seeds crosses 0.5). By then the displacement per update has fallen about 150×, and √v̂ is 1e-14 to 2e-13, far below Adam's eps of 1e-8. The reviewer confirmed that at 200k no coordinate has √v̂ ≥ eps, so the update is approximately (lr/eps)·m̂: a momentum-based update whose magnitude falls with the gradient. Float64 saturation is immaterial (C4.3).

**Established**
- Under this recipe the shortcut is acquired first at every ρ > 1/2.
- Which route the network ends on is decided after the training set is fit. That happens in a late phase where Adam's steps shrink by orders of magnitude, ending in an epsilon-dominated, momentum-based regime with shrinking updates.
- No reliability threshold exists independent of training length.

**Not established**
- The asymptotic route preference.
- Whether the late takeover belongs to the data and loss or to Adam's late-phase dynamics.
- How either route is represented: all measures are input interventions.
- Dependence on the data draw. Rule-only keeps a small positive shortcut dependence that probably comes from the draw (obs. 6).

## Across regimes (descriptive; each is a pattern in this expedition, not a general claim)

1. **Decodable and used came apart wherever both were measured.**
   - R1: step2 is decodable, but the interchange follows the counterfactual in ≤ 16% of pairs.
   - R2: z is decodable at 1.00, but it is not privileged among its sibling XORs, and steering could not test use.
   - R3: A is decodable at 0.91–1.00 from a trunk whose own A head reads 0.69–0.86.

   In each case a linear readout found structure that the network's own output did not, or could not be shown to, rely on.
2. **The generalizing solution can arrive after the training set is fit.**
   - R1's rnn-inter fit by about 500 updates. Its held-out accuracy was 0.34–0.36 at 1k and 0.96–1.00 at 10k.
   - R4's rule takeover happens after the fit, in a phase where the optimizer's steps shrink by orders of magnitude.

   Under these recipes, "trained to fit" and "trained to convergence of the function" are different stopping points.
3. **Unrelated prior training was not neutral; it slowed new learning in both regimes that measured it.**
   - R2: unrelated pretraining gave −0.09 to −0.10 under fine-tuning, and its frozen features read out below random features.
   - R3: a trunk trained only on unrelated B slowed A (740–1,100 against 540–700 updates). Learning unrelated B after A was slower than from scratch (590–1,400 against 370–480).

## Recommended next experiments (at most three, each justified by an observed result)

1. **R4: is the late rule takeover a property of the data or of Adam's late phase?**
   - *Why.* The 10× follow-up showed that the route choice at ρ = 15/16 and 7/8 is made after the fit, while Adam's displacement per update collapses about 150× and √v̂ falls far below eps.
   - *Design.* Branch ρ = 15/16 and 7/8 from the identical step-20k checkpoints under two changes:
     - (a) an optimizer that does not stall at tiny gradients: normalized GD, or Adam with eps far below the gradient scale;
     - (b) small weight decay.
   - *Reading.* Further growth under (a) would show that the original endpoint depended on the update rule. A plateau under another optimizer would still be a finite-budget observation, not a demonstration of what the data support or an asymptotic preference.
   - *Cost.* A few CPU minutes.
2. **R1: are the partial RNN arms slower or different?**
   - *Why.* The reviewer's untied-readout arm (0.52–0.63) and rnn-final (0.27–0.32) both sit between the MLP and rnn-inter and were still rising at 10k. rnn-inter itself rose from 0.34 to about 1.0 after the fit.
   - *Design.* Continue both arms with an exact resume from their 10k states to about 50k updates. The lane checks the resume bitwise against an uninterrupted short trajectory first.
   - *Reading.* If they reach rnn-inter's level, those components were not necessary for high accuracy on this evaluation; compare time-to-accuracy under the stated recipes. If they plateau within the budget, report a persistent finite-time gap, not necessity or impossibility.
   - *Cost.* A few CPU minutes.
3. **R3 (with R2 as a second instance if cheap): ablate the decodable subspace and measure the downstream behaviour.**
   - *Why.* Across-regimes observation 1. In R3 the end-of-B trunk carries A-specific linearly decodable structure and gives trunk-borne savings, but neither has been shown to cause the other.
   - *Design.* Project the A-readout subspace (fit on the frozen end-of-B trunk) out of the trunk's output. Then relearn A with a reset head.
   - *Reading.*
     - If the savings over a B-only trunk vanish while appropriate matched ablations preserve them, that supports a causal contribution from the removed directions. Check disruption of other information and off-distribution effects before identifying a specific reused computation.
     - If savings survive, the tested readout direction is not necessary under this intervention; redundant coding and relearning remain alternatives.
     - The R2 analogue: ablate the z direction against the x1⊕x3 direction from frozen y1 features, and refit the y2 readout.
   - *Cost.* Under a minute of CPU on published checkpoints.

## Branch and integration

- **Integration branch.** `swarm/regimes`, from `swarm/exploration` at `7c4389c`.
- **What was merged.** Lane PRs #41 (R2), #45 (R3), #44 (R1) and #43 (R4), each after the reviewer's final review and the lane's wording fixes, plus the reviewer's branch. Each lane changed only its own directory. `common/` is unchanged since `0c7bd19`, and its tests pass on the integrated head.
- **What was left alone.** `swarm/exploration`, e4, both main branches and upstream.
- **Size.** The four lane directories hold about 29 MB in Git, and the review holds 2.7 MB. Most of it is figures and per-step CSVs. Published checkpoints come to about 5.6 MB across the lanes, and the largest single file is 1.7 MB.

## Integration review note

PR #49 review tightened the scope of baseline, optimizer, and proposed-intervention claims; no measurements changed. Run the five lane/common unittest files in separate Python processes: the research scripts use local modules named `task` and `train`, which collide when suites are loaded together. This is a test-discovery limitation, not a failed numerical check in the isolated experiment entry points. RNN checkpoints from r1 were not published; follow-ups must recover and verify them or explicitly replay the original runs before claiming exact continuation.
