# R2 shared subcomputations: question and design (M0)

Lane `research/regimes/shared/`, run IDs `r1-share-…`, seeds 52000–52999, label `r1-share-cloud-1`. Written before any main run.

## Question

When a network that has learned y1 = (x1⊕x2)⊕x3 then learns y2 = (x1⊕x2)∧x4 faster or from fewer examples than a fresh network, is that because it reuses a representation of z = x1⊕x2? Or does the benefit come from something a z-free task also gives: pretraining at all, or sensitivity to the right input bits?

## Task (Bobby's family kept, scored for its imbalance)

Inputs are ten binary bits, all 1,024 enumerated (`task.py`). y1 = x1⊕x2⊕x3 is balanced. y2 = z∧x4 is 1 on 256/1,024.

- **Imbalance.** "Always 0" scores 0.75 on y2. A network that reads only x4 (predict y2 = x4) also scores 0.75 overall, and 0.833 balanced accuracy. On the x4 = 0 half, y2 is constantly 0, so that half says nothing about z.
- **Primary y2 metric.** Held-out accuracy on the x4 = 1 half. There y2 = z exactly, the half is balanced, and chance and every z-blind heuristic score 0.5.
- **Also reported.** Overall accuracy against the 0.75 majority baseline, balanced accuracy, cross-entropy (the label-prior baseline is H(1/4) = 0.562 nats), and accuracy on the x4 = 0 half.
- **Why not change y2.** The balanced alternatives either change the function class (z⊕x4 makes the pair two 3-parities) or add a relevant bit (a multiplexer on x5). Splitting by x4 removes the imbalance problem from the metric without changing Bobby's task. The x4 = 0 half also gives a built-in selectivity test for interventions: a z intervention should leave y2 there unchanged.

## z is a hypothesis, not a property of the tasks

- y1 is symmetric in x1, x2, x3. It can be computed through x1⊕x3, x2⊕x3, or a sum-and-threshold code with no pairwise intermediate.
- A sum code (hidden units that are functions of x1+x2+x3) does not determine x1⊕x2. When x1+x2+x3 = 1, z is 1 for (1,0,0) and (0,1,0) but 0 for (0,0,1).
- y2 = (x1∧x4)⊕(x2∧x4) needs no explicit z either.
- With ReLU units, x1⊕x2 = x1+x2−2·relu(x1+x2−1), and x1∧x2 = relu(x1+x2−1). So "computing z" and "computing AND" can share a unit, which is why probes also include x1∧x2.

## Arms (one run = one arm × one paired replicate)

| arm | phase 1 (pretraining, trunk + head `pre`) | phase 2 (from phase-1 end, Adam reset) |
|---|---|---|
| `fresh` | none | trunk + head `new` on y2 |
| `rel` | y1 = x1⊕x2⊕x3 | same |
| `part` | x2⊕x3⊕x4: shares two of y2's three relevant bits, cannot contain x1⊕x2 | same |
| `unrel` | x5⊕x6⊕x7: the same function up to a bit permutation, sharing no bit with y2 | same |
| `same` | y2 itself (positive control: the pipeline can show transfer) | same |
| `joint-rel` | none | trunk + heads `pre` = y1 and `new` = y2, both labeled on the same D_n |
| `joint-unrel` | none | as joint-rel with `pre` = x5⊕x6⊕x7 |
| `alone-y1` | none | trunk + head `new` on y1, the separate-training counterpart for y1 |

`part` replaces the coordinator's x1⊕x3⊕x5 suggestion because it matches `rel` on how many of y2's relevant bits it shares (two). So `rel` versus `part` isolates whether the shared pair is {x1, x2} (the z pair) or not.

- **Frozen readout.** In every transfer arm, from the same phase-1 end state, only head `new` is trained on the frozen last-layer features of D_n. It gets the same updates and learning rate. `fresh` gives the untrained random-feature readout.
- **Row-shuffled null.** Dropped. Its expected chance score adds nothing to the chance and majority lines already drawn.

## Architecture, objective, data

- **Network.** `common/nets.py` MLP, 10 → 64 → 64 ReLU, softmax heads `pre` and `new` (2 classes each) on the last hidden layer. He/LeCun init, zero biases.
- **Pairing.** Head initializations depend only on (init seed, head name). Every arm of a replicate starts from bitwise the same weights.
- **Objective.** Mean cross-entropy per head, summed over trained heads. Full-batch Adam with lr 3×10⁻³, no weight decay, and the optimizer state reset at every phase start in every arm.
- **Split.** A data-seeded permutation of the 1,024 inputs.
  - The first 512 are the training pool. D_n = pool[:n], nested across n and shared across arms.
  - The other 512 are a fixed test set that no phase of any arm trains on. Phase-1 pretraining uses the whole pool.
  - n ∈ {8, 16, 32, 64, 128, 256, 512}.
- **Coverage.** Coverage of the eight (x1, x2, x4) patterns, and positive counts per n, go in each run's `config.json`. For example, data seed 52100 at n = 8 covers 4 of 8 patterns, at n = 16 covers 7, and at n ≥ 32 covers all 8.
- **Updates.** Phase 1: 2,000 updates on 512 rows. Phase 2 and readout: 2,000 updates on n rows. Metrics at about 30 log-spaced steps, including step 0 and the last step. The reported endpoint is the last step, never a test-selected best.
- **Replicates.** r ∈ {0, 1, 2} uses init seed 5220r and data seed 5210r, shared by all arms. Across replicates, init and data vary together; both are recorded by role. These are three exploratory paired replicates, not a power calculation.

**Exposure and parameters**

- Pretrained arms: 512 inputs × 2,000 updates of their phase-1 task, then n × 2,000 of y2.
- `fresh` and `alone-y1`: n × 2,000 of one task.
- Joint arms: n × 2,000 with two labels per input.
- Trunk: 4,864 parameters. Each head: 130.
- Joint network: 5,124 parameters. Two separate networks: 9,988.

Widths, not parameters, are equalized, so each task's own pathway has the same capacity in every arm. Comparisons are made at matched exposure:

- `rel` against `part` and `unrel`, for pretraining;
- `joint-rel` against `joint-unrel`, for extra supervision.

`rel` against `fresh` alone is not evidence of reuse.

## Plausible outcomes and what distinguishes them

Measured on held-out x4 = 1 accuracy against n and against updates:

1. **No transfer.** `rel` ≈ `fresh`.
2. **Exposure.** `rel` ≈ `part` ≈ `unrel`, all different from `fresh` (better or worse). Pretraining as such matters, not its content.
3. **Bit overlap.** `rel` ≈ `part` > `unrel`, `fresh`. Sharing relevant inputs is enough, and the z pair is not special.
4. **Related-specific transfer.** `rel` > `part` ≥ `unrel`. Consistent with z reuse, but not proof. It splits further on the mechanism checks below:
   - 4a. No selective z intervention in the transferred network: transfer through something other than our z.
   - 4b. A selective intervention: a z direction inherited from the y1 network, which when flipped flips y2 only where x4 = 1.

**What counts as "visible".** In this exploratory map, an arm difference is called visible at a given n when all three paired differences have the same sign and their mean is at least 0.05. That is a descriptive threshold, not a test.

**Prior (stated so it can be wrong).** I expect 3, or 4a, mainly at n = 16–64. The y1 network will likely use a sum-like code from which x1⊕x2 is poorly decodable, and its value for y2 would then be attention to x1 and x2 together with suppression of x5–x10. `unrel` may transfer negatively, because it must unlearn x5–x7.

## Mechanism measurements (after the first figure, on saved checkpoints)

- **Irrelevant-input dependence.** For each bit, the fraction of the 1,024 inputs whose prediction flips when that bit alone flips. Recorded for `new` at the end of every phase 2, and for `pre` after phase 1 and phase 2 (forgetting).
- **Probes.**
  - Logistic probes, fit on pool activations and scored on test, for x1⊕x2, x1⊕x3, x2⊕x3, x1⊕x4, x5⊕x6, x1∧x2, x1 and x4.
  - At both hidden layers, on the initial, pretrained and fine-tuned (n = 64 and 512) networks.
  - z decodability counts only relative to the initialization and to the same network's x1⊕x3. Decodability is not use.
- **Selective intervention.** Move each input's hidden activation along the z class-mean-difference direction to the other z class mean. Then:
  - read the y1 flips in the pretrained network;
  - read the y2 flips in the fine-tuned network, split by x4;
  - apply the pretrained network's z direction inside the fine-tuned network, as an inheritance test.
  - Controls: the x1⊕x3 and x5⊕x6 directions, and a random direction of matched norm.
  - Reported only if probes find a z direction above baseline.

## Controls and known confounders

- **Exposure, norm growth, optimizer warm state.** Controlled by `unrel` and `part` at identical phase-1 updates, and by the optimizer reset everywhere.
- **Extra supervision.** Controlled by `joint-unrel`.
- **Same-task upper reference.** `same`.
- **Random-feature readout.** `fresh` readout.
- **Confounder: test inputs in pretraining.** Avoided: phase 1 uses only the pool.
- **Confounder: small-n pattern gaps.** Coverage is recorded. A gap can make every arm fail on a pattern; that is a data limit, not a transfer effect.
- **Confounder: y1 and the pool.** y1 labels plus x3 and x4 determine y2 exactly, so a perfect y1 network "knows" y2 on paper. That is the transfer being asked about, not leakage. No y2 label from the test set is used anywhere.
- **Confounder: lr and update count.** Fixed in advance, not tuned per arm. A different lr could change which arms transfer. That is a limitation of this map.

## First figure

y2 held-out x4 = 1 accuracy, for `fresh`, `rel`, `part`, `unrel` and `same`:

- against n at the final step;
- against updates at two values of n.

Each replicate is drawn as its own trace. Training accuracy is drawn next to held-out, with chance (0.5) and majority (0.75, overall-accuracy panel) lines. Capped or failed runs are marked. A second panel row shows the same for the frozen readouts. The joint/separate comparison is a supporting figure.

## Budget and stopping rules

- **Budget.** 25 min elapsed numerical time, 1 CPU, `--workers 1`.
- **Calibration first.** The actual script, the `rel` arm, calibration seeds 952200/952100, not pooled. Timeouts are then set at about 3× the measured time, at most 900 s.
- **Planned runs.** 24: 8 arms × 3 replicates. If calibration projects more than 15 min for them, drop in this order: the n = 8 point, then `alone-y1`, then n = 256.
- **Positive-control gate.** If `fresh` at n = 512 ends below 0.95 held-out x4 = 1 accuracy, or `same` fails to transfer, stop before interpreting and report.
- **Phase-1 gate.** A pretraining run that ends below 0.99 test accuracy on its own task is kept and flagged. If two or more of the 12 do, stop and report rather than retune.
- **No silent fixes.** No lr, width or update retuning inside the initial comparison. Any change is the one adaptive follow-up, written below before it runs.

## Literature

- **Read.** arXiv:2303.11873, Merrill, Tsilivis & Shukla, "A Tale of Two Circuits: Grokking as Competition of Sparse and Dense Subnetworks": the abstract and the setup.
  - Setup: sparse (40, 3)-parity, 1,000 samples, one hidden ReLU layer of width 1,000, hinge loss, SGD lr 0.1, weight decay 0.01, batch 32.
  - Finding: a sparse subnetwork, a few neurons whose norms grow fast, takes over at generalization. It is found by pruning hidden neurons with the smallest incoming-weight norms.
- **Bearing on this design.** It motivates looking at per-neuron input weights and flip sensitivity on irrelevant bits, rather than only at accuracy. It does not study transfer between tasks.
- **Not retrieved.** Nothing else was looked up.

## Follow-up

(Empty until the first figure is pushed. Motivation and prediction go here before the run.)
