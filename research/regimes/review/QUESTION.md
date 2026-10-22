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

(Counterexample refinements are recorded here, dated, before they are run. Correction, 03:10: the first four entries were labelled with clock estimates that ran 8 to 19 minutes ahead; they now carry their commit times. Each commit precedes the runs it plans, as `registry.jsonl` shows.)

**2026-10-22 02:52 (commit 40dc5ea), R4, after reading M2 (`f0f68f7`) and before computing.**
- *Revised target.* The lane's M2 reading has three parts:
  - every ρ > 1/2 passes through a shortcut phase;
  - exceptions are fit by memorization first;
  - the rule's contribution then keeps growing, so the threshold at T depends on T.
- *What the data already settle.* My pre-stated memorization test is already answered by the lane's own numbers, and I recheck them independently. At ρ ≥ 15/16 the exceptions are fit while reversed accuracy stays low. At ρ ≤ 7/8, reversed accuracy and rule flips approach 1 on unseen contexts.
- *The weaker claim.* The README says that at ρ = 15/16 "which route wins depends on the held-out context". That is the claim I attempt to break.
- *Counterexample.* Route choice may instead depend on the rule pattern (some of the 8 parity patterns already handled by the rule), or be unstructured.
- *Test (analysis only).* From each ρ = 15/16 seed's final.npz, take the 32 held-out contexts × 8 rule patterns matrix of reversed-environment correctness. Split its variance into a context main effect and a rule-pattern main effect.
- *Prediction if the lane is right:* the context share is well above the pattern share in each seed, and many contexts are all-correct or all-wrong.
- *Prediction if the counterexample holds:* the pattern share is comparable or larger.
- *Also checked.* The identity "flip_s = 1 − reversed and flip_rule = reversed whenever reliable = 1", over every logged row.

**2026-10-22 02:54 (commit 4b7d070), R2, after reading M2 (`55d6787`) and before computing.**
- *Revised target.* The M2 summary reads the frozen y1 features as giving a linear y2 readout of 0.98–1.00 at n ≥ 128, with +0.15 to +0.35 over `part` at n = 32–128. The tempting reading is that y1 pretraining builds *our* z = x1⊕x2.
- *Counterexample.* y1 = x1⊕x2⊕x3 is symmetric in x1, x2 and x3. A y1 network may therefore support the gated readout for any of the three pairwise XORs, and z is not privileged.
- *Test (analysis only, published `pretrained.npz` of the three `rel` runs).*
  - First reproduce the lane's readout exactly: fresh copy of the init `new` head, Adam lr 3e-3, 2,000 updates on D_n, unstandardized last-layer features.
  - Then fit the same readout for y2' = (x1⊕x3)∧x4 and y2'' = (x2⊕x3)∧x4.
  - Score on the test rows with x4 = 1, at n ∈ {32, 64, 128, 512}.
  - Reference: the same readouts on the untrained network.
- *Prediction if z is privileged:* the y2 readout exceeds the y2' and y2'' readouts by ≥ 0.1 at n ≥ 128 on each replicate.
- *Prediction if the counterexample holds:* all three are within about 0.05 of each other, so frozen-feature transfer says "pairwise XORs of y1's bits are linearly available", not "z".

**2026-10-22 02:58 (commit a285cfe; runs launched 02:58:21), R2 M4 (`f493c1a`), before running.**
- *Target.* NOTE.md's fine-tuning reading: `part` matches `rel` and `split` gives nothing, which "points at sharing the gate bit x4, or at fewer y2-irrelevant pretraining bits".
- *Counterexample arm `gate`.* Pretrain on g = x3⊕x4⊕x5. It shares only the gate bit x4 with y2: no x1, no x2. It has two y2-irrelevant bits, x3 and x5, which puts it between `part`/`rel` (one) and `unrel` (three).
- *Recipe.* The lane's exact train.py at `f493c1a`, loaded read-only, with g added to its label table in memory. The three lane replicates' init and data seeds are reused on purpose so the arm is paired with theirs. This is a new configuration, not a new independent seed.
- *Wrapper check.* One replay of the lane's `part` replicate 52200 must reproduce its published final.csv rows bitwise.
- *Predictions,* on fine-tuned test x4 = 1 accuracy at n = 32–64, paired against `fresh` and `part`:
  - if the gate bit drives fine-tuned transfer: `gate` − `fresh` ≥ +0.05 on all three replicates, and `gate` is within about 0.05 of `part`;
  - if the y2-irrelevant bit count drives it: `gate` lands between `part` and `unrel`, at or below `fresh`;
  - if sharing x2 is what `part` adds: `gate` is below `part` on all three replicates.
- *Budget.* 4 runs of about 10 s each.

**2026-10-22 03:01 (commit d875f95; runs launched 03:01:55), R3 M4 (`a23bdd8`), before running.**
- *Target.* NOTE obs. 2: A stays linearly decodable after B. A fresh readout on the frozen end-of-B trunk gets 0.91–1.00, against 0.49–0.79 on a B-only trunk and 0.58–0.63 untrained.
- *Counterexample.* The A-then-B trunk could be generically richer, not A-specific. After two parity tasks, its units may make any 3-bit parity linearly readable, including ones never trained.
- *Test.* The end-of-B checkpoints are local-only in the lane, so I replay the B phase from each published ckA. The runs are `r1-rev-hist-replay-{rel,unrel}-d{1000,10000}-s5320{0,1,2}-a1`, using the lane's `history.py` at `a23bdd8`, unedited.
- *Replay check.* Each replay's end-of-B snapshot metrics and readout must equal the lane's `summary.json` snapshot for the matching branch run.
- *Controls.* Then the lane's own readout recipe is fitted, same rows and budget, with only the target changed, to:
  - C1 = x7⊕x8⊕x9 (bits no task uses);
  - C2 = x0⊕x1⊕x7 (A's and Brel's pair with an unused third bit).
  Each is fitted on the untrained trunk, ckA and every end-of-B trunk.
- *Prediction if retention is A-specific:* on end-of-B trunks, A − C1 ≥ 0.25 held-out on every seed and condition, and C1 stays within about 0.1 of its untrained-trunk level.
- *Prediction if it is generic:* C1 on end-of-B trunks rises toward A's level.
- *Expectation for C2.* It may be high on related-B trunks, because both A and Brel contain x0⊕x1. That would say what is shared, not that A survived.
- *Budget.* 12 runs, about 1 min with 2 workers.

**2026-10-22 03:12, R1 M4 (`de0ef92`), before computing or running.**
- *Targets.* NOTE.md has two readings.
  - (i) The MLP "mostly answers with a seen neighbour's answer", so its above-chance held-out accuracy (`final` 0.14–0.17, `inter` 0.20–0.25) is not composition.
  - (ii) A shared transition plus intermediate states supervised through one shared readout gave near-perfect zero-shot composition (`rnn-inter` 0.96–1.00). The NOTE disclaims separating the two parts.
- *Counterexample 1 (analysis only; the R1 plan above).* Held-out success may be concentrated on rows a neighbour-copying heuristic already answers.
  - Split held-out rows by whether the true answer equals the final label of some training row that differs in exactly one op slot (N), or not (¬N). A pure neighbour-copier scores 0 on ¬N by construction.
  - Data: the logged `held_final_pred` at step 10,000 of all 15 MLP and RNN runs, scored against my oracle's labels.
  - *Prediction if `rnn-inter` composes:* its ¬N accuracy is ≥ 0.9 on every seed and within about 0.05 of its N accuracy.
  - *Prediction if the counterexample holds:* `rnn-inter`'s ¬N accuracy is well below its N accuracy (< 0.5).
  - *For the MLP and `rnn-final` I expect the opposite:* ¬N accuracy near 0 (≤ 0.05), which would support reading (i). If MLP `inter` has ¬N accuracy ≥ 0.2, reading (i) understates partial composition.
- *Counterexample 2 (runs): is the shared readout needed?* This was the coordinator's question; the NOTE names it as a next question.
  - Arm `rnn-untied`: the lane's `train.py` at `de0ef92`, loaded read-only and unedited except that h1, h2, h3 get their own readouts V1, V2, V3, each initialized as a copy of the lane's V.
  - The start is identical to the lane's `rnn-inter` of the same seed. The lane seeds 51200–51202 are reused on purpose so the arm is paired with theirs; this is a new configuration, not a new independent seed.
  - Only the tying changes. That adds 2,064 readout parameters (19,208 → 21,272); data, labels, updates and optimizer are unchanged.
  - *Wrapper checks, before training.* At init the untied logits and trunk gradients equal the lane's bitwise, with V1 + V2 + V3 gradients = the lane's V gradient. A central-difference check on a small random untied net passes.
  - *Replay check.* The tied mode on seed 51200 must reproduce the lane's published `metrics.jsonl` line for line.
  - *Prediction if the shared readout is needed:* held-out final ≤ 0.5 at 10,000 updates on all three seeds, near `rnn-final` (0.27–0.32).
  - *Prediction if the shared transition plus intermediate supervision suffices:* ≥ 0.8 on all three seeds, near `rnn-inter`.
  - In between is reported as partial. I also report whether V3 read off h2 decodes step2 (a code shared across positions arising anyway).
- *Also checked without prediction:* every NOTE number, the 73% and ~35% base rates, the interchange pair selection, and the follow-up's pre-registration order.
- *Budget.* 4 runs of about 40 s with 2 workers, about 1.5 min.

**2026-10-22 03:24, R4 M4 (`d2f154f`), before computing.**
- *Target.* NOTE follow-up item 4 and Limitations: "the slowing tracks the optimizer stalling". The evidence cited is the median √v̂ falling to about 1e-14, below Adam's eps = 1e-8, and 15–23% of training rows reaching float64 saturation.
- *Counterexample.* The displacement collapse may not need the optimizer at all. With √v̂ ≪ eps, Adam's step is about (lr/eps)·m̂, which is gradient descent with a fixed step. The collapse would then be the gradient's own decay as the smallest training margins grow, a property of the loss and data. The median coordinate may also not be the one that moves the weights. Separately, saturated rows (margin > 36.7) may carry a negligible share of the gradient however they round.
- *Test (analysis only).* Use the published Adam states (`step20000.npz`, `final.npz` at 200k) of every follow-up run. For each, compute:
  - the next Adam step;
  - the same step with eps = 0;
  - the GD-like step (lr/eps)·m̂;
  - the share of the step's squared norm carried by coordinates with √v̂ ≥ eps;
  - the share of the exact-arithmetic gradient weight Σ(1 − p_target) carried by saturated rows. 1 − p is computed as 1/(1 + e^m), which does not round to 0.
- *Predictions if the NOTE's reading holds:*
  - at 200k, ≥ 95% of the step's squared norm is on coordinates with √v̂ < eps;
  - the step is within 10% of (lr/eps)·m̂;
  - the eps = 0 step is ≥ 100× larger.
  If so, the stall is the eps regime acting on a decaying gradient, and an eps-free or normalized optimizer would not stall.
- *Prediction if the counterexample holds:* ≥ 25% of the step's squared norm is on coordinates with √v̂ ≥ eps at 200k. The collapse is then not set by eps.
- *Saturation.* I predict that saturated rows carry < 1e-4 of Σ(1 − p) at 200k, so saturation is not a material cause. ≥ 1% would make it one.
- *Also checked without prediction.* That step20000.npz equals the initial final.npz, that the a1 partials equal the a2 reruns, that the registry handles the interruption honestly, that every follow-up table number matches, and whether the figures use flip rates as independent evidence.
