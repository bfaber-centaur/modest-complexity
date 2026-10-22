# R4 competing predictive routes: question and design (M0)

Lane `research/regimes/routes/`, run IDs `r1-route-…`, seeds 54000–54999, label `r1-route-cloud-1`. Written before any main result.

## Question

When a parity rule and a single shortcut bit both predict the training label, which one does full-batch training of a small ReLU network come to depend on, as a function of how reliable the shortcut is on the training set? And does that dependence change over the course of training?

"Depends on" means input-level dependence of the trained function on held-out inputs: how its prediction and logit margin change when the shortcut bit or a rule bit is flipped. It is not a claim about how either route is represented. Nothing here assumes which route is "simpler" for the network.

## Task

- **Inputs.** Ten bits, encoded as ±1 so neither value of any bit is a silent zero input: rule bits r1, r2, r3 at positions 0–2, the shortcut bit s at position 3, and six irrelevant bits at positions 4–9.
- **Rule label.** y = r1 XOR r2 XOR r3. It is fixed, so there is no task seed.
- **Base.** A base is the rule bits plus the irrelevant bits. There are 8 rule patterns × 64 irrelevant patterns ("contexts"), so 512 bases. An input is a base plus a value of s, so the complete domain has 1,024 inputs.
- **Split by context** (data seed 54100). 32 of the 64 contexts are training contexts and 32 are held-out. Training bases are all 8 rule patterns crossed with the training contexts, which gives 256 bases. Held-out bases are the other 256.
  - No input and no base appears in both splits, under either value of s. Every held-out context is new.
  - The held-out set is closed under flipping s and under flipping any rule bit, so those interventions never touch a training input.
  - Every context occurs with all 8 rule patterns in training, so each irrelevant bit is exactly independent of y on the training set.
- **Shortcut reliability ρ** (exception seed 54101). Draw one uniform u per training base. Within each rule pattern (32 training bases), the k bases with the lowest u are exceptions with s = 1 − y, and the rest have s = y.
  - k ∈ {0, 1, 2, 4, 8, 16} gives ρ = 1 − k/32 ∈ {1, 0.969, 0.938, 0.875, 0.75, 0.5}, realized exactly.
  - Exceptions are nested across ρ: every exception at a given k is also an exception at every larger k.
  - Labels and s are each exactly balanced (128/128), and the exceptions are spread evenly over the rule patterns, so no rule pattern predicts exception status.
- **Pairing.** Every condition uses the same 256 training bases, the same labels and the same three initialization seeds. Only s on the exception bases differs. The held-out set and the evaluation environments are identical in every run.

## Conditions

| condition | training target | training s | role |
|---|---|---|---|
| ρ = 1 | y | s = y everywhere | both routes perfectly predictive |
| ρ = 31/32, 15/16, 7/8, 3/4 | y | k exceptions per rule pattern | the sweep |
| ρ = 1/2, **rule-only** | y | s independent of y (16 exceptions per pattern) | the rule is the only predictive route |
| **shortcut-only** | s | the ρ = 1/2 inputs | the label is s and is independent of the rule bits |

- **Rule-only.** This means s is present, varying and uninformative. It does not mean s is held constant, which would leave the s weights untouched by gradient.
- **Shortcut-only.** It uses exactly the ρ = 1/2 training inputs with the labels replaced by s. It is the trainability reference for the shortcut route, and its held-out signature (below) is what pure shortcut use looks like.

## Model and objective

- **Network.** `common/nets.py`, 10 → 128 ReLU → one two-class softmax head, 1,666 parameters. He/LeCun initialization with zero biases, float64.
- **Objective.** Mean cross-entropy over the 256 training inputs.
- **Optimizer.** Full-batch Adam, lr 1e-3, no weight decay, default betas.
- **Length.** A fixed number of updates T for every run, with no early stopping. T is set from the positive controls before the main runs: 20,000 unless the rule-only control has not converged by then (see stopping rules).
- **Exposure.** Every condition sees 256 examples per update, so T × 256 example-gradients, the same parameter count and the same initialization. Nothing differs in the amount of supervision.

## Evaluation (every logged step, roughly log-spaced, step 0 included)

One forward pass over the complete 1,024-input domain gives all of the following.

**Accuracy against y**
- **Train**, overall and separately on the conforming and exception bases. Shortcut-only is scored against its own target s.
- **Held-out reliable**: s = y on every held-out base (256 inputs).
- **Held-out reversed**: s = 1 − y (256 inputs).
- **Held-out independent**: both values of s (512 inputs). This is exactly the mean of reliable and reversed, and it is reported for readability.

**Route dependence**, on held-out inputs under both values of s
- **Shortcut flip rate**: the fraction of the 256 pairs whose prediction changes when s alone flips.
- **Rule flip rate**: the same for flipping one rule bit, which flips y, averaged over the three rule bits.
- **Irrelevant flip rate**: the same for one irrelevant bit, averaged over the six bits. Only pairs whose endpoints are both held-out are used, because flipping a context bit can leave the held-out set.
- **Logit-level versions.** With margin m = z₁ − z₀, the shortcut signed delta is the mean of m(s=1) − m(s=0). The rule signed delta is the mean of m(y=1 member) − m(y=0 member). Each sign is positive when the function moves the way that route predicts. The mean absolute delta is also logged for all three kinds of flip.

The reference signatures are the following.
- **Pure rule:** reliable 1, reversed 1, rule flip 1, shortcut flip 0.
- **Pure shortcut:** reliable 1, reversed 0, independent 0.5, rule flip 0, shortcut flip 1.
- **Shortcut plus memorized exceptions** (the "third route"): train accuracy on the exception bases near 1, while held-out reversed accuracy stays near 0. Held-out contexts are new, so memorized exceptions cannot transfer.

## Plausible outcomes and what would distinguish them

1. **Shortcut-dominated at every ρ < 1 within T.** Reversed accuracy stays near 0 and shortcut flip near 1 for every ρ above 1/2. The exceptions are fit by memorization: exception-train accuracy reaches 1 with no rise in reversed accuracy.
2. **A reliability threshold.** Below some ρ*, final reversed accuracy and rule flip are near 1 and shortcut flip is near 0. Above it, the reverse holds. The final values against ρ show a step, consistent across seeds.
3. **Early shortcut, later rule.** Within a run, shortcut flip rises first and then falls, while rule flip and reversed accuracy rise later. This is visible in the trajectories, and it would make any "threshold" depend on T.
4. **A mixture.** Both routes contribute at logit level, with both signed deltas positive and of comparable size. Predictions then follow whichever is larger in each context, so reversed accuracy is intermediate and the two flip rates sum to more than 1, or depend on the context.

**Prior, stated so it can be wrong.** At initialization, s has a direct linear correlation with y, while 3-bit parity has none with any single bit or pair. That suggests the shortcut is picked up first whenever ρ > 1/2. I expect outcome 1 at ρ = 1 and at 31/32, the rule at 1/2, and outcome 3 somewhere in between. I have no confident guess for where. This is an argument about the gradient signal at initialization, not about simplicity.

## Controls and known confounders

- **Controls.** Rule-only (ρ = 1/2) and shortcut-only are the trainability references, and each is a positive control for its route. The irrelevant flip rate is the baseline sensitivity to a bit that carries no signal.
- **Rule-flip side effects.** Flipping a rule bit flips y, but a network could react to a single-bit change without computing parity. The signed rule delta and the reversed-environment accuracy are the direct parity checks, and the irrelevant flip rate is the reference.
- **Off-distribution environments.** At ρ = 1 the network never sees s ≠ y, so the reversed and independent environments are off its training distribution. That is the point of them, but read them as stress tests.
- **Training length.** Route dependence may still be moving at T. The figure shows whole trajectories, and claims are stated "at T".
- **One data draw.** Every run shares one split and one exception draw, and the three seeds vary initialization only. Dataset randomness is recorded but not sampled.
- **Recipe-specific.** One architecture, width, encoding, learning rate and optimizer, with no weight decay. Weight decay is a known lever on which solution is reached and is deliberately left out of the initial comparison.

## Validation (M1, before main runs)

- **Independent oracle and split check** (`test_routes.py`). It rebuilds the domain with `itertools` and pure-Python parity, then checks:
  - labels;
  - disjointness of bases and inputs across the split;
  - closure of the held-out set under s and rule flips;
  - realized ρ, label balance and s balance per condition;
  - nesting of exceptions;
  - that conditions differ only in column s on exception rows;
  - that shortcut-only is independent of y;
  - the flip-pair construction.
- **Finite-difference gradient check** of the actual 10 → 128 → 2 configuration on ±1 inputs.
- **Positive controls** with a separate calibration initialization seed, 54300, not pooled with the principal runs: rule-only and shortcut-only at T = 20,000, run two at a time. They also serve as the host calibration of the actual script (wall time, CPU time, peak RSS, versions, threads).

## First figure

- **Rows.** One per condition: ρ = 1 … 1/2, then shortcut-only.
- **Columns**, each over training updates on a log axis with step 0 drawn separately:
  - train accuracy (overall and exceptions);
  - held-out accuracy (reliable, independent, reversed), with chance at 0.5;
  - route dependence (shortcut, rule and irrelevant flip rates).
- **Seeds.** The three paired seeds are individual thin traces. Failed and capped runs are marked.
- **Supporting figure.** The value at T against ρ for each seed: reversed accuracy, exception-train accuracy, and the two signed logit deltas.

## Budget and stopping rules

- **Budget.** 25 min of elapsed numerical execution in total.
  - About 1.5 min for calibration and positive controls.
  - About 6–10 min for the initial comparison: 7 conditions × 3 seeds = 21 runs at T = 20,000, two at a time.
  - Up to about 8 min for one follow-up, and the rest held back.
  - The tally is kept in README.md. I stop and ask the coordinator before passing 25 min.
- **T rule.** If the rule-only positive control has not reached held-out independent accuracy ≥ 0.95 by 20,000 updates, T is raised once, to at most 80,000, before any main run. That change is recorded here. Nothing about the recipe is changed after the main runs start.
- **Timeouts.** About 3× the calibrated time, at most 900 s. Capped and failed runs are kept and shown.
- **Follow-up.** One adaptive follow-up, chosen after the first figure: longer training or width, in whichever ρ range the figure shows to be unsettled. Its motivation and prediction go under "Follow-up" below and are pushed before it runs.

## Follow-up (written after M2 at f0f68f7, before any follow-up run)

**Motivation.** The first figure shows that route dependence is still moving at T = 20,000 for ρ = 31/32, 15/16 and 7/8.
- **Rule growth comes late.** Every ρ > 1/2 fits its training set by 366–603 updates. The rule's contribution grows mostly after that, so the reliability "threshold" at T depends on T.
- **What the follow-up tests.** Whether the rule keeps taking over with longer training, at the reliabilities where it had not yet won, and whether ρ = 1 stays pure shortcut.

**A confounder found before running.** The step-20,000 checkpoints of seed 54200 (ρ = 1, 31/32, 15/16, 7/8, 1/2) show the following.
- **Gradients have fallen below Adam's denominator.**
  - The median current |gradient| is 1e-11 to 2e-10.
  - The median √v̂ is 1.5–2.8e-8, close to Adam's eps = 1e-8.
  - So late Adam steps are already much smaller than lr. Gradients decay faster than v̂ forgets them (β2 = 0.999), and eps adds to the denominator.
- **The float64 limit is near.** The largest training margin is 36.2, just below the point (about 36.7) where float64 rounds the target probability to exactly 1 and the target-logit gradient becomes exactly 0. No training row is saturated yet.
- **Consequence.** Beyond T, a flattening curve could come from the optimizer and arithmetic slowing down rather than from a settled preference.

**Design**
- **Runs.** The same recipe and seeds, with 10× longer training (200,000 updates), for ρ = 1, 31/32 and 15/16 × init seeds 54200–54202. That is 9 runs, IDs `r1-route-long-<cond>-s<seed>-a1`, with 150 log points.
- **Read-only diagnostics** added to `train.py` and logged at each step:
  - the parameter norm;
  - the net parameter displacement per update since the previous log step;
  - the minimum and median training margin;
  - the fraction of training rows with float64-saturated target probability;
  - the median √v̂.
- **Bitwise check.** Each run saves `step20000.npz`, which must equal the initial run's `final.npz` parameters bitwise. That checks that the added logging did not change training.
- **Cost.** About 140 s per run, so about 10.5 min at two at a time. The lane total would then be about 13.5 of 25 min.

**Predictions** (stated before running)
1. **ρ = 15/16.** The rule share Δrule / (|Δs| + |Δrule|) keeps rising past 0.5 (it was 0.40–0.48 at T). Reversed accuracy crosses 0.5 for all three seeds before 200,000 updates, unless the numerics stall first (prediction 4).
2. **ρ = 31/32.** The rule share keeps rising from 0.10–0.16, but reversed accuracy stays below 0.5 at 200,000. Low confidence.
3. **ρ = 1.** The rule share stays below 0.05, because without exceptions no training example's loss is reduced more by the rule than by s.
4. **Numerics.** The displacement per update falls by at least 10× between 20,000 and 200,000. The saturated fraction becomes non-zero, so the curves flatten on the log axis.
   - Any plateau after the displacement has collapsed is reported as optimizer- and precision-limited, not as a settled route preference.
   - Distinguishing observation: a rule share that still rises while the displacement per update is still near its value at T supports continued takeover. A rule share that freezes as the displacement collapses is uninformative about the asymptote.

**Separation.** The follow-up's results go in a separate figure (`figures/followup_long_*`) and a separate NOTE.md section. They are never merged into the initial-comparison figure.
