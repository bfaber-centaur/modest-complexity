# e4 dynamics: question log

Thread "Width and training-time map" (e1 size/time, e2 and e3 dynamics). Branch `claude/project-thread-j8mgkm`. Written on 2026-10-22, before any e4 measurement and before the e4 setup is pinned. Newest entries last; earlier entries are not edited.

## Q1 (2026-10-22, before computing)

### What I knew before writing this

All of the following comes from three filtered start files (e3 mechanisms, arm B): `starts/e3-me-filt-w128-i3000{0,1,2}-a1.npz`.
- **The sources.** These are the failed 30% runs at data seed 31000, inits 30000, 30001 and 30002, trained 50,000 updates at weight decay 1.
- **The edit.** The e2 per-unit dominant-frequency filter takes them to 0.431, 0.540 and 0.580 accuracy on the 483 common pairs at update 0.
- **The continuations.** Each started with fresh Adam moments and update count, at lr 1e-3, for 20,000 updates.

I read these numbers in the e3 mechanisms README, the audit and the synthesis. I have not measured them myself.

- **Weight decay 1.**
  - Common accuracy is 0.735, 0.789 and 0.863 at update 100. It falls to 0.22–0.27 at 1,000, 0.02–0.03 at 5,000 and 0.01–0.03 at 20,000.
  - The one-update peak (0.743, 0.797, 0.872) comes 18–37 updates after train accuracy first returns to 1 (audit investigation 3).
  - A rank trace remains at 20,000: P(a+b over S) is 0.39–0.45, against 0.14–0.18 for the unedited and sham arms.
  - The network that results prefers operand labels again (P(S over N) 0.92–0.93), but it is not the original: argmax agreement with A is 0.42–0.50.
- **Weight decay 0.** The early rise is the same (0.758, 0.805, 0.872 at update 100).
  - Common accuracy is 0.69–0.81 at 1,000, 0.63–0.74 at 5,000 and 0.56–0.62 at 20,000, a slow decline of 0.20–0.25 from the peak.
  - Train cross-entropy is about 3e-8 at 20,000 and still falling.
  - The weight norm doubles, from 38–39 to 75–76.
- **The early rise.** It survives lr 1e-4 and plain gradient descent (audit probe, D13), so it comes from fitting, not from the optimizer restart.
- **e3 dynamics (my lane).** At this recipe, unedited runs at 40–50% settle into a stationary state by about 34,000–48,000 updates. Their losses and norms are stationary.
- **Varma et al. 2023, "Explaining grokking through circuit efficiency"** (arXiv 2309.02390). I read about 100k of the HTML version's 126k characters; the remainder and the PDF were not read.
  - Ungrokking: a grokked network further trained on a much smaller dataset returns to poor test accuracy.
  - They report that the transition point (the converged accuracy as a function of dataset size) does not depend on weight decay. Decay changes time scales.
  - Their efficiency explanation relies on weight decay. It treats the objective as cross-entropy plus an L2 term on the parameter norm.
  - Their setting differs from ours. It is a one-layer transformer at P = 113, trained on full data and then moved to a subset. Ours is a fixed 30% dataset and an edited network.
  - Here the efficiency account is a hypothesis to keep in view, not an assumption. AdamW's decoupled decay is not gradient descent on cross-entropy plus L2.

### Question

Hold everything else fixed: the same three filtered start files, the same data, the same evaluation pairs, Adam moments and update count reset, lr 1e-3, and 20,000 updates. What do the trajectories do at weight decay 0.1 and 0.3, between the existing 0 and 1? After the early rise, does lower decay:
- (a) mainly delay the decline;
- (b) change how far it goes within the window; or
- (c) change the trajectory qualitatively?

### Definitions, declared before measuring

- **Accuracy series.**
  - For timing: own held-out accuracy (cal ∪ audit, 662 pairs) from `metrics.jsonl`, every 10 updates. The audit found it tracks the common set within 0.028 at published checkpoints.
  - For display and endpoints: common-set accuracy (the 483 pairs of `eval-d31000.npz`, a subset of own held-out) at published checkpoints.
- **Peak.** The maximum own held-out accuracy over updates 0–1,000. Report its value and update. The early rise is update 0 to the peak.
- **Decline time T_θ.** The first logged update after the peak at which own held-out accuracy is below θ. Two thresholds are declared:
  - θ = that run's own update-0 accuracy, meaning everything training added is gone;
  - θ = 0.10 (chance is 1/31 ≈ 0.032).

  A run that has not crossed by 20,000 is censored and reported as "> 20,000", not given a time. Any later re-crossing above θ is counted and reported.
- **Endpoint.** Common and own held-out accuracy at update 20,000.
- **Decay exposure.** E(t) = −Σ log(1 − lr·wd) = −t·log(1 − lr·wd). This holds because lr and wd are constant and `experiment.py` multiplies the weights by (1 − lr·wd) each update.
  - At 20,000 updates: wd 0.1 → 2.0, wd 0.3 → 6.0, wd 1 → 20.0. At wd 0, E is 0 throughout, so wd 0 appears on update axes only.
  - E aligns only the explicit shrinkage factor. It does not equate the rest of the AdamW dynamics across conditions.
- **Alongside accuracy:** train accuracy and cross-entropy, P(a+b over N), P(a+b over S), P(S over N), the σ-margin (shared e3 definitions, D6), and the weight norm. Norms are reported, never read as the cause by correlation alone.

### Expected patterns and what each would mean

- **Delay.** After the peak, the wd 0.1, 0.3 and 1 curves roughly overlap against E, and T_θ scales roughly as 1/wd.
  - wd 1 falls through 0.10 somewhere between 1,000 and 5,000 updates (E ≈ 1–5). So wd 0.3 would cross between about 3,300 and 17,000 updates, and wd 0.1 between about 10,000 and 50,000, possibly censored.
  - Reading: the explicit shrinkage sets the clock of the decline. That fits a Varma-like account, in which decay changes when the network ends up at the memorizing solution but not whether it does. It does not prove that account.
- **Extent.** wd 0.1 or 0.3 declines and then levels off at an intermediate accuracy, and at equal E it stays clearly above wd 1.
  - Reading: decay strength changes where the trajectory goes, not only how fast. A slower approach to the same end cannot be excluded without longer runs.
- **Qualitative change.** One decay value behaves like wd 0, the other like wd 1, which would point to a threshold between regimes. Or a trajectory is non-monotone: a second rise, or a fall while train fit is lost.
  - Reading: decay changes which dynamics operate, and E alignment is the wrong description.
- **The early rise** is expected to be the same at every decay value, as it is at 0 and 1. A difference would mean decay acts within the first ~100 updates (E ≤ 0.1 at wd 1).
- **Train fit.** At wd 1 the decline happens with train accuracy at 1. If a lower-decay run loses train fit, the decline would have to be read differently.

### Limitations stated before observing

- **The window is short for wd 0.1.** By 20,000 updates wd 0.1 reaches E = 2, which wd 1 reaches at about 2,000 updates, mid-decline. wd 0.1 therefore may not separate delay from persistence inside the window. A censored wd 0.1 run means "not yet", not "never".
- **wd 0.3 is the more informative arm for the delay reading.** It reaches E = 6, where wd 1 is fully collapsed.
- **The start files are not new replicates.** They are the same three as e3, so comparisons are paired by start file, and repeated checkpoints of one run are dependent observations.
- **Comparability.** The existing wd 0 and wd 1 runs were produced at `59c822e` and `ad6e923` (shared code equal to `9041a48`). New runs at a later pin are compared with them only after their code and settings are checked equal, and after one replay check if the coordinator wants one.

## Q1 addendum (2026-10-22, after measuring the existing wd 0 and wd 1 runs, before any new run)

`decay.py` was committed at `9257c00`, with outputs in `figures/existing/`. On the existing runs it measures the following.

- **wd 1.**
  - The peak is 0.751, 0.807 and 0.878, at updates 80, 70 and 70.
  - Accuracy falls below the update-0 level at updates 600, 470 and 470 (E = 0.60, 0.47, 0.47).
  - It falls below 0.10 at updates 1,680, 1,540 and 1,800 (E = 1.68, 1.54, 1.80). There are no re-crossings.
- **wd 0.** The peak is 0.766, 0.823 and 0.881, at updates 100, 110 and 80. Both thresholds are censored (> 20,000).

The order is init 30000, 30001, 30002 throughout.

**Prediction under the delay reading.** If the explicit shrinkage sets the clock, each new run crosses at the same exposure as its wd 1 partner from the same start file.

| start | wd 0.3: T(update-0 level) | wd 0.3: T(0.10) | wd 0.1: T(update-0 level) | wd 0.1: T(0.10) |
|---|---|---|---|---|
| i30000 | about 2,000 | about 5,600 | about 6,000 | about 16,800 |
| i30001 | about 1,570 | about 5,130 | about 4,700 | about 15,400 |
| i30002 | about 1,570 | about 6,000 | about 4,700 | about 18,000 |

All of these lie inside the 20,000-update window. That makes the earlier limitation about wd 0.1 less severe than stated in Q1, provided the delay reading holds.

**Decision rule, fixed now.** For each new run and each threshold, take r = observed T divided by predicted T.

- **Consistent with the exposure clock:** 1/1.5 ≤ r ≤ 1.5 for all six runs at both thresholds.
- **Slower than exposure:** r > 1.5, or censored. Lower decay then removes less per unit of exposure; something other than the explicit shrinkage, such as the cross-entropy drive or the Adam step, sets part of the clock.
- **Faster than exposure:** r < 1/1.5.
- **Mixed:** the arms or seeds disagree. Report per run, with no pooling.

The early rise and the endpoint are read separately from r.

## Q1 outcome (2026-10-22, after the six D6 runs)

Results are in [README.md](README.md) packet 1. Under the decision rule above, the verdict is **mixed**:
- at the 0.10 threshold, all six runs follow the exposure clock (r 0.80–0.92);
- at the update-0 threshold, all three wd 0.3 runs follow the exposure clock (r 0.76–0.87), but the wd 0.1 runs disagree: i30000 is on the clock (0.77) and i30001 and i30002 are faster (0.60, 0.64).

Not covered by the rule: every r is below 1, and r is lower at wd 0.1 than at wd 0.3 for every start and threshold. All decayed runs collapse within 20,000 updates with training fit kept. The early rise does not change with decay. The exposure-matched table in the README was chosen after viewing the figure and is descriptive only. Nothing above has been edited.

## Q2 (2026-10-22, before any run): decay switched off mid-decline, an endpoint intervention

The swarm coordinator reviewed packet 1 (PR #33, merged with the audit pending) and approved this direction for the lane's one follow-up (D8). It is an **endpoint intervention**, labelled separately from the constant-wd runs (D6/D8).

### Question

Once the wd 1 decline has started, does it need decay to continue?

### Design

- **Resume states.** The published e3 wd 1 checkpoints of `e3-me-filt-w128-i3000{0,1,2}-a1` at update 300 (the start of the D5 decline window) and update 1,000 (mid-decline). Each holds the weights, the Adam moments and the update count. SHA-256, all equal to the source manifests:

  | start | update 300 | update 1,000 |
  |---|---|---|
  | i30000 | `cff32a48a326452ec89eb8e1cda6feffb4d7c039fab7e9067fdc6dd7d2ac44da` | `25d37665914b9780973b64d3e2e6ccfe8546852f149fd1f59ed697a6d2579c04` |
  | i30001 | `21473ae3e1d20090113ae8ce0254c243b598e2a8b8ed86b7813f77bb95ceae21` | `138ed4d19e366518d80b2bb01e67a20c0614211db6cdd627158b9d9b81eff00f` |
  | i30002 | `ab0ddcde018ee69ce18e10b49823cf1fe85ab4433f3a8425e9addb5d9e550519` | `107c23b54b3a9c6119215e3d95a41f977572996dd3ea8ef654af2090e3d18809` |

- **What carries over.** `--resume` carries the weights, both Adam moments and the update count, so update numbers continue the source run's. Only wd changes, from 1 to 0, starting with update s + 1.
  - The decay is decoupled. In `experiment.py` `adam()`, the weights are multiplied by (1 − lr·wd) and the moments see only the gradient, so the carried moments are equally valid at wd 0.
- **Everything else** equals the source: the same data, evaluation pairs and lr 1e-3, with the total update count 20,000 (19,700 and 19,000 new updates).
- **Comparators.**
  - The e3 wd 1 run from the same start: decay stays on, from the identical state at s.
  - The e3 wd 0 run from the same start: decay-free from update 0.
- **Diagnostic, unclaimed and unpublished** (D3 allows this). Resuming i30000's update-300 checkpoint at wd 1 for 300 updates, from the pin, reproduced all 31 source metric rows (300–600) exactly, wall-clock fields aside. The checkpoint it wrote at 300 equals the source arrays. So in the intervention runs the only difference from the wd 1 comparator is wd.

### Measure, declared now

The measure is own held-out accuracy A (662 pairs, every 10 updates), as in Q1. The window is W = 1,000 updates after the switch at s.
- Δ_off = A_off(s + W) − A_1(s). A_off(s) = A_1(s), because it is the same state.
- Δ_1 = A_1(s + W) − A_1(s).
- Δ_0 = A_0(s + W) − A_0(s).
- f = (Δ_off − Δ_0) / (Δ_1 − Δ_0). Here 0 means the decline falls to the wd 0 drift, and 1 means it continues at the wd 1 rate.

**Reference values from the existing runs**, measured now:

| start | s | A_1(s) | A_1(s+W) | Δ_1 | A_0(s) | A_0(s+W) | Δ_0 | Δ_1 − Δ_0 |
|---|---|---|---|---|---|---|---|---|
| i30000 | 300 | 0.622 | 0.169 | −0.453 | 0.751 | 0.683 | −0.068 | −0.385 |
| i30001 | 300 | 0.680 | 0.131 | −0.548 | 0.789 | 0.736 | −0.053 | −0.495 |
| i30002 | 300 | 0.737 | 0.169 | −0.568 | 0.855 | 0.802 | −0.053 | −0.515 |
| i30000 | 1,000 | 0.240 | 0.065 | −0.175 | 0.690 | 0.660 | −0.030 | −0.145 |
| i30001 | 1,000 | 0.221 | 0.057 | −0.163 | 0.745 | 0.710 | −0.035 | −0.128 |
| i30002 | 1,000 | 0.261 | 0.085 | −0.177 | 0.811 | 0.782 | −0.029 | −0.148 |

### Readings, per run, no pooling

- **(i) Decay must stay on:** f ≤ 1/3.
- **(ii) The decline sustains itself once started:** f ≥ 2/3.
- **Partial:** 1/3 < f < 2/3.
- **(iii) Recovery:** A_off at any logged update after s reaches A_1(s) + 0.05. This is an additional label and can occur together with (i).
- **Mixed:** the seeds or the two switch points disagree.

**Secondary, reported but not part of the readings:**
- A at 20,000 against the wd 0 and wd 1 endpoints;
- T(0.10) after the switch, censored at 20,000;
- train cross-entropy, weight norm, P(a+b over S) and the σ-margin at checkpoints.

### Prediction

I expect (i) at both switch points, and no recovery. The reasoning comes from packet 1.
- Decline timing followed decay exposure from wd 0.1 to 1, so the decline rate per update scales with wd.
- If the decline sustained itself once started, a lower-decay run would gain on the clock after its decline began. Its r at 0.10 would then be below its r at the update-0 level.
- In all six runs it was the other way round.

### What could make the reading fragile

- **Δ_0 comes from a different state.** It is measured on the wd 0 run, which is at a higher accuracy (0.69–0.86) than the switched states (0.22–0.74), and the drift may depend on the state. If f lands near a cutoff, this is the first reason to doubt it.
- **At s = 1,000 the reference gap is only 0.13–0.15.** Run-to-run fluctuation of about 0.01 in a single logged value moves f by about 0.07 there.
- **The design is narrow.** There is one switch time per state, and only wd 1 → 0. It does not test partial decay, switching decay back on, or other switch times.

### Runs proposed (claimed only after the coordinator lists them in the plan)

- **IDs.** `e4-dy-filt-wd1to0-u300-w128-i3000{0,1,2}-a1` and `e4-dy-filt-wd1to0-u1000-w128-i3000{0,1,2}-a1`, with purpose `endpoint-intervention`.
- **Settings.** `--resume <e3 run>/checkpoint_000{300,1000}.npz --expect-init-sha256 <above> --wd 0 --steps 20000 --lr 0.001 --data-seed 31000 --init-seed <seed> --control-seed 32000 --diagnostic-seed 33000 --log-every 10 --checkpoint-every 1000 --diagnostics none --expect-split-sha256 aa8f7504…`.
- **Checkpoints.**
  - Every 1,000, plus the resume update (written automatically).
  - Extras: `500,1300,1500,2500` for u300 and `1300,1500,2500` for u1000.
  - That is at most 5 off the every-1,000 grid, about 7 MB per run.
- **Cost.** About 50 s per run on this host. About 2.6 min elapsed on 2 workers, about 5.5 of the lane's 20 min in total.

## Q2 outcome (2026-10-22, after the six D13 runs)

Results are in [README.md](README.md) packet 2. Under the Q2 reading, all six runs read **(i) decay must stay on**, at both switch points: f is between −0.01 and 0.14. No run recovers; the maximum after the switch is below A(s) in every run. This matches the Q2 prediction.

Two things Q2 did not anticipate, reported as observations:
- For about 50 updates after the switch, the decline continues at close to the wd 1 rate, and it fades over about 200 updates. That lag accounts for most of f at s = 300.
- The off-from-1,000 runs do not drift down at all after the window, while the wd 0 run does.

Nothing above has been edited.
