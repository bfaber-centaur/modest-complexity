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
