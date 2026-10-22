# R4 competing predictive routes: note (M4)

Lane `research/regimes/routes/`. Design and predictions: [QUESTION.md](QUESTION.md), pushed before any result. Commands, calibration, budget and manifest: [README.md](README.md). Every number below comes from a committed run file; run IDs are `r1-route-<condition>-s<seed>-a1` (initial comparison) and `r1-route-long-<condition>-s<seed>-a1` (follow-up; `-a2` for ρ = 1 and 31/32 at seed 54202, rerun after a container restart killed a1), and ranges are over the three paired init seeds 54200–54202.

## Question

When a 3-bit parity rule and a single shortcut bit s both predict the training label, which does full-batch training of a small ReLU network come to depend on, as a function of the shortcut's training reliability ρ, and does that dependence change during training? "Depends on" is input-level: how held-out predictions and logit margins move when s or a rule bit is flipped. It is not a claim about representation, and nothing assumes either route is "simpler".

## Design

- **Task.** Inputs are ten ±1 bits: rule bits 0–2, s at 3, irrelevant bits 4–9. y = parity of bits 0–2.
- **Split.** 32 of the 64 irrelevant contexts are training contexts (256 training bases), and the other 32 give 256 held-out bases. No input or base crosses the split, and the held-out set is closed under s and rule-bit flips.
- **Reliability.** Exceptions (s = 1 − y) are k per rule pattern out of 32, from one uniform draw, so they nest across ρ = 1 − k/32 ∈ {1, 31/32, 15/16, 7/8, 3/4, 1/2}. ρ = 1/2 is the rule-only control. Shortcut-only uses the ρ = 1/2 inputs with label s. Every condition shares training bases, labels, held-out set and init seeds; only s on the exception bases differs.
- **Model.** `common/nets.py` 10-128-2 ReLU network, full-batch Adam, lr 1e-3, no weight decay, T = 20,000 updates. Every run sees 256 examples per update.
- **Measures**, on the 512 held-out inputs at about 100 log-spaced steps: accuracy in the reliable (s = y), independent and reversed (s = 1 − y) environments; prediction flip rates for flipping s, one rule bit, or one irrelevant bit; and the signed logit-margin change for flipping s (positive = toward the shortcut) or a rule bit (positive = toward parity).
- **Checks.** `test_routes.py` rebuilds the domain independently and checks labels, split, closure, realized ρ, balance, nesting, pairing, flip pairs, the metrics on oracle functions, and finite-difference gradients. Rule-only and shortcut-only positive controls (init seed 54300) each reach their route's signature.

## Observations: initial comparison

Primary figure: [figures/routes_over_training.png](figures/routes_over_training.png). Values at T: [figures/routes_at_T.png](figures/routes_at_T.png). Both have CSVs beside them.

| condition | held-out reversed acc at T | signed Δs at T | signed Δrule at T | rule share at T |
|---|---|---|---|---|
| ρ = 1 | 0.00 | 38.2–39.7 | 0.02–0.13 | 0.00 |
| ρ = 31/32 | 0.03–0.06 | 37.1–39.4 | 4.5–7.2 | 0.10–0.16 |
| ρ = 15/16 | 0.25–0.42 | 22.8–27.3 | 18.2–20.9 | 0.40–0.48 |
| ρ = 7/8 | 0.95–0.98 | 11.8–12.6 | 27.5–27.9 | 0.69–0.70 |
| ρ = 3/4 | 1.00 | 4.7–5.5 | 31.9–32.6 | 0.86–0.87 |
| ρ = 1/2 (rule-only) | 1.00 | 0.50–0.74 | 32.7–33.7 | 0.98 |
| shortcut-only | 0.00 | 37.9–39.6 | −0.02 to 0.03 | 0.00 |

Rule share = Δrule / (|Δs| + |Δrule|). Every run reaches train accuracy 1.0 and held-out reliable accuracy 1.0.

1. **Shortcut first, at every ρ > 1/2.** Shortcut flip rate reaches 0.95 at 22–41 updates, with reliable accuracy 1 and reversed accuracy near 0. That is the shortcut-only signature, reached on the same timescale as the shortcut-only control.
2. **The training set is fit before the rule wins.** Training accuracy, exceptions included, reaches 1.0 at 366–603 updates. At that moment reversed accuracy is 0.00 (31/32), 0.04–0.06 (15/16), 0.11–0.12 (7/8) and 0.30–0.33 (3/4).
3. **The rule takes over after the fit, slowly.** Reversed accuracy crosses 0.5 at 546–737 updates for ρ = 3/4 (train loss 1–2·10⁻²) and at 2,989–3,651 updates for ρ = 7/8 (train loss 1–3·10⁻⁴). It has not crossed by T at 15/16 or 31/32, where Δrule is still rising.
4. **Outcome by the categories set in advance.** At T, ρ = 1 matches outcome 1 (shortcut only). ρ = 31/32 is shortcut-dominated with memorized exceptions at the prediction level, but has a small positive rule contribution at the logit level. ρ = 15/16 is a context-dependent mixture (outcome 4): both signed changes are large, reversed accuracy is intermediate, and the irrelevant-bit flip rate is 0.11–0.14, so which route wins depends on the held-out context. ρ = 7/8 and 3/4 match outcome 3 (early shortcut, later rule). There is no clean reliability threshold independent of T.
5. **The three seeds agree closely.** Across seeds the trajectories differ mostly in timing by a few tens of updates; the largest spread at T is reversed accuracy 0.25–0.42 at ρ = 15/16.
6. **Rule-only keeps a small positive Δs (0.50–0.74) at T in all three seeds**, about 2% of Δrule, although s is exactly uncorrelated with y in its training set. The sign is shared across init seeds, so it probably comes from the single data draw. In that draw, products of s with an irrelevant bit do correlate with y (|corr| up to 0.14), which a network can exploit on training contexts. This explanation is untested.

## Follow-up: 10× longer training (adaptive, separate)

Motivation and predictions were pushed before running (QUESTION.md "Follow-up", commit 5710c35). Figure: [figures/followup_long.png](figures/followup_long.png). The runs are ρ = 1, 31/32 and 15/16, with the same seeds and recipe, at 200,000 updates. Each run's step-20,000 checkpoint is bitwise equal to the initial run's final state.

| condition | reversed acc, 20k → 200k | rule share, 20k → 200k | displacement per update, 20k → 200k | saturated train rows at 200k |
|---|---|---|---|---|
| ρ = 1 | 0.00 → 0.00 | ≤ 0.003 → ≤ 0.003 | 1.1–1.3e-4 → 7.1–7.6e-7 | 0.00 |
| ρ = 31/32 | 0.03–0.06 → 0.04–0.08 | 0.10–0.16 → 0.12–0.19 | 2.7–3.1e-4 → 1.7–1.8e-6 | 0.17–0.20 |
| ρ = 15/16 | 0.25–0.42 → 0.40–0.63 | 0.40–0.48 → 0.46–0.54 | 2.7–3.1e-4 → 1.9e-6 | 0.15–0.23 |

Against the predictions:
1. **ρ = 15/16: partly wrong.** The rule share kept rising and the rule gained, but reversed accuracy crossed 0.5 for only one seed (54200, at about 28k updates, 0.63 at 200k). The other two reached 0.40 and 0.44.
2. **ρ = 31/32: as predicted.** The rule share rose slightly, and reversed accuracy stayed below 0.1.
3. **ρ = 1: as predicted.** The rule share stayed ≤ 0.003; there is no rule at the logit level without exceptions.
4. **Numerics: as predicted, and decisive.**
   - The parameter displacement per update fell about 10× between 20k and about 39k updates, and about 150× by 200k.
   - The median √v̂ fell from about 2e-8 at 20k to about 1e-14, so Adam's eps dominates its denominator.
   - 15–23% of training rows at ρ < 1 reached float64 saturation, where the target-logit gradient is exactly zero.
   - At ρ = 15/16, most of the post-T rule-share gain came before about 39k: +0.04–0.05, against +0.01–0.02 over the next 160k updates. Reversed accuracy gained 0.07–0.12 and then 0.03–0.09.
   - After that the curves only creep, while the displacement per update falls another 15×. The slowing tracks the optimizer stalling; it is not evidence of a settled preference.

**What this adds.** Ten times more training moves the picture only slightly: ρ = 15/16 drifts from slightly shortcut-leaning to evenly split, and the ends do not move. With this recipe, which route training ends on is mostly decided in the first roughly 40k updates, after which the optimizer barely moves. The asymptotic preference under continued margin growth is not observable here. The follow-up therefore cannot say whether ρ = 15/16 or 31/32 would eventually go to the rule.

## Limitations

- **One data draw.** All runs share one split and one exception draw; the three seeds vary initialization only. Observation 6 is a reminder that the draw can leave a signature.
- **One recipe.** One width, depth, encoding, learning rate and optimizer, with no weight decay and full batches. Weight decay and minibatch noise are known levers on which solution is reached and were not varied.
- **Prediction flip rates are not independent measures here.** Whenever held-out reliable accuracy is 1, shortcut flip = 1 − reversed accuracy and rule flip = reversed accuracy exactly. The signed logit changes carry the extra information.
- **Input interventions only.** The flips show what the function depends on near held-out inputs, not how either route is represented or computed. A rising Δrule is consistent with a parity computation but is not proof of one.
- **Off-distribution environments.** At ρ = 1 the network never sees s ≠ y, so the reversed environment is a stress test outside its training distribution.
- **Late training is numerically delicate.** By T, the median current gradient (about 1e-10) is far below the median √v̂ (1.5–2.8·10⁻⁸) and Adam's eps (10⁻⁸), and the largest training margins approach the float64 saturation point. Late changes are slowed by the optimizer, not only by the loss landscape. The follow-up shows that past about 40k updates training has nearly stopped.

## Next question

Wherever the rule wins, it wins after the training set is already fit, while the training loss is small and shrinking. So the "which route" answer here may belong to the late-phase optimizer as much as to the data.

**Next question.** Is the post-fit takeover a property of the loss and data, or of Adam's late-phase behavior?

**Cheapest test.** Rerun ρ = 15/16 and 7/8 from the identical step-20,000 checkpoints, branched explicitly, under two changes:
- an optimizer that does not stall at tiny gradients, such as normalized gradient descent, or Adam with eps far below the gradient scale and a stated precision guard;
- small weight decay, which changes what the late phase favors.

**What would distinguish them.** If the rule share keeps growing under the first change, the T-dependent threshold reflects optimizer slowdown. If it stalls anyway, it reflects the solution the data supports.

Two other questions are left open:
- whether a second exception draw moves the ρ = 15/16 mixture (the data seed was not varied);
- whether the context-dependent mixture at 15/16 comes from separate units for the two routes or from one shared readout. That one needs representation-level interventions this lane did not attempt.
