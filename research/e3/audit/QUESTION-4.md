# e3 audit, fourth investigation: the early-rise probe (D13) and the weight-decay-0 continuations (written 2026-10-22, about 00:50 UTC, before running anything below)

Session `e3-au-cloud-1`. Base `ba72767`. Budget used so far: about 1.1 of 30 min.

## Already known

- From investigation 3 (wd 1, Adam lr 1e-3, fresh moments), B's common-set accuracy rises in every source. At the first update with train accuracy 1 (updates 69, 45, 33) it is 0.735, 0.778 and 0.839, from 0.431, 0.540 and 0.580 at update 0. That is a gain of +0.30, +0.24 and +0.26.
- The coordinator's D12 summary of the weight-decay-0 runs: B keeps 0.56–0.62 at 20,000 updates, against 0.01–0.03 at wd 1; the early rise is the same; C stays at chance. I have not opened those runs or their figures.
- Fresh Adam's first update moves almost every coordinate by about ±lr, whatever the gradient's size.

## Probe (D13): does B's early rise need the restart's step rule?

These are diagnostic replays with my own code (`research/audit/independent.py`), from the three published B start files and the 299 training pairs of split `aa8f7504…`. They get no run IDs and nothing is published as a run.

| condition | update rule | decoupled decay per update | updates |
|---|---|---|---|
| reference | Adam, lr 1e-3, zero moments, t = 1, 2, … (investigation 3) | × (1 − 1e-3) | 300 |
| small Adam | Adam, lr 1e-4, zero moments, t = 1, 2, … | × (1 − 1e-4) | 3,000 (the same lr × updates, and the same total decay) |
| gradient descent | w ← w × (1 − 1e-3) − η g, with no moments | × (1 − 1e-3) | 3,000 |

For gradient descent, η is fixed per source from training data only. It is set so that the first update's gradient step has the same norm as the reference's first Adam step, η = ‖Δ_Adam,1‖ / ‖g_0‖.

**Decision rule.** For each condition and source, the gain is the common-set accuracy at the first update with train accuracy 1, minus its update-0 value. If train accuracy never reaches 1, the update with the highest train accuracy is used and marked "fit not reached".
- **"A property of fitting from the edited point"** if the gain is at least 0.10 in all three sources under both small Adam and gradient descent.
- **"Tied to the restart's step rule"** if the gain is below 0.05 in all three sources under both.
- **"Mixed"** otherwise, reported per condition.

The peak in each run and the common-set accuracy as a function of train accuracy are reported alongside.

## Weight-decay-0 continuations (PR #25)

**Software checks.**

| # | Check | PASS if |
|---|---|---|
| W1 | Provenance | As K1 for the six runs `e3-me-<filt|sham>-wd0-w128-i<30000|30001|30002>-a1`: registry events, 20,000 updates, start-file hash and wd 0 in `config.json`, step-0 checkpoint equal to the start file with zero moments, manifest hashes |
| W2 | Checkpoint predictions | As K2: logged accuracies exactly; cross-entropies within 1e-10 relative plus 1e-14 |
| W3 | Dense replay | As K3 with wd 0: updates 1–300 replayed independently match the published checkpoints within 1e-9 × the largest weight |

**Interpretation rules, on the common pairs.**
1. "Decay is necessary for B's erasure within 20,000 updates" is SUPPORTED LOCALLY if B at wd 0 is above B at wd 1 by at least 0.30 at 20,000 updates in all three sources.
2. "The early rise is the same at both settings" is SUPPORTED LOCALLY if the dense-replay peaks differ by at most 0.03 in each source, and their updates by at most 20.
3. C at wd 0 "stays at chance" if its common-set accuracy is at most 0.065 at every published checkpoint.
4. For B at wd 0 at 20,000 updates, I report descriptively: P(a+b over S), P(a+b over N) and P(S over N), argmax agreement with A, and weight distance from A. I also report the weight norm of A, B at wd 1 and B at wd 0. This bears on the lane's untested candidate that, under decay, the operand-label memorizer is the cheaper solution.

## Expectations, stated before computing

- I expect W1–W3 to pass, and rules 1–3 to hold as D12 describes.
- For the probe, I expect "a property of fitting from the edited point". The rise begins while train errors are being repaired, and I see no reason why sign-like steps should matter for that. I hold this loosely: gradient descent fixes the largest-gradient coordinates first, which could change which errors are repaired first.

## Failure modes

- η for gradient descent is one choice. A different step size could change the result, so a "mixed" or negative outcome would call for a second step size before any conclusion.
- Three sources, one split and one width; the evaluation pairs are shared.
