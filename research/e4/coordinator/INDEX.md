# e4 review index (figures first)

**e4 is open** (2026-10-22). The question: between the post-filter accuracy peak and the collapse, what changes in the computation, and which changes cause the loss of correct answers?

Round e4 runs under the [charter](charter/README.md). Assignments, windows, measures, budgets and claims are in [DECISIONS.md](DECISIONS.md) (D1–D10). Run claims are in [../plan.json](../plan.json) and the literature correspondence is in [LITERATURE.md](LITERATURE.md). e1–e3 are complete and unchanged; e3's packet is [../../e3/coordinator/INDEX.md](../../e3/coordinator/INDEX.md).

e4 base for workers: `ee028d96560a09736ab7846d95cb7a43f80c0aaf`. Shared code is unchanged since `72cfb5d` (D3).

Figures, observations, the explanations that survived or failed, confounds, run accounting and at most three next questions will be added here as packets are reviewed and audited.

## Status

| lane | initial experiment | status |
|---|---|---|
| dynamics | D6: six constant-decay runs at wd 0.1 and 0.3 from the filtered starts | packet 1 integrated (PR #33), audit pending; follow-up direction approved (decay off from a mid-decline state) |
| mechanisms | analysis of published e3 checkpoints; D7: six dense diagnostic replays | claimed |
| auditor | audit of the decomposition, the strongest cheap control (random-feature starts, D11–D12), dynamics and intervention checks | control runs activated |

## 1. Decay strength between the peak and the collapse (dynamics, packet 1, not yet audited)

![exposure axis](../dynamics/figures/decay_exposure.png)

Six new runs at wd 0.1 and 0.3 from the three filtered start files, with the e3 wd 0 and wd 1 runs as endpoints, on the decay-exposure axis E = −t·log(1 − lr·wd). The update-axis version is [decay_update.png](../dynamics/figures/decay_update.png), and the packet is [../dynamics/README.md](../dynamics/README.md).

- **Observation.**
  - Fit times (70/50/40) and peaks (within 0.004) do not depend on decay.
  - Every decayed run collapses within 20,000 updates while training accuracy stays at 1. Common accuracy at 20k is 0.06–0.09 at wd 0.1, 0.03–0.05 at wd 0.3 and 0.01–0.03 at wd 1.
  - wd 0 is censored at 0.56–0.62.
  - The global weight norm grows through the decline at every decay value.
- **Pre-stated rule: MIXED.**
  - At the 0.10 threshold, all six runs are on the exposure clock (r 0.80–0.92).
  - At the update-0 threshold, wd 0.3 is on the clock but two of the three wd 0.1 runs are faster.
  - Outside the rule: all twelve ratios are below 1, and they are lower at lower decay.
- **Explanation (lane, untested).** Within 20k, decay strength mainly sets when the decline happens. A decay-independent drift (the slow wd 0 decline) may add to it.
