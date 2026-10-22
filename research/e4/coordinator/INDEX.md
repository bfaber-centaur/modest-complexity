# e4 review index (figures first)

**e4 is open** (2026-10-22). The question: between the post-filter accuracy peak and the collapse, what changes in the computation, and which changes cause the loss of correct answers?

Round e4 runs under the [charter](charter/README.md). Assignments, windows, measures, budgets and claims are in [DECISIONS.md](DECISIONS.md) (D1–D10). Run claims are in [../plan.json](../plan.json) and the literature correspondence is in [LITERATURE.md](LITERATURE.md). e1–e3 are complete and unchanged; e3's packet is [../../e3/coordinator/INDEX.md](../../e3/coordinator/INDEX.md).

e4 base for workers: `ee028d96560a09736ab7846d95cb7a43f80c0aaf`. Shared code is unchanged since `72cfb5d` (D3).

Figures, observations, the explanations that survived or failed, confounds, run accounting and at most three next questions will be added here as packets are reviewed and audited.

## Status

| lane | initial experiment | status |
|---|---|---|
| dynamics | D6: six constant-decay runs at wd 0.1 and 0.3 from the filtered starts | claimed |
| mechanisms | analysis of published e3 checkpoints; D7: six dense diagnostic replays | claimed |
| auditor | audit of the decomposition, the strongest cheap control, dynamics and intervention checks | dispatched |
