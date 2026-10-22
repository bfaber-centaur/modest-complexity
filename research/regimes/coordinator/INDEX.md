# r1 regime expedition: index

Plan: [../PLAN.md](../PLAN.md). Decisions: [DECISIONS.md](DECISIONS.md). Dispatch: [DISPATCH.md](DISPATCH.md).

| regime | directory | thread | branch | M0 | M2 first figure | M4 note | review |
|---|---|---|---|---|---|---|---|
| R1 composing small programs | [composition/](../composition/) | Composing small programs | `claude/project-thread-xb3nfz` | `ff9b721` | — | — | M1 PASS |
| R2 shared subcomputations | [shared/](../shared/) | Shared subcomputation reuse | `claude/project-thread-y589c6` | `db574da` | [PR #41](https://github.com/bfaber-centaur/modest-complexity/pull/41) `55d6787`, [figure](https://github.com/bfaber-centaur/modest-complexity/blob/claude/project-thread-y589c6/research/regimes/shared/figures/transfer.png) | — | M1 PASS; C2.1, C2.2; M2 pending |
| R3 training history and recovery | [history/](../history/) | Forgetting and recovery | `claude/project-thread-fbrzec` | `3369031` | — | — | M1 PASS |
| R4 competing predictive routes | [routes/](../routes/) | Rule versus shortcut | `claude/project-thread-d3kg6e` | `020c86b` | [PR #43](https://github.com/bfaber-centaur/modest-complexity/pull/43) `f0f68f7`, [figure](https://github.com/bfaber-centaur/modest-complexity/blob/claude/project-thread-d3kg6e/research/regimes/routes/figures/routes_over_training.png) | — | M1 PASS; M2 pending |
| independent review | [review/](../review/) | Regime expedition review | `claude/project-thread-wvait2` | `1a4f507` (shared code PASS, PR #39 merged) | M1 checks for all lanes, PR #42 merged | — | — |

Lane PRs stay open through M4 and are merged after the reviewer's M2/M4 check. Review status lives in [../review/REVIEW.md](../review/REVIEW.md).
