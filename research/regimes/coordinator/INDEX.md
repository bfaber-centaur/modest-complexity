# r1 regime expedition: index

Plan: [../PLAN.md](../PLAN.md). Decisions: [DECISIONS.md](DECISIONS.md). Dispatch: [DISPATCH.md](DISPATCH.md).

| regime | directory | thread | branch | M0 | M2 first figure | M4 note | review |
|---|---|---|---|---|---|---|---|
| R1 composing small programs | [composition/](../composition/) | Composing small programs | `claude/project-thread-xb3nfz` | `ff9b721` | [PR #44](https://github.com/bfaber-centaur/modest-complexity/pull/44) `85f5078`, [figure](https://github.com/bfaber-centaur/modest-complexity/blob/claude/project-thread-xb3nfz/research/regimes/composition/figures/heldout.png) | [NOTE](../composition/NOTE.md) `141b828` | M4 no BLOCK; C1.1, C1.2 fixed; **merged** `bdaa3f6` |
| R2 shared subcomputations | [shared/](../shared/) | Shared subcomputation reuse | `claude/project-thread-y589c6` | `db574da` | [PR #41](https://github.com/bfaber-centaur/modest-complexity/pull/41) `55d6787`, [figure](https://github.com/bfaber-centaur/modest-complexity/blob/claude/project-thread-y589c6/research/regimes/shared/figures/transfer.png) | [NOTE](../shared/NOTE.md) `f08e593` | M4 no BLOCK (C2.1–C2.6), fixes applied; **merged** `cede7e4` |
| R3 training history and recovery | [history/](../history/) | Forgetting and recovery | `claude/project-thread-fbrzec` | `3369031` | [PR #45](https://github.com/bfaber-centaur/modest-complexity/pull/45) `9d758e1`, [figure](https://github.com/bfaber-centaur/modest-complexity/blob/claude/project-thread-fbrzec/research/regimes/history/figures/recovery_curves.png) | [NOTE](../history/NOTE.md) `a23bdd8` | M4 no BLOCK or CONCERN; counterexample failed; **merged** `e558460` |
| R4 competing predictive routes | [routes/](../routes/) | Rule versus shortcut | `claude/project-thread-d3kg6e` | `020c86b` | [PR #43](https://github.com/bfaber-centaur/modest-complexity/pull/43) `f0f68f7`, [figure](https://github.com/bfaber-centaur/modest-complexity/blob/claude/project-thread-d3kg6e/research/regimes/routes/figures/routes_over_training.png) | [NOTE](https://github.com/bfaber-centaur/modest-complexity/blob/claude/project-thread-d3kg6e/research/regimes/routes/NOTE.md) `d2f154f` | M4 no BLOCK; C4.1, C4.3 wording fixes requested |
| independent review | [review/](../review/) | Regime expedition review | `claude/project-thread-wvait2` | `1a4f507` (shared code PASS, PR #39 merged) | M1 checks for all lanes, PR #42 merged | — | — |

Lane PRs stay open through M4 and are merged after the reviewer's M2/M4 check. Review status lives in [../review/REVIEW.md](../review/REVIEW.md).
