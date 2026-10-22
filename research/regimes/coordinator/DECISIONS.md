# r1 decision log

Plan: [../PLAN.md](../PLAN.md). Prompts: [prompts/](prompts/). Launch texts: [DISPATCH.md](DISPATCH.md). Newest entries come last. Earlier entries are not edited, only superseded by later ones.

## D1 (2026-10-22) Open r1 as a separate expedition

Bobby asked for four regime investigations and an independent reviewer, separate from the running e4 round ("more agents!").
- **Integration branch.** `swarm/regimes` was created from `swarm/exploration` `7c4389c`, the e4 head at setup.
- **What stays untouched.** e4's threads, `research/e4/` and the top-level shared files. Nothing is pushed to `swarm/exploration`, either `main` or upstream.
- **Workers.** Five fresh project threads. None of the e1–e4 worker threads is reused.

## D2 (2026-10-22) Lanes own their plans; IDs and seeds are partitioned

Each lane owns `research/regimes/<lane>/plan.json`. There is no central claim step, because claims through the coordinator were a bottleneck in e1–e4.
- **ID collisions.** Prevented by the per-lane prefix (`r1-comp-`, `r1-share-`, `r1-hist-`, `r1-route-`, `r1-rev-`, `r1-co-`).
- **Seed collisions.** Prevented by disjoint ranges (PLAN §1).
- **Order.** A plan entry must be pushed before its launch. The reviewer can verify this from commit order.

## D3 (2026-10-22) One small shared network instead of four

All four regimes need the same thing: a small classifier on binary or one-hot inputs, multiple heads or tasks, frozen features, optimizer resets at phase boundaries, and checkpoints that carry optimizer state. `research/regimes/common/nets.py` is one tested implementation of that, so the reviewer checks one set of gradients instead of four.
- **No task logic.** It holds none, and lanes may write their own model if the regime needs one.
- **Runner.** `run_jobs.py` generalizes `launch_runs.py`'s registry to arbitrary lane scripts, and imports its `read_events`. It also sets two glibc malloc variables (PLAN §3).
- **Tests.** `test_nets.py` covers 8 cases. The finite-difference test moves biases off zero, because binary inputs with zero biases put the all-zero row exactly on the ReLU kink. Lanes using binary inputs should know this when they check their own gradients.

## D4 (2026-10-22) Calibration and budgets

- **Calibration** is in PLAN §4. Attempt a1 ran before a backward-pass copy fix and the malloc settings and was about 1.6–2× slower; a2 ran after both. Both attempts are kept.
- **Coordinator compute.** About 7 of its 12 minutes went on calibration: a1 took 2 × 161 s, a2 took 2 × 40 s.
- **Budgets.** 25 min for each investigator and 20 min for the reviewer, as elapsed numerical execution.
- **Unit cost.** Runs at regime sizes cost 10–35 s per 10,000 full-batch updates, so the budgets are far above the initial comparisons. They are caps, not targets.

## D5 (2026-10-22) Milestones, follow-ups, merging

- **Milestones** are M0–M4 (PLAN §5).
- **Follow-up.** A lane may run its one adaptive follow-up without waiting for the coordinator, once M2 is pushed and the follow-up's motivation and prediction are committed.
- **Merging.** The coordinator merges lane PRs into `swarm/regimes` under Bobby's standing approval to integrate worker PRs at its own judgment. It merges after review, or with "review pending" in INDEX.
- **Final PR.** The expedition's final PR for Bobby will be from `swarm/regimes` and will not be merged by the coordinator.

## D6 (2026-10-22) Reviewer's shared-code concerns: no mid-flight change to `common/`

The reviewer passed `common/` at `0c7bd19` (PR #39, merged as `c6b8bc7`) with three usage concerns.
- **C0.3, relative `--root` outside the repository root.** It is real but unreachable with PLAN §3's commands, which run from the root. Changing `common/` now would move every lane's base, so it is not fixed in r1. The rule instead: run `run_jobs.py` from the repository root.
- **C0.1, a head that is trainable but absent from targets.** Lanes must exclude a head from `trainable`, or reset the optimizer with wd = 0, before claiming that the head is untouched.
- **C0.2, unfreeze without reset.** Lanes must call `adam_state` at any unfreeze and say so in the config.

These two are forwarded to R2 (frozen features) and R3 (phases, readouts). R3's M0 design (`3369031`) already satisfies both for its initial comparison.

## D7 (2026-10-22) C0.1 and C0.2 go through the reviewer, not lane messages

This supersedes D6's "forwarded to R2 and R3". The reviewer checks C0.1 and C0.2 per lane (REVIEW.md §1–§4), so no separate lane message was sent. Both initial designs already avoid the patterns:
- R2 (`db574da`): reset at every phase start; frozen readouts train only `new`; no unfreeze.
- R3 (`3369031`): reset at every phase; the inactive head is excluded from `trainable`; wd 0.

All four M0 designs had arrived by 02:42 UTC ([MILESTONES.md](MILESTONES.md)). None needed a coordinator change. Recipes differ across lanes on purpose, each lane's own choice:
- R1 uses wd 0.1; R2, R3 and R4 use wd 0.
- R2 uses lr 3e-3.
Cross-regime comparisons of training speed are therefore not meaningful.

## D8 (2026-10-22) Close r1

All four lanes reached M4 with one pre-registered follow-up each. Each was merged after the reviewer's final review and the lane's wording fixes:
- R2: PR #41 (`cede7e4`);
- R3: PR #45 (`e558460`);
- R1: PR #44 (`bdaa3f6`);
- R4: PR #43 (`f607b2a`).

The reviewer's branch is merged up to `aed5003`. No BLOCK was raised in any regime.

**Compute (elapsed numerical execution).**

| session | used |
|---|---|
| R1 | 4.8 of 25 min |
| R2 | 4.1 of 25 min |
| R3 | 4.9 of 25 min |
| R4 | 14.4 of 25 min |
| reviewer | 2.9 of 20 min |
| coordinator | about 7 of 12 min |

**Runs.** 180 lane and reviewer runs in all: 178 complete, 2 failed (R4, container restart, rerun as a2), none capped.

**Deliverable.** [SYNTHESIS.md](SYNTHESIS.md). No second adaptive run was launched in any lane. Where a lane named one, it appears as a recommended next experiment and was not run.
