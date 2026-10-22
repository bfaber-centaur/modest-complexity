# Composition round c2: index

Bobby's packet: [charter/](charter/) (verbatim). Decisions, assignments, budgets, seeds: [DECISIONS.md](DECISIONS.md). Launch texts and threads: [DISPATCH.md](DISPATCH.md). Milestone log (reviewer feed): [MILESTONES.md](MILESTONES.md).

| item | where | status |
|---|---|---|
| integration branch | `swarm/composition-c2` from `swarm/exploration` `5c0129b` | open |
| shared code pin | [`common/`](../common/) at the pin commit below | reviewer: CONCERN C1, no BLOCK (D9) |
| recovered r1 checkpoints | [`recovered/`](../recovered/) (27 original files, hashes matched) | done (D2) |
| resume and no-change verification | [verify_resume.out](verify_resume.out), runs in [runs/](runs/) | identical (D2) |
| time lane | `time/` | M2 (PR #55), follow-up F1 pending approval |
| sharing lane | `sharing/` | M2 (PR #54), follow-up approved |
| generalization lane | `generalization/` | M4 (PR #51 ready), awaiting Part B + NOTE review |
| mechanisms lane | `mechanisms/` | M4 (PR #52), awaiting review |
| reviewer | `review/` | pin checked (PR #50) |
| final draft PR to `swarm/exploration` | | not opened |

Pin commit: `b3e8132` (shared code `common/` as of that commit). Later coordinator commits change only `coordinator/` and add files to `recovered/` (D9) unless DECISIONS records a re-pin.
