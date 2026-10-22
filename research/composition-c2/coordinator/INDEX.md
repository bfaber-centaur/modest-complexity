# Composition round c2: index

Bobby's packet: [charter/](charter/) (verbatim). Decisions, assignments, budgets, seeds: [DECISIONS.md](DECISIONS.md). Launch texts and threads: [DISPATCH.md](DISPATCH.md). Milestone log (reviewer feed): [MILESTONES.md](MILESTONES.md).

| item | where | status |
|---|---|---|
| integration branch | `swarm/composition-c2` from `swarm/exploration` `5c0129b` | open |
| shared code pin | [`common/`](../common/) at the pin commit below | reviewer: CONCERN C1, no BLOCK (D9) |
| recovered r1 checkpoints | [`recovered/`](../recovered/) (28 original files, hashes matched) | done (D2) |
| resume and no-change verification | [verify_resume.out](verify_resume.out), runs in [runs/](runs/) | identical (D2) |
| time lane | `time/` | M2 (PR #55) reviewer PASS on numbers, CONCERN T1 (eps regime); follow-up F1 approved and running |
| sharing lane | `sharing/` | merged (PR #54); reviewer PASS on M2 and wd-0 numbers; wd-0 decode/group tables not checked |
| generalization lane | `generalization/` | merged (PR #51); reviewer PASS on Parts A, B, A5; follow-up anatomy numbers not checked |
| mechanisms lane | `mechanisms/` | merged (PR #52); reviewer PASS on numbers, CONCERN M1 adopted in NOTE (re-check of the revised wording pending) |
| reviewer | `review/` | pin, generalization, mechanisms, sharing and time M2 reviewed ([REVIEW.md](../review/REVIEW.md)) |
| synthesis | [SYNTHESIS.md](SYNTHESIS.md) | draft; waiting on time F1 and last reviews |
| final draft PR to `swarm/exploration` | | not opened |

Pin commit: `b3e8132` (shared code `common/` as of that commit). Later coordinator commits change only `coordinator/` and add files to `recovered/` (D9) unless DECISIONS records a re-pin.
