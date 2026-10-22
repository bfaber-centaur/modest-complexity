# Composition round c2: index

Bobby's packet: [charter/](charter/) (verbatim). Decisions, assignments, budgets, seeds: [DECISIONS.md](DECISIONS.md). Launch texts and threads: [DISPATCH.md](DISPATCH.md). Milestone log (reviewer feed): [MILESTONES.md](MILESTONES.md).

| item | where | status |
|---|---|---|
| integration branch | `swarm/composition-c2` from `swarm/exploration` `5c0129b` | open |
| shared code pin | [`common/`](../common/) at the pin commit below | reviewer: CONCERN C1, no BLOCK (D9) |
| recovered r1 checkpoints | [`recovered/`](../recovered/) (28 original files, hashes matched) | done (D2) |
| resume and no-change verification | [verify_resume.out](verify_resume.out), runs in [runs/](runs/) | identical (D2) |
| time lane | `time/` | merged (PR #55); reviewer PASS on M2, F1 and NOTE fixes; T1 (eps regime) and T2 (100k rise length-3-specific) carried in NOTE |
| sharing lane | `sharing/` | merged (PR #54); reviewer PASS on M2 and wd-0 numbers; wd-0 decode/group tables not checked |
| generalization lane | `generalization/` | merged (PR #51); reviewer PASS on Parts A, B, A5 and the follow-up anatomy |
| mechanisms lane | `mechanisms/` | merged (PR #52); reviewer PASS on numbers and on the revised NOTE (M1 adopted) |
| reviewer | `review/` | every lane and the synthesis reviewed ([REVIEW.md](../review/REVIEW.md)); PRs #50, #53, #56–#60 merged; 7.9 of 25 min |
| synthesis | [SYNTHESIS.md](SYNTHESIS.md) | final for handoff; reviewer PASS (§6) |
| final draft PR to `swarm/exploration` | see MILESTONES | opened as a draft for Bobby; not merged by the coordinator |

Pin commit: `b3e8132` (shared code `common/` as of that commit). Later coordinator commits change only `coordinator/` and add files to `recovered/` (D9) unless DECISIONS records a re-pin.
