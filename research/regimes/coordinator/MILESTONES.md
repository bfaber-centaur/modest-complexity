# r1 lane milestones (coordinator's log)

The reviewer reads this file for what has arrived. Times are UTC on 2026-10-22. Each row gives the lane branch and commit as reported; fetch that commit to review it.

| time | lane | milestone | branch | commit | coordinator's note |
|---|---|---|---|---|---|
| 02:39 | R4 routes | M0 QUESTION.md | `claude/project-thread-d3kg6e` | `020c86b` | Split by context (32/32 of the 64 irrelevant patterns), held-out closed under s and rule flips; nested exceptions stratified by rule pattern, ρ ∈ {1, 31/32, 15/16, 7/8, 3/4, 1/2}, plus shortcut-only. No objection from the coordinator. |
| 02:41 | R3 history | M0 QUESTION.md | `claude/project-thread-fbrzec` | `3369031` | A = x0⊕x1⊕x2; related B = x0⊕x1⊕x3, unrelated B = x4⊕x5⊕x6; 256/768 split; separate heads, wd 0, the active head and trunk only trainable, Adam reset every phase; d ∈ {100, 1000, 10000}; continued-A, d = 0, fresh-A, scratch-B; conflict (notA) separate. Its design already avoids C0.1 and C0.2. Follow-up (b), optimizer carried, would hit C0.1. |
| 02:41 | review | shared code, QUESTION.md | `claude/project-thread-wvait2` | `1a4f507` | PR #39 merged into swarm/regimes (`c6b8bc7`). |
| 02:42 | R2 shared | M0 QUESTION.md, task.py, test_task.py, train.py, plan.json | `claude/project-thread-y589c6` | `db574da` | Keeps y2 = z∧x4. The primary metric is held-out accuracy on the x4 = 1 half (y2 = z there; chance 0.5). Net 10 → 64 → 64. Arms: fresh, rel, part (x2⊕x3⊕x4), unrel (x5⊕x6⊕x7), same (positive control), joint-rel, joint-unrel, alone-y1, plus frozen readouts. n = 8…512 nested; 512 pool / 512 test. Within a replicate, init and data seeds vary together (recorded by role). The host has 1 CPU. C0.1/C0.2 forwarded. |
