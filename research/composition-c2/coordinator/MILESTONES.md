# c2 milestone log (UTC, 2026-10-22)

| time | lane | milestone | branch / commit / note |
|---|---|---|---|
| 03:42 | coordinator | packet received | `charter/` |
| 03:46 | coordinator | r1 checkpoints recovered from the R1 and reviewer containers; all hashes match | D2 |
| 03:52 | coordinator | resume from r1 ckpt-1000 reproduces r1 to 10k bitwise (3 runs); calibration of 4 schemes | D2, D6 |
| 03:56 | coordinator | pin `b3e8132` pushed; time and reviewer launched in the reused r1 threads | DISPATCH |
| 03:58 | coordinator | sharing, generalization, mechanisms threads started | DISPATCH |
| 04:04 | time | M0 QUESTION.md | `claude/project-thread-xb3nfz` b7d1ab6; on-host continuation check and calibration running |
| 04:05 | mechanisms | M0 QUESTION.md | `claude/project-thread-gqcd3n` 65f9245; flags that in shared-transition models whole-state interchange at a boundary equals evaluation on a recombined program of length i+3-j, so cross-length cells duplicate the generalization lane (cross-check only); non-equivalent tests are history consistency, semantic-centroid patches, random/magnitude controls |
| 04:05 | generalization | M0 QUESTION.md | `claude/project-thread-25n2g5` e5a4415; new-draw rule: r1 rule and all 84 length-1–3 products distinct; frozen Part A started |
| 04:07 | coordinator | r1 P2 (`r1-comp-interp-s51200-a1` ckpt-10000) added to `recovered/` so the generalization lane's reference line is reproducible from Git | RECOVERY.json |
| 04:06 | sharing | M0 QUESTION.md, plan.json, ADAPTIVE.md | `claude/project-thread-b72jcj` f84dbc3; 6 new UT/UU runs + 6 labelled TT/TU replays with --diag |
| 04:08 | time | M1 on-host continuation check + calibration | cd92b0d; 3 resumes s51200 ckpt-1000→3000 match all published r1 keys (41 steps); 3.7–4.1 ms/update with 2 jobs; 16 s of 40 min used; nine 40k continuations planned, waiting on pin PASS |
| 04:09 | reviewer | pin check: CONCERN C1 (resume does not check recipe), no BLOCK; training may start | PR #50, `claude/project-thread-wvait2` 39a88c4; D9 |
| 04:10 | coordinator | training opened in all lanes; C1 remedy relayed | D9 |
| 04:11 | generalization | M1+M2 frozen length 1–6 (Part A), figures/length.png, draft PR #51 | 30c495f; length 3 reproduces r1 exactly; found pin defect D10 (run a1 failed, kept); Part B calibration and draws started |
