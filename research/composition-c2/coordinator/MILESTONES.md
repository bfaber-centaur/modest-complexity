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
| 04:12 | sharing | M1 oracle/init checks PASS; calibration 6.1/6.3/7.2/7.1 ms/update (TT/TU/UT/UU); 12 research runs launched (6 new UT/UU + 6 labelled TT/TU replays) | 0b65db5; 0.5 of 20 min |
| 04:17 | mechanisms | M1+M2, PR #52 | d7e6aee; 27 recovered checkpoints reproduce r1; interchange = recombined-program evaluation (bitwise); centroid states outperform natural states where natural fail; ~1.2 of 15 min |
| 04:19 | coordinator | mechanisms follow-up approved: candidate B (residual dose, inside/outside centroid span, held-out subspace interchange vs random subspaces), `c2-mech-residual-a1`; alternative A not run | QUESTION.md "Follow-up" |
| 04:20 | reviewer | D10 confirmed (nothing silently wrong); generalization Part A PASS on every number; G1 CONCERN (tied/inter success past depth 3 is lower on programs containing a held-out 3-op window, all seeds, 0.01–0.14), G2 figure note; time M0 notes T0.1, T0.2 | PR #53 merged (0e79200); G1/G2 relayed to generalization, T0.1/T0.2 to time |
| 04:21 | time | T0.1/T0.2 settled as QUESTION.md revisions R1 (f9fb2f9, 04:12Z) and R2 (5f48e4d, 04:21Z), before reading any continuation record; held-out final CE (all 128 rows, <10% change 40k→50k) stands in for ¬N CE, which is not logged; continuations running |
| 04:22 | generalization | Part B done: 16 runs on 4 new tasks (2 draws × 2 splits × 2 inits); tied/inter held 0.98–1.00 (¬N 0.88–1.00), tied/final 0.42–0.56 (¬N ≤ 0.06), inter > final 8/8; B3 failed 3/8 (draw 63001: all four ops send 2→1); 6.1 of 25 min | PR #51 at 99c83c8, figures/replication.png |
| 04:24 | coordinator | generalization follow-up approved: frozen first-error anatomy at steps 4–6 (F1–F3, incl. reviewer G1 window split), `c2-gen-followup-anatomy-a1` | QUESTION.md "Follow-up" at 99c83c8 |
| 04:24 | mechanisms | M3+M4: follow-up `c2-mech-residual-a1` ran as approved; NOTE.md + reviewer handoff; post-hoc descriptive `c2-mech-energy-a1` (random controls not energy-matched) logged in ADAPTIVE.md; ~1.8 of 15 min | PR #52 at 817a768 |
| 04:26 | sharing | M2: held-out at 10k TT 1.00/0.96/1.00, TU 0.63/0.52/0.52, UT 0.84/0.68/0.63, UU 0.77/0.61/0.70; UU > TU every seed (lane prediction 3 failed); TT/TU replays bitwise = r1 originals; 7.6 of 20 min | PR #54 at d8735fc |
| 04:27 | coordinator | sharing follow-up approved: matched no-decay comparison, all four cells at wd 0, seeds 51200–51202, 12 runs `c2-share-wd0-*` | QUESTION.md "Follow-up" |
