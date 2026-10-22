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
