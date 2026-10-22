# c3 milestone log

| time (UTC) | who | milestone | where |
|---|---|---|---|
| 04:57 | coordinator | Bobby's c3 packet received and queued until c2's handoff | cmsg_01APdvkRg6kVtZy2e9JoDcdTYZspzxZdnEueRqJCsDp7Qd |
| 05:08 | coordinator | c3 shared code (c3train, c3correct, 8 tests pass); calibration 16.1–16.6 s per 2,000 updates; no-change resume of r1 ckpt-1000 equals r1 at all 41 steps | DECISIONS D3, D8; `verify_keep.json` |
| 05:18 | coordinator | c2 handed off (PR #61 draft for Bobby, head 7fe75e6); `swarm/composition-c3` created from it by merge, without merging c2; shared code pinned at be41d16 (8 tests pass on the merged head) | D1, D3 |
| 05:23 | correction | M0 QUESTION.md + ADAPTIVE.md before any result: arms A–E, main (seen states) and grouped-prefix (8/4/4 per symbol, seed 71300) evaluations, calibration on grouped fit→val only, follow-up candidates F1 depth / F2 task draw | `claude/project-thread-gqcd3n` 3e3f73d |
| 05:30 | reviewer | pin be41d16 PASS: schedule against D5 for every t < 50k, check_source accepts 27 valid and refuses 14 wrong pairings, no-change resume = c2-time-tt-inter (21 rows incl. diag), drop = c2train final-only, mid-taper resume bitwise, c3correct identity and class mean reproduce O4; notes N1–N4 (N1: per-step-readout checkpoints bound to config by run ID only, their hashes being in review/runs/MANIFEST.sha256); 1.7 of 20 min | PR #62 merged (d47d2bc) |
| 05:30 | scaffolding | M0 QUESTION.md (arms, flags, measures, 5 pp rule, six predictions, follow-up default reset-vs-carried in keep and drop) and M1 calibration 6.5–7.1 ms/update (forecast about 30 of 40 min); plan entries pass check_source dry; frozen L1–6 evaluation reproduces c2 length.csv 12/12; test_scaf.py 9/9 | `claude/project-thread-xb3nfz` b1928fe, de9ff42 |
| 05:31 | coordinator | training and fitting opened in both lanes | |
