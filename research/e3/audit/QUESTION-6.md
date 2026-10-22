# e3 audit, sixth investigation: dynamics packet 2, the continuations past 50k (written 2026-10-22, about 01:30 UTC, before computing anything below)

Session `e3-au-cloud-1`. Base `3289ef1` (packet 2 integrated as PR #30). Budget used so far: about 14.6 of 30 min; this investigation is capped at about 10 min, and anything that does not fit is NOT CHECKED.

I have read the packet-2 README (Table 4, items 1–6, provenance) and one continuation run's config and manifest fields. I have not opened its metrics, checkpoints, `continuation.csv` or `packet1_settle.csv` beyond the header row.

| # | Check | PASS if |
|---|---|---|
| P1 | Resume provenance, all ten runs | Each config's `resume` names the source run's update-50,000 checkpoint, and `resume_sha256` equals the source file's SHA-256 and the source manifest's hash. The continuation's published update-50,000 checkpoint is byte-for-byte equal to the source's in U, V, W, both moments and step. Registry: one running and one complete event. Completed steps equal the planned final count |
| P2 | Short replay after the resume | For `e3-dy-add-w128-i30001-tf40-d41002-c100k-a1` (the run that drifted in investigation 5), my independent replay of updates 50,001–50,100 from the source checkpoint with its moments reproduces the logged row at 50,100: train, cal and audit accuracy identical; cross-entropies within 1e-8 relative + 1e-14. The tolerance allows for the late rounding difference between the codes found in investigation 5 (5e-13 per update). I also report, as found, a replay of 50,001–60,000 against the published 60k checkpoint (weight distance, and eval-pair accuracy and σ-margin) |
| P3 | Packet-1 correction (item 6), from `metrics.jsonl` | My own recomputation of own held-out accuracy (cal and audit pooled, weighted by their sizes from the regenerated split) for all 24 packet-1 runs, and of the earliest logged update from 20,000 on after which it stays within 0.01 of its update-50,000 value. PASS if every run matches `packet1_settle.csv` exactly. The README's ranges (37,000–47,000 for the six 45% runs, 34,000–47,000 for the six 50% runs) follow from my values |
| P4 | Table 4 on the ladder eval pairs | `ladder_check.Ladder.score` on every published checkpoint of the ten continuations reproduces Table 4's accuracies at the printed precision. Flatness (items 1 and 3) is read from my values |
| P5 | Launch deviation (D16) | `git diff 72cfb5d 2f0aa7a` touches only `research/e3/dynamics/QUESTION.md` (no shared code, plan or lock file). All ten manifests record `producing_code_commit` `2f0aa7a`. Where the runs record their git status, it names only `research/e3/dynamics/registry.jsonl` |

**Interpretation.** Investigation 5's rule 3 ("the 45% runs are still rising at 50,000", SUPPORTED LOCALLY from the 30k and 50k checkpoints) is re-read here.
- It is CONTRADICTED if, in my P3 values, every 45% run has settled within 0.01 before 50,000, and my P4 values change by at most 0.02 in accuracy from 50k to 150k in all six 45% runs.
- Otherwise it stays as it was, with the new evidence reported.

**Expectation.** P1, P3, P4 and P5 pass. P2 passes on the 100-update window, and the 60k replay drifts in weights as in investigation 5 while the measures agree. Rule 3 becomes CONTRADICTED.
