# c3 correction: adaptive decisions

Dated entries, each written before the step it governs is executed. The design they adapt is [QUESTION.md](QUESTION.md).

## 2026-10-22 05:32Z: calibrated map (before any evaluation)

- **Gate.** The reviewer passed pin `be41d16` (REVIEW.md §1, PR #62), as relayed by the coordinator at 05:30Z.
- **Run.** `c3-corr-calib-s900200-a1` used only grouped fit → val prefixes, with init seed 900200, on all six bases. Base hashes were unchanged.

| k | lr | mean B val MSE |
|---|---|---|
| 16 | 1e-3 | 0.3065 |
| 16 | 1e-2 | 0.2940 |
| 32 | 1e-3 | 0.2638 |
| 32 | 1e-2 | 0.2577 |
| 64 | 1e-3 | **0.2539** (chosen) |
| 64 | 1e-2 | 0.2702 |

- **Choice.** k = 64, lr = 1e-3, by the declared rule (no other cell was within 1%). It is fixed for every arm, base and evaluation, and declared in QUESTION.md's addendum.
- **Runs added.** `c3-corr-main-a1` and `c3-corr-grouped-a1` were added to plan.json with these values.
- **Reviewer note N1.** check_source binds the per-step-readout configs by run ID only, so `load_base` keeps its own SHA-256 check against c2 `recovered/RECOVERY.json` for all six files.

## 2026-10-22 05:40Z: follow-up F2 approved; post-hoc V2 reference

- **F2.** The coordinator approved it at 05:37Z, with conditions: a fixed recipe and no recalibration, the same grouped rule with a predeclared seed, both evaluations reported, and a 5-minute budget. The full declaration and predictions are in the QUESTION.md addendum.
- **Code changes, made before any F2 result.**
  - `corr_core.bases('f2')` reads SHA-256 from each run's MANIFEST.json.
  - `corr_core.task(tid)` takes the task ID.
  - `corr_eval.py --bases f2` computes the split per task, and checks A against each run's logged step-10000 held accuracy.
  - `test_corr.py` gains the F2 verification test (8 tests).
- **Post-hoc reference (the coordinator's request; not a prediction, no fitting).** Run `c3-corr-posthoc-v2-a1` (`corr_posthoc.py`) applies the per-step-readout bases' own step-2 readout V2 to h_2 of the grouped split's fit, val and eval prefixes. V2 was trained on all 128 prefixes, all of which are base training states (D2). So this shows that the frozen h_2 is linearly read by a readout fit on every prefix. It is not a held-out decoder.

**Result (post hoc, 05:39Z).** `c3-corr-posthoc-v2-a1`: V2 reads the symbol at 1.00 on fit, val and eval prefixes of all three per-step bases. Hashes were unchanged and nothing was fit. Read it as described above, not as a held-out decoder.

## 2026-10-22 05:45Z: correction to the first-figure text

The first README and PR text said no learned arm beats A on eval-prefix held rows. That is wrong for one base: D in per-step 51201 reaches 0.625 against A's 0.562, which is 2 of 32 rows and +15% of the gain. The data and figure were always correct. The README is corrected, and the PR body is updated.

