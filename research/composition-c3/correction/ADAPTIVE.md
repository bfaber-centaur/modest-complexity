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
