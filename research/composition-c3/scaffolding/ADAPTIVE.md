# c3 scaffolding lane: adaptive choices

Dated entries for every choice made after [QUESTION.md](QUESTION.md) (b1928fe). None changes an arm, a constant, a threshold or a prediction.

## 2026-10-22 05:20Z, before any branch run: calibration seed and runs

- `c3-scaf-calib-{drop,taper}-s900210-a1` use calibration seed 900210 (D7 allows 900200 and up), from a fresh init.
- Each runs 2,000 updates with the switch at 500 (taper 1,000 for the taper run). These are timing runs, not research runs.
- Results: 6.5–7.1 ms per update including evaluation, two jobs. The 12 branch timeouts were set to 900 s.

## 2026-10-22 05:31–05:36Z, before any branch row was read: analysis code

These are implementation details of measures QUESTION.md already declares.

- **Retention rule tolerance.** The rule compares Δ_keep with ±0.05 using a tolerance of 1e-12. Without it, a difference of exactly 5 pp (for example 0.95 − 0.90) misses the rule through float rounding. No real measure can hit the boundary exactly: 5 pp is 6.4 of 128 rows, 2.1 of 42 and 1.6 of 32.
- **Schedule check at S + 1,999.** S + 1,999 is not a logged row, since rows are logged every 50 updates. The step weights are checked at every logged row from `metrics.jsonl`. At S − 1, S, S + 1, S + 1,998, S + 1,999 and S + 2,000 they are checked from each run's recorded schedule (`config.json`) against an independent restatement (`checks.json` → `schedule_points`).
- **Row at 10k.** For the reused arms A and D, the row at step 10,000 is c2's first continuation row rather than r1's last row. The two are equal in every r1 key (asserted), and c2's row also carries `held_subset` and `diag`.
- **Neighbour compatibility.** This is computed with the c2 time lane's own `compatibility` function, loaded read-only from `../../composition-c2/time/time_analyze.py`.
- **Workflow.** The analysis additions were written in a scratch copy and merged into `scaf_analyze.py` after the last branch had launched. Every branch records commit de9ff42, and its `git status` shows only `registry.jsonl`, which `run_jobs.py` itself appends to.

## 2026-10-22 06:00Z, after the branch results were read: threshold reading

- **Change.** The threshold is now read on each branch's trajectory from init: r1 tied/inter's rows before S, followed by the branch's rows. Before, it was read from S only, which reported a "crossing" at S = 10,000 for branches already above 0.95 there.
- **Why it is not outcome-driven.** This is the c2 time lane's convention ("a crossing before 10,000 is read from r1's record"), which QUESTION.md adopts by name. `scaf_threshold.csv` marks such crossings `before_switch` and also gives the value at S.
- **What it changes.** Nothing at S = 1k: every crossing is after S, so the result is the same either way. At S = 10k, the held-out crossings become keep's (7,700 / 9,150 / 8,200). ¬N in s51201 crosses after S under both readings: 11,300 for drop, 11,150 for taper.
