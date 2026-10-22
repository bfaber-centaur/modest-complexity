# r1 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked, in what order, budget, counterexamples stated in advance.
- [REVIEW.md](REVIEW.md): findings per regime, plus shared code.
- [check_common.py](check_common.py): independent checks of `research/regimes/common/`. Output: [out/check_common.txt](out/check_common.txt).
- `oracle_{composition,shared,history,routes}.py`: per-lane oracle, split and metric checks against a named lane commit (the lane's files are read from git, never edited). Outputs in `out/`.
- [lanes.py](lanes.py): PASS/FAIL printing and loading a lane module from a commit.

## Commands

Environment as in [../PLAN.md](../PLAN.md) §3 (uv 0.12.23, `uv sync --locked --group analysis`, one BLAS thread).

```bash
uv run --locked python research/regimes/review/check_common.py > research/regimes/review/out/check_common.txt
uv run --locked python research/regimes/review/oracle_composition.py ff9b721 > research/regimes/review/out/oracle_composition.txt
uv run --locked python research/regimes/review/oracle_shared.py c4939c2 > research/regimes/review/out/oracle_shared.txt
uv run --locked python research/regimes/review/oracle_history.py f71c84f > research/regimes/review/out/oracle_history.txt
uv run --locked python research/regimes/review/oracle_routes.py 4872093 > research/regimes/review/out/oracle_routes.txt
```

## Budget tally (20 min of numerical execution)

| date (UTC) | item | wall | running total |
|---|---|---|---|
| 2026-10-22 02:40 | check_common.py (code checks; 2 s of it is a sleep) | 3.5 s | 0.1 min |
| 2026-10-22 03:00 | four oracle scripts (enumeration, no training; R1 includes an 8! search) | < 2 s each | 0.2 min |

Host for the checks above: 4 visible CPUs, Intel Xeon 2.10 GHz, CPython 3.12.14 and NumPy from the locked environment, one BLAS thread.
