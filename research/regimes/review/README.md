# r1 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked, in what order, budget, counterexamples stated in advance.
- [REVIEW.md](REVIEW.md): findings per regime, plus shared code.
- [check_common.py](check_common.py): independent checks of `research/regimes/common/`. Output: [out/check_common.txt](out/check_common.txt).

## Commands

Environment as in [../PLAN.md](../PLAN.md) §3 (uv 0.12.23, `uv sync --locked --group analysis`, one BLAS thread).

```bash
uv run --locked python research/regimes/review/check_common.py > research/regimes/review/out/check_common.txt
```

## Budget tally (20 min of numerical execution)

| date (UTC) | item | wall | running total |
|---|---|---|---|
| 2026-10-22 02:40 | check_common.py (code checks; 2 s of it is a sleep) | 3.5 s | 0.1 min |

Host for the checks above: 4 visible CPUs, Intel Xeon 2.10 GHz, CPython 3.12.14 and NumPy from the locked environment, one BLAS thread.
