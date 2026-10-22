# r1 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked, in what order, budget, counterexamples stated in advance.
- [REVIEW.md](REVIEW.md): findings per regime, plus shared code.
- [check_common.py](check_common.py): independent checks of `research/regimes/common/`. Output: [out/check_common.txt](out/check_common.txt).
- `oracle_{composition,shared,history,routes}.py`: per-lane oracle, split and metric checks against a named lane commit (the lane's files are read from git, never edited). Outputs in `out/`.
- [lanes.py](lanes.py): PASS/FAIL printing and loading a lane module from a commit.
- `recompute_{routes,shared,composition}.py`: key numbers recomputed from published lane artifacts, plus analysis-only counterexamples.
- [arm_shared.py](arm_shared.py), [compare_gate.py](compare_gate.py), [replay_history.py](replay_history.py), [specificity_history.py](specificity_history.py), [check_history_m4.py](check_history_m4.py), [rnn_untied.py](rnn_untied.py), [plan.json](plan.json), [registry.jsonl](registry.jsonl): the reviewer's training runs (R2 gate-only arm and a replay check; R3 B-phase replays; R1 untied-readout RNN arm and a replay check). Run outputs are local (`runs/regimes/`, gitignored); `runs/` here holds their small files with a manifest.

## Commands

Environment as in [../PLAN.md](../PLAN.md) §3 (uv 0.12.23, `uv sync --locked --group analysis`, one BLAS thread).

```bash
uv run --locked python research/regimes/review/check_common.py > research/regimes/review/out/check_common.txt
uv run --locked python research/regimes/review/oracle_composition.py ff9b721 > research/regimes/review/out/oracle_composition.txt
uv run --locked python research/regimes/review/oracle_shared.py c4939c2 > research/regimes/review/out/oracle_shared.txt
uv run --locked python research/regimes/review/oracle_history.py f71c84f > research/regimes/review/out/oracle_history.txt
uv run --locked python research/regimes/review/oracle_routes.py 4872093 > research/regimes/review/out/oracle_routes.txt
uv run --locked python research/regimes/review/recompute_routes.py f0f68f7 > research/regimes/review/out/recompute_routes.txt
uv run --locked python research/regimes/review/recompute_shared.py 55d6787 > research/regimes/review/out/recompute_shared.txt
uv run --locked python research/regimes/common/run_jobs.py --plan research/regimes/review/plan.json --root runs/regimes \
    --registry research/regimes/review/registry.jsonl --session-label r1-rev-cloud-1 --workers 2 \
    --ids r1-rev-share-replay-part-s52200-a1 r1-rev-share-gate-s52200-a1 r1-rev-share-gate-s52201-a1 r1-rev-share-gate-s52202-a1
uv run --locked python research/regimes/review/compare_gate.py f493c1a runs/regimes > research/regimes/review/out/compare_gate.txt
# R3: the 12 r1-rev-hist-replay-* IDs from plan.json through run_jobs.py as above, then
uv run --locked python research/regimes/review/specificity_history.py a23bdd8 runs/regimes > research/regimes/review/out/specificity_history.txt
uv run --locked python research/regimes/review/check_history_m4.py a23bdd8 58cb3e3 > research/regimes/review/out/check_history_m4.txt
# R1: r1-rev-comp-rnn-replay-inter-s51200-a1 and r1-rev-comp-rnn-untied-s5120{0,1,2}-a1 through run_jobs.py as above, then
uv run --locked python research/regimes/review/recompute_composition.py de0ef92 runs/regimes > research/regimes/review/out/recompute_composition.txt
# (RUN_ROOT research/regimes/review/runs gives the same output from the published files)
```

## Budget tally (20 min of numerical execution)

| date (UTC) | item | wall | running total |
|---|---|---|---|
| 2026-10-22 02:40 | check_common.py (code checks; 2 s of it is a sleep) | 3.5 s | 0.1 min |
| 2026-10-22 02:49 | four oracle scripts (enumeration, no training; R1 includes an 8! search) | < 2 s each | 0.2 min |
| 2026-10-22 02:52 | recompute_routes.py (rescoring 24 published checkpoints) | 0.7 s | 0.2 min |
| 2026-10-22 02:55 | recompute_shared.py (36 linear readouts × 2,000 updates on published features) | 17.3 s | 0.5 min |
| 2026-10-22 02:58 | runner: `r1-rev-share-gate-s5220{0,1,2}-a1`, `r1-rev-share-replay-part-s52200-a1` (2 workers) | 18.4 s | 0.8 min |
| 2026-10-22 03:02 | runner: 12 × `r1-rev-hist-replay-*` (B phase from published ckA, 2 workers) | 25.3 s | 1.2 min |
| 2026-10-22 03:03 | specificity_history.py (54 readouts × 2,000 updates on replayed trunks) | 18.6 s | 1.5 min |
| 2026-10-22 03:12 | runner: `r1-rev-comp-rnn-replay-inter-s51200-a1`, `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1` (2 workers; wrapper checks 0.15 s each) | 84.8 s | 2.9 min |
| 2026-10-22 03:16 | recompute_composition.py (rescoring and analysis, no training) | < 1 s | 2.9 min |

Host for the checks above: 4 visible CPUs, Intel Xeon 2.10 GHz, CPython 3.12.14 and NumPy from the locked environment, one BLAS thread.
