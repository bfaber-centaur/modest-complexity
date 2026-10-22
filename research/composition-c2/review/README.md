# c2 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked and the predictions, written before each measurement (dated Revisions).
- [REVIEW.md](REVIEW.md): verdicts with exact artifact references.
- [rev_pin.py](rev_pin.py): pin check of `b3e8132`, plus D10. Its output is [out/rev_pin.txt](out/rev_pin.txt).
- [rev_gen.py](rev_gen.py): generalization lane checks at lane commits: Part A with CX-G1, Part B with CX-G2, A5, and the follow-up anatomy with CX-G3. Its output is [out/rev_gen.txt](out/rev_gen.txt).
- [rev_time.py](rev_time.py): time lane checks at a lane commit, and CX-T1/CX-T2. Its output is [out/rev_time.txt](out/rev_time.txt).
- [rev_share.py](rev_share.py): sharing lane checks at a lane commit, CX-S1, and the wd 0 follow-up with CX-S2. Its output is [out/rev_share.txt](out/rev_share.txt).
- [rev_mech.py](rev_mech.py): mechanisms lane checks at a lane commit, and CX-M1. Its output is [out/rev_mech.txt](out/rev_mech.txt).
- [plan.json](plan.json), [registry.jsonl](registry.jsonl): reviewer runs (IDs `c2-rev-`, seeds 65000–65999). Outputs are local in `runs/c2/` (gitignored). [runs/](runs/MANIFEST.sha256) holds their small files and a manifest.

## Commands

Environment: uv 0.12.23, `uv run --locked`, OMP/OPENBLAS/MKL threads = 1.

```bash
uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/review/plan.json --root runs/c2 \
    --registry research/composition-c2/review/registry.jsonl --session-label c2-rev-cloud-1 --workers 1 \
    --ids c2-rev-verify-final-s51200-a1
uv run --locked python research/composition-c2/review/rev_pin.py runs/c2 > research/composition-c2/review/out/rev_pin.txt
uv run --locked python research/composition-c2/review/rev_gen.py 30c495f 99c83c8 543ed95 fd4e21d > research/composition-c2/review/out/rev_gen.txt
uv run --locked python research/composition-c2/review/rev_mech.py 817a768 817a768 > research/composition-c2/review/out/rev_mech.txt
uv run --locked python research/composition-c2/review/rev_share.py 83b3303 501071c > research/composition-c2/review/out/rev_share.txt
uv run --locked python research/composition-c2/review/rev_time.py 543ed95 > research/composition-c2/review/out/rev_time.txt
```

`rev_pin.py` §7 reads the run's `ckpt-10000.npz`, which is local-only.

## Budget tally (25 min of numerical execution, D6)

| when (UTC; from 04:16 on, the commit that recorded it) | item | wall | running total |
|---|---|---|---|
| 04:01 | `c2-rev-verify-final-s51200-a1` (resume 1,000 → 10,000) | 52.4 s | 0.9 min |
| 04:05 | `rev_pin.py` first pass (§0–7, includes `test_c2.py` 13 s and two one-update guard runs) | 15.6 s | 1.1 min |
| 04:08 | `rev_pin.py` with §8 (c2probe) | 21.5 s | 1.5 min |
| before 8867974 (04:20) | `rev_pin.py` with §9 (D10) and the post-pin recovery entry | 22.0 s | 1.9 min |
| before 8867974 (04:20) | `rev_gen.py` (generalization Part A): one crash on the `steps.csv` summary row, then two complete passes | 3.4 + 15.6 + 16.3 s | 2.4 min |
| before 413fddc (04:24) | `rev_mech.py` (mechanisms M1–M2, CX-M1), two passes | 13.4 + 13.5 s | 2.9 min |
| before 58b00bf (04:26) | `rev_mech.py` at `817a768` with follow-up B | 25.3 s | 3.3 min |
| before 5893f9f (04:31) | `rev_share.py` (sharing M2, CX-S1) | 0.6 s | 3.3 min |
| before 5893f9f (04:31) | `rev_gen.py` with Part B and CX-G2 | 20.4 s | 3.7 min |
| before 7ab43ed (04:34) | `rev_time.py` (time M2, CX-T1, CX-T2), two passes, plus a one-off read of the logged eps-regime steps | 2.0 + 1.7 + 1 s | 3.8 min |
| before c321113 (04:37) | A5 checks: two one-off reads (about 6 s), then `rev_gen.py` with `a5()` | 6 + 42.8 s | 4.6 min |
| after 3be6632 (04:42) | `rev_share.py` with the wd 0 follow-up: one crash (no `diag` in the 10k row), the CX-S2 pass, the NOTE-check pass, a one-off read of the table values, and a final pass after fixing two rounding tolerances in my checks | 0.7 + 0.9 + 1.2 + 1 + 1.1 s | 4.7 min |
| after 651de53 (04:51) | generalization anatomy: `anatomy()` alone, then the full `rev_gen.py` after correcting my aggregation of one NOTE check (per seed and step, not pooled) | 48.0 + 88.5 s | 7.0 min |

Times from 04:16 on were first written as clock guesses that ran 3–9 minutes ahead. They were replaced with the commit that recorded each run.
