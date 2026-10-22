# c2 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked and the predictions, written before each measurement (dated Revisions).
- [REVIEW.md](REVIEW.md): verdicts with exact artifact references.
- [rev_pin.py](rev_pin.py): pin check of `b3e8132`, plus D10. Its output is [out/rev_pin.txt](out/rev_pin.txt).
- [rev_gen.py](rev_gen.py): generalization lane checks at a lane commit, and CX-G1. Its output is [out/rev_gen.txt](out/rev_gen.txt).
- [rev_mech.py](rev_mech.py): mechanisms lane checks at a lane commit, and CX-M1. Its output is [out/rev_mech.txt](out/rev_mech.txt).
- [plan.json](plan.json), [registry.jsonl](registry.jsonl): reviewer runs (IDs `c2-rev-`, seeds 65000–65999). Outputs are local in `runs/c2/` (gitignored). [runs/](runs/MANIFEST.sha256) holds their small files and a manifest.

## Commands

Environment: uv 0.12.23, `uv run --locked`, OMP/OPENBLAS/MKL threads = 1.

```bash
uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/review/plan.json --root runs/c2 \
    --registry research/composition-c2/review/registry.jsonl --session-label c2-rev-cloud-1 --workers 1 \
    --ids c2-rev-verify-final-s51200-a1
uv run --locked python research/composition-c2/review/rev_pin.py runs/c2 > research/composition-c2/review/out/rev_pin.txt
uv run --locked python research/composition-c2/review/rev_gen.py 30c495f > research/composition-c2/review/out/rev_gen.txt
uv run --locked python research/composition-c2/review/rev_mech.py 817a768 817a768 > research/composition-c2/review/out/rev_mech.txt
```

`rev_pin.py` §7 reads the run's `ckpt-10000.npz`, which is local-only.

## Budget tally (25 min of numerical execution, D6)

| time (UTC) | item | wall | running total |
|---|---|---|---|
| 04:01 | `c2-rev-verify-final-s51200-a1` (resume 1,000 → 10,000) | 52.4 s | 0.9 min |
| 04:05 | `rev_pin.py` first pass (§0–7, includes `test_c2.py` 13 s and two one-update guard runs) | 15.6 s | 1.1 min |
| 04:08 | `rev_pin.py` with §8 (c2probe) | 21.5 s | 1.5 min |
| 04:16 | `rev_pin.py` with §9 (D10) and the post-pin recovery entry | 22.0 s | 1.9 min |
| 04:17–04:20 | `rev_gen.py` (generalization Part A): one crash on the `steps.csv` summary row, then two complete passes | 3.4 + 15.6 + 16.3 s | 2.4 min |
| 04:25–04:27 | `rev_mech.py` (mechanisms M1–M2, CX-M1), two passes | 13.4 + 13.5 s | 2.9 min |
| 04:29 | `rev_mech.py` at `817a768` with follow-up B | 25.3 s | 3.3 min |
