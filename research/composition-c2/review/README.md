# c2 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked and the predictions, written before each measurement (dated Revisions).
- [REVIEW.md](REVIEW.md): verdicts with exact artifact references.
- [rev_pin.py](rev_pin.py): pin check of `b3e8132`. Its output is [out/rev_pin.txt](out/rev_pin.txt).
- [plan.json](plan.json), [registry.jsonl](registry.jsonl): reviewer runs (IDs `c2-rev-`, seeds 65000–65999). Outputs are local in `runs/c2/` (gitignored). [runs/](runs/MANIFEST.sha256) holds their small files and a manifest.

## Commands

Environment: uv 0.12.23, `uv run --locked`, OMP/OPENBLAS/MKL threads = 1.

```bash
uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/review/plan.json --root runs/c2 \
    --registry research/composition-c2/review/registry.jsonl --session-label c2-rev-cloud-1 --workers 1 \
    --ids c2-rev-verify-final-s51200-a1
uv run --locked python research/composition-c2/review/rev_pin.py runs/c2 > research/composition-c2/review/out/rev_pin.txt
```

`rev_pin.py` §7 reads the run's `ckpt-10000.npz`, which is local-only.

## Budget tally (25 min of numerical execution, D6)

| time (UTC) | item | wall | running total |
|---|---|---|---|
| 04:01 | `c2-rev-verify-final-s51200-a1` (resume 1,000 → 10,000) | 52.4 s | 0.9 min |
| 04:05 | `rev_pin.py` first pass (§0–7, includes `test_c2.py` 13 s and two one-update guard runs) | 15.6 s | 1.1 min |
| 04:08 | `rev_pin.py` with §8 (c2probe) | 21.5 s | 1.5 min |
