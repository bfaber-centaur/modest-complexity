# c3 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked and the predictions, written before each measurement (Revisions are labelled by the commit that records them).
- [REVIEW.md](REVIEW.md): verdicts with exact artifact references.
- [rev_c3pin.py](rev_c3pin.py): pin check of `be41d16` (D3). It reuses the c2 reviewer's own forward, loss and oracle from `../../composition-c2/review/rev_pin.py`. Its output is [out/rev_c3pin.txt](out/rev_c3pin.txt).
- [rev_corr.py](rev_corr.py): correction-lane recomputation, C1–C7, CX-C1, CX-C2 (run `c3-rev-corr-a2`; output [out/rev_corr.txt](out/rev_corr.txt)). [rev_corr_note.py](rev_corr_note.py) checks the NOTE's numbers ([out/rev_corr_note.txt](out/rev_corr_note.txt)).
- [plan.json](plan.json), [registry.jsonl](registry.jsonl): reviewer runs (IDs `c3-rev-`; these reuse the original init seeds for pairing, and new seeds would come from 73000–73999). Outputs are local in `runs/c3/` (gitignored).

## Commands

Environment: uv 0.12.23, `uv run --locked`, OMP/OPENBLAS/MKL threads = 1.

```bash
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c3/review/plan.json --root runs/c3 \
    --registry research/composition-c3/review/registry.jsonl --session-label c3-rev-cloud-1"
$R --workers 2 --ids c3-rev-verify-keep10k-s51202-a1 c3-rev-drop-s51200-a1 c3-rev-c2final-s51200-a1 c3-rev-taper-whole-s51201-a1
$R --workers 1 --ids c3-rev-taper-cont-s51201-a1   # its plan entry holds the hash of the first run's ckpt-1800
uv run --locked python research/composition-c3/review/rev_c3pin.py runs/c3 > research/composition-c3/review/out/rev_c3pin.txt
$R --workers 1 --ids c3-rev-corr-a2   # rev_corr.py --lane c3f03cd; stdout in runs/c3/c3-rev-corr-a2.log
python3 research/composition-c3/review/rev_corr_note.py runs/c3/c3-rev-corr-a2/rev_corr.json
```

## Budget tally (20 min of numerical execution, D8)

| commit before the run (time) | item | wall | running total |
|---|---|---|---|
| 05aac5a (05:22) | four runs, two workers (1.9 + 1.9 + 6.8 + 10.9 s each) | 14.8 s | 0.25 min |
| plan entry (05:24) | `c3-rev-taper-cont-s51201-a1` | 5.6 s | 0.3 min |
| d3717f9 (05:26) | `rev_c3pin.py` first pass (includes both test suites, 30 s); then a one-off finite-difference step-size scan | 31.4 + 15 s | 1.1 min |
| ddff562 (05:28) | `rev_c3pin.py` second pass | 32.0 s | 1.7 min |
| ae77579 (05:47) | `c3-rev-corr-a1`, failed in C7 (my panel mapping) after every other check had run | 189.7 s | 4.9 min |
| 63625a5 (05:51) | `c3-rev-corr-a2` | 191.3 s | 8.1 min |
| 7155765 (05:55) | `rev_corr_note.py` (reads only) | 0.5 s | 8.1 min |
