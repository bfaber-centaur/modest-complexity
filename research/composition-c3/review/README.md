# c3 reviewer directory

- [QUESTION.md](QUESTION.md): what is checked and the predictions, written before each measurement (Revisions are labelled by the commit that records them).
- [REVIEW.md](REVIEW.md): verdicts with exact artifact references.
- [rev_c3pin.py](rev_c3pin.py): pin check of `be41d16` (D3). It reuses the c2 reviewer's own forward, loss and oracle from `../../composition-c2/review/rev_pin.py`. Its output is [out/rev_c3pin.txt](out/rev_c3pin.txt). It checks `be41d16` and runs at that commit. From `82cef77` on, `check_source` also takes the Adam count, so `rev_c3pin.py` stops there with a TypeError. `rev_repin.py` RP3 reruns its 41 pairings under the new signature.
- [rev_corr.py](rev_corr.py): correction-lane recomputation, C1–C7, CX-C1, CX-C2 (run `c3-rev-corr-a2`; output [out/rev_corr.txt](out/rev_corr.txt)). [rev_corr_note.py](rev_corr_note.py) checks the NOTE's numbers ([out/rev_corr_note.txt](out/rev_corr_note.txt)).
- [rev_repin.py](rev_repin.py): re-pin check of `82cef77` (D10), RP0–RP7 and CX-R1 (run `c3-rev-repin-a1`; output [out/rev_repin.txt](out/rev_repin.txt)).
- [rev_scaf.py](rev_scaf.py): scaffolding-lane review, S0–S7, CX-S1 and CX-S2 (run `c3-rev-scaf-a1`; output [out/rev_scaf.txt](out/rev_scaf.txt)). [rev_scaf_note.py](rev_scaf_note.py) checks the NOTE's numbers ([out/rev_scaf_note.txt](out/rev_scaf_note.txt)).
- [rev_scaf_fu.py](rev_scaf_fu.py): scaffolding follow-up review, F1–F4, CX-F1 and the NOTE's §F numbers (run `c3-rev-scaf-fu-a1`; output [out/rev_scaf_fu.txt](out/rev_scaf_fu.txt)). [rev_scaf_fu_posthoc.py](rev_scaf_fu_posthoc.py): post-hoc reset-effect table ([out/rev_scaf_fu_posthoc.txt](out/rev_scaf_fu_posthoc.txt)).
- [rev_synth.py](rev_synth.py): synthesis check (SY4 run counts and wall time from the c3 registries; post-hoc keep control on logged rows; the correction call site at the head). Reads only; output [out/rev_synth.txt](out/rev_synth.txt).
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
$R --workers 2 --ids c3-rev-repin-drop-s51200-a1 c3-rev-repin-taper-s51201-a1 c3-rev-reset-drop-s51200-a1 c3-rev-reset-keep-s51201-a1
$R --workers 2 --ids c3-rev-reset-drop-cont1200-s51200-a1 c3-rev-reset-drop-cont1000-s51200-a1   # entries hold the first stage's checkpoint hashes
$R --workers 1 --ids c3-rev-repin-a1   # rev_repin.py --runs runs/c3; stdout in runs/c3/c3-rev-repin-a1.log
$R --workers 1 --ids c3-rev-scaf-a1    # rev_scaf.py --lane 4512bce; stdout in runs/c3/c3-rev-scaf-a1.log
python3 research/composition-c3/review/rev_scaf_note.py runs/c3/c3-rev-scaf-a1
$R --workers 1 --ids c3-rev-scaf-fu-a1   # rev_scaf_fu.py --lane 5317c54
python3 research/composition-c3/review/rev_scaf_fu_posthoc.py runs/c3/c3-rev-scaf-fu-a1
uv run --locked python research/composition-c3/review/rev_synth.py > research/composition-c3/review/out/rev_synth.txt   # at 05515a2 or later (runs test_corr.py)
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
| 5ce9d97 (06:13) | four c3train runs at `82cef77`, two workers | 12.8 s | 8.3 min |
| 55cf081 (06:14) | two continuations, two workers | 2.7 s | 8.4 min |
| f3abedc (06:16) | `c3-rev-repin-a1` (includes both test suites, 29 s) | 39.5 s | 9.0 min |
| d7a1613 (06:24) | `c3-rev-scaf-a1` | 77.8 s | 10.3 min |
| 8f2b28c (06:27) | `rev_scaf_note.py` (reads only) | 0.6 s | 10.3 min |
| 5236c16 (06:45) | `c3-rev-scaf-fu-a1` | 72.2 s | 11.5 min |
| after the run (06:47) | `rev_scaf_fu_posthoc.py` (reads only) | < 1 s | 11.5 min |
| d31d228, 42210be (07:03) | `rev_synth.py` (reads only; includes `test_corr.py`), plus a first registry-only pass | 2.0 s | 11.5 min |
