# c2 mechanisms lane (thread "Hidden-state interchange")

- **Question, measures and predictions:** [QUESTION.md](QUESTION.md), committed at `65f9245` before any measurement.
- **Adaptive choices:** [ADAPTIVE.md](ADAPTIVE.md).
- **Assignment:** DECISIONS D7.
- **Inputs:** the 27 recovered r1 checkpoints in `../recovered/`, each verified by SHA-256 against `RECOVERY.json` before use.
- **No training.**

## Files

| file | role |
|---|---|
| `mech_core.py` | checkpoint list and verified loading, prefix/suffix enumeration, patching with the declared readout map, agreement measures |
| `mech_eval.py` | M1: reproduce r1's published metrics line for all 27 checkpoints |
| `mech_interchange.py` | M2–M4: interchange over all pairs, consistency, centroid patches, controls, geometry and in-run checks |
| `mech_probe.py` | D5 decode table (`c2probe.decode_table` unchanged, probe seed 64300) |
| `mech_collect.py` | copies run outputs into `runs/RUN_ID/` with `MANIFEST.json` |
| `mech_figure.py` | `figures/interchange.*` (primary) and `figures/centroid.*` (supporting) |
| `test_mech.py` | 6 tests, own process |

## Commands

From the repository root, with uv 0.12.23 (PLAN §3) and `uv sync --locked --group analysis`:

```bash
uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py         # 10 tests, own process
uv run --locked python -m unittest -v research/composition-c2/mechanisms/test_mech.py   # 6 tests, own process
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/mechanisms/plan.json --root runs/c2 --registry research/composition-c2/mechanisms/registry.jsonl --session-label c2-mech-cloud-1"
$R --workers 2 --ids c2-mech-calib-a1
$R --workers 2 --ids c2-mech-eval-a1 c2-mech-interchange-a1 c2-mech-probe-a1
uv run --locked python research/composition-c2/mechanisms/mech_collect.py --root runs/c2 c2-mech-calib-a1 c2-mech-eval-a1 c2-mech-interchange-a1 c2-mech-probe-a1
uv run --locked python research/composition-c2/mechanisms/mech_figure.py
```

## Calibration and budget

The budget is 15 min of elapsed time with a job running (DECISIONS D6). Host label `c2-mech-cloud-1` has 4 CPUs and one BLAS thread per job.

| run | status | wall s (registry) | what |
|---|---|---|---|
| `c2-mech-calib-a1` | complete | 18.1 | interchange on the nine `ckpt-0` files: about 2 s per checkpoint |
| `c2-mech-eval-a1` | complete | 0.5 | M1, 27 checkpoints |
| `c2-mech-interchange-a1` | complete | 50.1 | M2–M4, 27 checkpoints |
| `c2-mech-probe-a1` | complete | 12.6 | D5 tables, nine 10k checkpoints |

- **Elapsed with a job running:** about 1.2 min of 15. The second batch ran concurrently with two workers, so it counts as about 50 s.
- **Not counted:** tests and figure rendering, each under 10 s.
- **No failed or capped runs.** The registry is append-only.

## M1: oracle and published metrics (`runs/c2-mech-eval-a1/`)

- **Labels.** All 512 rows equal an independent loop over r1's `execute`, and ¬N has 42 rows.
- **Published metrics.** All 27 checkpoints (3 conditions × 3 seeds × steps 0, 1,000 and 10,000) reproduce their published r1 metrics line exactly: every head's train and held-out accuracy and cross-entropy, and `held_final_pred`.
- **Held-out final at 10k:**

  | condition | 51200 | 51201 | 51202 |
  |---|---|---|---|
  | tied/inter | 1.000 | 0.961 | 1.000 |
  | per-step readout/inter | 0.633 | 0.523 | 0.523 |
  | tied/final | 0.273 | 0.312 | 0.320 |

## M2 observations (10k; first figure)

[figures/interchange.png](figures/interchange.png) is the primary figure (`.svg`, `.csv`); [figures/centroid.png](figures/centroid.png) is supporting. Every number below is in `runs/c2-mech-interchange-a1/` (`cells.csv`, `pairs.csv`, `geometry.csv`) or `runs/c2-mech-probe-a1/decode_table.csv`.

**Equivalence (checks, not findings).** These hold for all 27 checkpoints (`checks.json`):
- Every patched final state equals the plain forward pass on `donor prefix ++ base suffix` bitwise (maximum difference 0.0).
- Self-swaps leave every state and prediction unchanged.
- Cells with equal i − j give identical predictions for every (start, program).

So the matrices below are accuracy on recombined programs of length L = 3 + i − j, re-indexed. Same-step swaps (L = 3) are ordinary length-3 evaluation. For tied/inter and tied/final, L = 1..6 is the generalization lane's frozen length evaluation, so agreement between the lanes is a cross-check.

**Natural interchange S, by L (seeds 51200 / 51201 / 51202):**

| condition | L = 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| tied/inter | 1.00 / 1.00 / 1.00 | 1.00 ×3 | 1.00 ×3 | 1.00 / 0.99 / 1.00 | 0.99 / 0.93 / 0.98 | 0.95 / 0.74 / 0.88 | 0.86 / 0.56 / 0.72 |
| per-step readout (`V3`) | 0.12 / 0.25 / 0.12 | 0.44 / 0.50 / 0.62 | 0.52 / 0.58 / 0.56 | 0.91 / 0.88 / 0.88 | 0.26 / 0.20 / 0.26 | 0.16 / 0.15 / 0.18 | 0.13 / 0.13 / 0.14 |
| tied/final | 0.00 / 0.25 / 0.25 | 0.12 / 0.09 / 0.06 | 0.15 / 0.15 / 0.09 | 0.82 / 0.83 / 0.83 | 0.14 / 0.12 / 0.14 | 0.12 / 0.11 / 0.12 | 0.13 / 0.13 / 0.12 |

L = 0 has only 8 rows: the readout applied to h_0.

**History consistency C.** C is the agreement of patched answers between different histories with the same donor symbol.
- **tied/inter.** C is ≥ 0.966 wherever S ≥ 0.97. It falls with S at long lengths: at (3, 0), L = 6, it is 0.77 / 0.43 / 0.61. So the long-program errors depend on the donor's history; they are not consistently wrong answers.
- **The other two conditions.** C is 0.16–0.58 in every off-diagonal cell, and 0.64–0.83 on the diagonal.
- **Held-out histories (i = 3).** The agreement between held-out-history and training-history donors is close to the within-group C, for example 0.75 / 0.41 / 0.58 at L = 6 in tied/inter.

**Semantic-centroid patches.** These are not input recombinations, and they show the effect that whole-state interchange cannot.
- **tied/inter.** S_cent ≥ 0.94 in all 48 cells, including L = 6, where natural states give 0.56–0.86. The class-mean state carries the answer across every tested position and length. The history-specific part of the natural state goes with the long-length errors.
- **per-step readout.** At (2, 2), the depth-2 centroid followed by op 3 is correct for all 32 (symbol, op) pairs in all three seeds (1.00). Natural depth-2 states give 0.88–0.91 overall and 0.52–0.63 on held-out programs. `V3` reads the depth-2 centroid's symbol at 0.88 / 1.00 / 1.00, against 0.52–0.58 for natural h_2.
- **tied/final.**
  - At (2, 2) the centroid gives 0.88 / 0.88 / 0.97, against 0.27–0.32 for natural states on held-out programs.
  - At (1, 1) the centroid gives 0.65–0.77, which is below natural (0.82–0.83).
  - Off the diagonal it stays at chance.
- **(3, 3)** is a linear readout of a class mean (1.00 everywhere), so it is near-trivial.

**Controls.** In cells with at least 32 combinations, target-hits of the permuted donor (R1) and the per-unit marginals (R2) are 0.00–0.28. The L = 0 cells have 8 combinations and give 0.00–0.38. Real-donor S in tied/inter exceeds both controls by ≥ 0.73 wherever S ≥ 0.8.

**Descriptive geometry and decoding.**
- **History spread** (within-symbol over between-centroid squared distance) is > 1 at depths 1–3 in every condition: tied/inter 1.2–2.8, per-step readout 1.2–6.5, tied/final 1.7–9.0. It grows with depth.
- **D5 table.** Held-out h_3 decodes at 0.94–0.99 (tied/inter), 0.51–0.60 (per-step) and 0.29–0.34 (tied/final).
- Neither shows use.

**Pair subsets** (`pairs.csv`).
- **Both rows correct unpatched** (`both_correct`). Off the diagonal this changes S by ≤ 0.013. On the diagonal it raises S by ≤ 0.02 at (1, 1) and (2, 2), and by up to 0.12 at (0, 0). At (3, 3) it gives 1.00 by construction, because the recombined program is then the donor's own correct row.
- **Same-state/different-history pairs** give the same S as all pairs, exactly as derived in QUESTION.md.

## Predictions scored (QUESTION.md)

| # | prediction | outcome |
|---|---|---|
| 1 | M1 exact | held, 27/27 |
| 2 | diagonals ≥ 0.99 / 0.87–0.91 / 0.81–0.83 | held |
| 3 | tied/inter: L = 1, 2 ≥ 0.95 | held (1.00) |
|  | L = 0 ≤ 0.5 | **failed**: 1.00; the shared readout reads h_0, which it was never trained on |
|  | L = 4 ≥ 0.8; L = 5, 6 lower, in 0.3–0.95 | held |
|  | C ≥ 0.95 where S ≥ 0.95 | **failed narrowly**: C 0.91–0.92 in two cells with S = 0.95 |
|  | S_cent within 0.05 of S at L ≤ 4 | **failed narrowly**, in the other direction: S_cent exceeds S by up to 0.07 (seed 51201) |
|  | spread < 0.1 | **failed**: 1.2–2.8 |
| 4 | tied/final: off-diagonal ≤ 0.25 | held |
|  | C below 1 | held |
|  | S_cent below S on the diagonal | **mixed**: below at (1, 1), above at (2, 2) |
| 5 | per-step: `V3` on h_2 0.5–0.6 | held (0.52–0.58) |
|  | lower on h_1 | **mixed**: 0.44 / 0.50 / 0.62 |
|  | L = 4–6 between the other two | held only at L = 4 |
|  | lower on held-out histories | held at (3, 3) only |
| 6 | controls 0.05–0.25 in every cell | **failed** in small cells (0.00–0.38) |
|  | margin ≥ 0.5 | held |
| 7 | snapshots: ckpt-0 off-diagonal near chance | held (0.00–0.25) |
|  | ckpt-1000 tied/inter low off the diagonal | held for L ≥ 4 (0.20–0.24); L ≤ 2 is already 1.00 |

## Reviewer handoff (M2)

1. **Equivalence and its consequence.** The equivalence is `test_mech.py` and `checks.json`. Its consequence is that natural whole-state interchange adds no evidence beyond evaluation on recombined programs.
2. **Pair aggregation.** `pairs.csv` indexes (donor prefix, suffix) combinations by pair. The QUESTION.md derivation says same-state S equals all-pairs S.
3. **Centroid reading.** Centroids are fit on training rows and use the row's true symbol. They are off-manifold states, and every (symbol, op) pair at depth 2 occurs in training. Is "history-specific variation goes with the failures" the right bound?
4. **Readout map.** The per-step readout map (`V3`, recipient frame) and the secondary `V1`/`V2` columns (`S_V1`, `S_V2` in `cells.csv`).
5. **Controls.** In the small cells (8–32 combinations) the controls are noisy.
