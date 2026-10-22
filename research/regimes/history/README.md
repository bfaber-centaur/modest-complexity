# R3 training history and recovery: commands, environment, budget, runs

Design and predictions: [QUESTION.md](QUESTION.md), pushed at 3369031 before any run. Results so far: [figures/](figures/). NOTE.md follows at M4.

## Files

| file | what it is |
|---|---|
| `history.py` | Tasks, split and the phase runner: optimizer reset per phase, frozen-head assertion, checkpoints with SHA-256, in-run readouts and functional-change measures |
| `test_history.py` | Independent checks: oracle by enumeration, split, depth-0 readout finite differences, branch = in-process sequence bitwise, optimizer reset against a re-implementation, SHA mismatch refused |
| `figures.py` | `figures/recovery_curves.*` (primary) and `figures/survival_summary.*`, each with a CSV |
| `plan.json`, `registry.jsonl` | Every run's argv and timeout (each pushed before launch), and the runner's append-only status log |
| `runs/<ID>/` | Published copies: `config.json`, `summary.json` (snapshots: metrics, readouts, null, functional change), `runtime.json`, `metrics.jsonl.gz`, `run.log`. Phase-A runs also carry `ckpt_t2000.npz` (= ckA) |
| `runs/MANIFEST.jsonl` | SHA-256 and size of every file in every local run directory, published or not (unpublished: init and branch checkpoints, about 43 KB each) |

## Commands

```bash
# environment (PLAN section 3): uv 0.12.23, then
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m unittest -v research/regimes/common/test_nets.py research/regimes/history/test_history.py

# M1: phase-A runs (fresh-A reference, positive control, ckA) and scratch-B runs
uv run --locked python research/regimes/common/run_jobs.py --plan research/regimes/history/plan.json \
  --root runs/regimes --registry research/regimes/history/registry.jsonl --session-label r1-hist-cloud-1 --workers 2 \
  --ids r1-hist-A-s53200-a1 r1-hist-A-s53201-a1 r1-hist-A-s53202-a1 \
        r1-hist-scratch-Brel-s53200-a1 r1-hist-scratch-Bunrel-s53200-a1 r1-hist-scratch-Brel-s53201-a1 \
        r1-hist-scratch-Bunrel-s53201-a1 r1-hist-scratch-Brel-s53202-a1 r1-hist-scratch-Bunrel-s53202-a1
# initial comparison: the 39 branch entries of plan.json (d0, rel, unrel, contA, conflict x 3 seeds), same runner
#   --ids $(python3 -c "import json; print(' '.join(r['run_id'] for r in json.load(open('research/regimes/history/plan.json'))['runs'][9:48]))")

# figures, from the local runs or from the published copies (identical CSVs, checked)
uv run --locked python research/regimes/history/figures.py --runs runs/regimes
uv run --locked python research/regimes/history/figures.py --runs research/regimes/history/runs
```

Branch entries name their source checkpoint and its SHA-256 (`--from … --from-sha256 …`). The run refuses a mismatch, and also checks the weights against the hash stored at save time.

## Environment and calibration (this host)

- **Host.** Linux 6.18.44 x86_64, Intel Xeon @ 2.80 GHz, 4 visible CPUs (affinity 0–3).
- **Software.** CPython 3.12.14 and NumPy 2.3.5 (scipy-openblas 0.3.30) from the locked uv environment, with all BLAS/OpenMP thread variables at 1.
- **Workers.** Two concurrent workers through `run_jobs.py`, which also sets the malloc variables.
- **Measured over all 48 runs.**
  - Full-batch updates (256 rows, 10→128→2 heads) ran at 1665–2003 per second (median 1858).
  - Each snapshot (checkpoint, two 2000-update readouts, functional change) took about 1 s.
  - Peak RSS was 44 MiB per process.
  - The longest run took 12.1 s wall, including startup.
- **Timeouts.** M1 runs used 300 s. Branch timeouts were set after M1 to 3 × (updates/1500 + 4 s), at least 30 s.
- **Outcome.** No run was capped or failed.

## Budget tally (elapsed numerical execution; two concurrent jobs count once)

| step | UTC | elapsed |
|---|---|---|
| lane unit tests (not research runs; listed for honesty) | 02:41 | ~14 s |
| M1: 3 phase-A + 6 scratch-B runs | 02:43:46–02:44:23 | 37 s |
| initial comparison: 39 branch runs | 02:45:01–02:47:35 | 154 s |
| figures (under a minute, not counted) | | ~10 s |
| **total** | | **~3.4 min of 25** |

## Run manifest summary

48 runs, all `complete` in `registry.jsonl`, none failed or capped:

| ID pattern | count |
|---|---|
| `r1-hist-A-s{53200,53201,53202}-a1` | 3 |
| `r1-hist-scratch-{Brel,Bunrel}-s*-a1` | 6 |
| `r1-hist-d0-s*-a1` | 3 |
| `r1-hist-{rel,unrel,contA,conflict}-d{100,1000,10000}-s*-a1` | 36 |

**Seeds by role.** Data split 53100, init 53200–53202 (paired across all arms), readout init 53300, null permutation 53301. There is no task seed: the tasks are fixed parities.

**Split.** 256 train and 768 held-out. Positive labels in train: A 132, Brel 132, Bunrel 121. Per-pattern counts of each task's three bits in train range from 24 to 38 (`config.json` → `split_stats`).

**Dependence.** Branches from one ckA are dependent, so each comparison has three independent units (init seeds), all on one data split.

## First observations (M2, descriptive; ranges over the three init seeds, from `figures/survival_summary.csv`)

**d = 100.** B is not yet learned: held-out B is 0.47–0.52. Held-out A through the original head stays at 0.988–0.999.

**d = 1000 and d = 10000.** B is learned (≥ 0.93 held-out), and held-out A through the original head falls:
- related B: 0.76–0.86 at d = 1000 and 0.69–0.80 at d = 10000;
- unrelated B: 0.76–0.83 and 0.71–0.77.

The A head is bitwise unchanged, so this is trunk drift. Continued-A for the same updates keeps ≥ 0.995.

**Frozen-feature readout** of A, held-out:
- on the end-of-B trunk: 0.91–1.00 for both B tasks at both durations;
- on a trunk trained on the same B for the same d without A history: 0.49–0.79;
- on the untrained trunk: 0.58–0.63;
- on the end-of-A trunk: 0.99–1.00;
- on shuffled labels: 0.46–0.58.

This matches QUESTION.md outcome 2: A-relevant features remain linearly decodable while the original head reads them wrongly.

**Relearning** to held-out A ≥ 0.95 takes 37–560 updates, against 540–700 for fresh-A from the same init. Savings ratios are 0.05–0.81, and nothing was censored. This alone does not say what carried the savings, because B never touches the A head.

**Related vs unrelated B.** There is no consistent ordering across seeds in end-of-B A accuracy or relearning cost. In seed 53200 unrelated B did more damage; in 53201 and 53202 related B did.

**Learning B after A.** Unrelated B took longer to reach criterion than from scratch in all three seeds (590–1400 against 370–480 updates). Related B took about the same (510–640 against 460–600).

**Conflict (separate experiment).** notA through the shared head drives held-out A to 0.00–0.02 by d = 1000. The readout recovers 0.98–1.00, as expected and uninformative, because notA needs the same features. Relearning takes 190–440 updates at d ≥ 1000 and 29–54 at d = 100.

**B after the return to A.** Held-out B ends at 0.76–0.94 after d = 1000 of B, and at 0.88–1.00 after d = 10000.
