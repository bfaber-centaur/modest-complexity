# c2 generalization lane: commands, runs, budget

Question, predictions and measures: [QUESTION.md](QUESTION.md). Adaptive choices: [ADAPTIVE.md](ADAPTIVE.md). Thread "Longer programs and new tasks", branch `claude/project-thread-25n2g5`, session label `c2-gen-cloud-1`.

## Files

| file | contents |
|---|---|
| `gen_task.py` | task structure (products of any length, function classes, group order, identifiability from final tables) and the new-draw rule (`--draw`) |
| `gen_eval.py` | frozen evaluation of tied/tied checkpoints at lengths 1–6, reference systems, checks against each run's own logged metrics |
| `gen_figures.py` | figures from committed outputs only |
| `plan.json`, `registry.jsonl` | run plan (entries committed before launch) and the runner's append-only log |
| `runs/<run ID>/` | committed outputs of each run (failed attempts included) |
| `figures/` | PNG + SVG + source CSV |

## Environment

```bash
# Cloud containers ship uv 0.11.32, which cannot fetch CPython 3.12.14 (research/regimes/PLAN.md §3):
pip download --no-deps uv==0.12.23 -d /tmp/uvwheel && python3 -m zipfile -e /tmp/uvwheel/uv-0.12.23-*.whl /tmp/uvwheel/x
mkdir -p ~/.local/bin && cp /tmp/uvwheel/x/uv-0.12.23.data/scripts/uv ~/.local/bin/ && chmod +x ~/.local/bin/uv
export PATH=~/.local/bin:$PATH; uv sync --locked --group analysis
uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py   # own process; 10 tests OK on this host
R="uv run --locked python research/regimes/common/run_jobs.py --plan research/composition-c2/generalization/plan.json \
   --root runs/c2 --registry research/composition-c2/generalization/registry.jsonl --session-label c2-gen-cloud-1"
```

Host `vm`: 4 visible CPUs, 16 GB, one BLAS thread per process (set by `run_jobs.py`).

## Part A: frozen evaluation (no training)

```bash
uv run --locked python research/composition-c2/generalization/gen_task.py --task-id ops51000-ls51100   # task facts in QUESTION.md
$R --workers 1 --ids c2-gen-frozen-orig-a2
cp runs/c2/c2-gen-frozen-orig-a2/* research/composition-c2/generalization/runs/c2-gen-frozen-orig-a2/
uv run --locked python research/composition-c2/generalization/gen_figures.py length
```

- `c2-gen-frozen-orig-a1` **failed** (kept, `runs/c2-gen-frozen-orig-a1/stdout.log`): `c2rnn.forward` without a step map uses the map 1..L, which fails its own range assertion for L > 3 even when both groups are tied. Tied groups ignore the map, so a2 passes the explicit map `[1, 2, 3, 3, …]` that `test_c2.py` exercises. Pin note for the coordinator and reviewer; the shared code was not edited.
- `c2-gen-frozen-orig-a2` complete, 7.1 s. Every check in `checks.json` holds: SHA-256 of all six checkpoints equal RECOVERY.json; at length 3 the 128-character held-out prediction string and the train/held final accuracies equal r1's logged step-10000 row; length-1/2 accuracy equals the logged train step1/step2 accuracy; the causality identity holds exactly (max difference 0); the oracle loop agrees with `c2task.rows` on every row of lengths 1–6; P2's hash equals r1's manifest.

Figure: [figures/length.png](figures/length.png) (`.svg`, source `figures/length.csv`).

## Budget (25 min elapsed, D6)

| run | status | wall s (runner) |
|---|---|---|
| c2-gen-frozen-orig-a1 | failed (exit 1) | 0.2 |
| c2-gen-frozen-orig-a2 | complete | 7.1 |
| **total** | | **7.3 s of 1,500 s** |
