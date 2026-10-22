# c2 generalization lane: commands, runs, budget

Question, predictions and measures: [QUESTION.md](QUESTION.md). Adaptive choices: [ADAPTIVE.md](ADAPTIVE.md). Thread "Longer programs and new tasks", branch `claude/project-thread-25n2g5`, session label `c2-gen-cloud-1`.

## Files

| file | contents |
|---|---|
| `gen_task.py` | task structure (products of any length, function classes, group order, identifiability from final tables) and the new-draw rule (`--draw`) |
| `gen_eval.py` | frozen evaluation of tied/tied checkpoints at lengths 1–6, reference systems, checks against each run's own logged metrics |
| `gen_figures.py` | figures from committed outputs only (`length`, `replication`, `followup`) |
| `gen_publish.py` | copies finished runs into `runs/<ID>/` with a SHA-256 manifest |
| `gen_followup.py` | the follow-up: first-error anatomy at steps 4–6 |
| `tasks.json` | the draw rule's decision log and every new task's structure |
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

## Part B: replication on new tasks

```bash
uv run --locked python research/composition-c2/generalization/gen_task.py --draw > research/composition-c2/generalization/tasks.json
$R --workers 1 --ids c2-gen-calib-inter-s900130-a1
$R --workers 2 --ids $(python3 -c "import json; print(' '.join(r['run_id'] for r in json.load(open('research/composition-c2/generalization/plan.json'))['runs'] if '-ops63' in r['run_id']))")
uv run --locked python research/composition-c2/generalization/gen_publish.py --root runs/c2 c2-gen-calib-inter-s900130-a1 <the 16 IDs>
$R --workers 1 --ids c2-gen-frozen-rep-a1      # reads the committed run directories
cp runs/c2/c2-gen-frozen-rep-a1/* research/composition-c2/generalization/runs/c2-gen-frozen-rep-a1/
uv run --locked python research/composition-c2/generalization/gen_figures.py replication
```

- **Draws** (`tasks.json`, committed before training): seeds 63000 and 63001 were both accepted by the rule; split seeds 63100–63103 all accepted. No rejected draw or split. Tasks `ops63000-ls63100`, `ops63000-ls63101`, `ops63001-ls63102`, `ops63001-ls63103`. In draw 63001 all four ops send symbol 2 to symbol 1; this is allowed by the rule and raises its 3-of-4-slot neighbour vote to 0.41 (0.08–0.12 on the other draws).
- **Calibration** (`c2-gen-calib-inter-s900130-a1`, 2,000 updates, one job alone): 4.1 ms/update, peak RSS 50 MiB. Part B runs took 39–44 s each with two concurrent jobs (about 4.2 ms/update).
- **Runs:** all 16 complete, none capped or failed. Each `runs/<ID>/` holds config.json (full recipe: every flag explicit, reviewer C1), metrics.jsonl (every 50 updates, with `--diag`), runtime.json, stdout.log, ckpt-10000.npz (weights + Adam state) and MANIFEST.json (ckpt-0 by hash only; it is the deterministic init of the recorded seed). The calibration checkpoint is listed by hash and not committed.
- `c2-gen-frozen-rep-a1`: every check holds for all 16 (length-3 prediction strings and accuracies equal each run's own logged step-10000 row; causality identity exact).

Figure: [figures/replication.png](figures/replication.png) (`.svg`; sources `figures/replication.csv`, `figures/replication_trajectories.csv`). The original-task column is drawn from r1's published metrics (¬N scored from the logged prediction strings).

## Follow-up: first-error anatomy (approved at 99c83c8)

```bash
$R --workers 1 --ids c2-gen-followup-anatomy-a1
cp runs/c2/c2-gen-followup-anatomy-a1/* research/composition-c2/generalization/runs/c2-gen-followup-anatomy-a1/
uv run --locked python research/composition-c2/generalization/gen_figures.py followup
```

`c2-gen-followup-anatomy-a1` complete, 11.5 s: frozen forward passes of the 11 tied/inter `ckpt-10000` (hashes checked), outputs `anatomy.csv`, `windows.csv`, `summary.json`. Figure: [figures/followup.png](figures/followup.png) (`.svg`, `figures/followup.csv`).

## A5: the time lane's 50k states

```bash
C=543ed95   # time lane branch claude/project-thread-xb3nfz when evaluated
mkdir -p runs/c2-inputs/time-$C && git archive $C $(git ls-tree -d --name-only $C research/composition-c2/time/runs/ | grep -E 'c2-time-tt-(inter|final)-s5120[0-2]-a1$') | tar -x -C runs/c2-inputs/time-$C
$R --workers 1 --ids c2-gen-frozen-50k-a1
cp runs/c2/c2-gen-frozen-50k-a1/* research/composition-c2/generalization/runs/c2-gen-frozen-50k-a1/
uv run --locked python research/composition-c2/generalization/gen_figures.py length
```

`c2-gen-frozen-50k-a1` complete, 6.7 s. All six `ckpt-50000` hashes match the time lane's MANIFEST.json; every check against their own logged step-50000 rows holds. Drawn as dashed lines in `figures/length.png`.

## Budget (25 min elapsed, D6)

Elapsed wall time with at least one job running, from the registry (union of run intervals):

| runs | status | elapsed s |
|---|---|---|
| c2-gen-frozen-orig-a1 | failed (exit 1) | 0.2 |
| c2-gen-frozen-orig-a2 | complete | 7.1 |
| c2-gen-calib-inter-s900130-a1 | complete | 8.6 |
| 16 Part B runs, two workers | all complete | 337.6 |
| c2-gen-frozen-rep-a1 | complete | 14.2 |
| c2-gen-followup-anatomy-a1 | complete | 11.5 |
| c2-gen-frozen-50k-a1 | complete | 6.7 |
| **total** | | **385.8 s (6.4 min) of 1,500 s** |

No run was capped. The one failure (frozen a1) is kept with its log.
