# e2 dynamics: run claim request (cycle 2)

For the coordinator to add to `research/e2/plan.json` (DECISIONS.md D5). Question and predictions: [QUESTION.md](QUESTION.md) Q2, written before these runs.

Reason: a nested training-share boundary on fixed evaluation pairs. With data seed 31000 each class's 16 unordered pairs are shuffled once and the training set takes the first 5 (30%, e1), 6 (40%), 7 (45%) or 8 (50%, e1f2), so the four training sets nest and the 483 common never-trained pairs stay untrained at every share. At 40% the audit split equals the common set exactly; at 45% the common set is the audit split plus one calibration group. Checked with `experiment.data` at the shared code `fe284a5` (no code change needed; `--train-fraction` is already in the launcher).

| share | train / cal / audit pairs | added to the 30% set | split_sha256 |
|---|---|---|---|
| 0.4 | 359 / 119 / 483 | 60 pairs, all e1 calibration | `5c843bb877255505545a35a718349e654b02b7feb535561fa15751374075e5a9` |
| 0.45 | 417 / 119 / 425 | 118 pairs, all e1 calibration | `44e5774e0051bdd952ed089b3bda8513922fc856eed1efe56239b82cf633c7aa` |

Target hash is unchanged (`6f1576d4…2046`). Every field not listed is the `common` block of `research/e2/plan.json` (50,000 steps, lr .001, log every 100, checkpoint every 1,000, diagnostics none, data seed 31000).

```json
[
 {"run_id": "e2-dy-add-w128-i30000-tf40-a1", "owner": "dynamics", "task": "add", "wd": 1, "width": 128, "init_seed": 30000, "train_fraction": 0.4, "split_sha256": "5c843bb877255505545a35a718349e654b02b7feb535561fa15751374075e5a9"},
 {"run_id": "e2-dy-add-w128-i30001-tf40-a1", "owner": "dynamics", "task": "add", "wd": 1, "width": 128, "init_seed": 30001, "train_fraction": 0.4, "split_sha256": "5c843bb877255505545a35a718349e654b02b7feb535561fa15751374075e5a9"},
 {"run_id": "e2-dy-add-w128-i30002-tf40-a1", "owner": "dynamics", "task": "add", "wd": 1, "width": 128, "init_seed": 30002, "train_fraction": 0.4, "split_sha256": "5c843bb877255505545a35a718349e654b02b7feb535561fa15751374075e5a9"},
 {"run_id": "e2-dy-add-w128-i30000-tf45-a1", "owner": "dynamics", "task": "add", "wd": 1, "width": 128, "init_seed": 30000, "train_fraction": 0.45, "split_sha256": "44e5774e0051bdd952ed089b3bda8513922fc856eed1efe56239b82cf633c7aa"},
 {"run_id": "e2-dy-add-w128-i30001-tf45-a1", "owner": "dynamics", "task": "add", "wd": 1, "width": 128, "init_seed": 30001, "train_fraction": 0.45, "split_sha256": "44e5774e0051bdd952ed089b3bda8513922fc856eed1efe56239b82cf633c7aa"},
 {"run_id": "e2-dy-add-w128-i30002-tf45-a1", "owner": "dynamics", "task": "add", "wd": 1, "width": 128, "init_seed": 30002, "train_fraction": 0.45, "split_sha256": "44e5774e0051bdd952ed089b3bda8513922fc856eed1efe56239b82cf633c7aa"}
]
```

Expected cost: about 2 minutes per run on this host (e1 width-128 runs here took 116-121 s wall for 50,000 updates), 12 minutes of process time in all, about 6 minutes elapsed with two workers and one BLAS thread each. Job timeout 600 s. This lane's e2 usage before these runs: under 1 minute (analysis only).

Launch, from a clean tree containing the plan commit:

```
uv run --locked python launch_runs.py --plan research/e2/plan.json --root runs/e2 \
  --registry research/e2/dynamics/registry.jsonl --session-label e2-dy-cloud-1 --workers 2 --timeout 600 \
  --ids e2-dy-add-w128-i3000{0,1,2}-tf40-a1 e2-dy-add-w128-i3000{0,1,2}-tf45-a1
```

Publication: `publish_run.py` to `research/e2/dynamics/runs/` with steps 1,000-10,000 every 1,000 plus 15,000, 20,000, 30,000, 40,000, 50,000 and first fit (about 4 MB per run).
