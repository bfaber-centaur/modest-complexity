# e2 decision log

Charter: [charter/](charter/README.md). Launch messages: [DISPATCH.md](DISPATCH.md). Newest entries last; earlier entries are not edited, only superseded by later ones.

## D1 (2026-10-21) Close e1

All 24 map runs and the 4 follow-up runs (e1f1, e1f2 ×3) are complete; none failed or was capped (`research/coordinator/RUNS.md`). The last e1 integrations are the mechanism lane's per-unit packet (PR #12, `2a3719e`) and audit Part 4 (PR #13, `5c23e33`). e1 artifacts stay where they are and remain reusable; e1 notes get dated pointer notes, not rewrites.

## D2 (2026-10-21) Four sessions: the existing threads take the e2 roles

| e2 role | session (thread) | owned directory | branch | registry / session label |
|---|---|---|---|---|
| coordinator | "Exploratory swarm launch" | `research/e2/coordinator/`, `research/e2/plan.json`, `research/e2/data/`, shared code | `swarm/exploration` | `research/e2/coordinator/registry.jsonl` / `e2-co-cloud-1` |
| dynamics explorer | "Width and training-time map" (e1 size/time) | `research/e2/dynamics/` | `claude/project-thread-j8mgkm` | `research/e2/dynamics/registry.jsonl` / `e2-dy-cloud-1` |
| mechanisms explorer | "Frequency retention and rescaling" (e1 mechanisms) | `research/e2/mechanisms/` | `claude/project-thread-psralm` | `research/e2/mechanisms/registry.jsonl` / `e2-me-cloud-1` |
| independent auditor | "Correctness audit" (e1 auditor) | `research/e2/audit/` | `claude/project-thread-id1tmm` | `research/e2/audit/registry.jsonl` / `e2-au-cloud-1` |

No new threads. Reason: the charter fixes four sessions, and these sessions already hold the environment and the e1 artifacts. Consequence: both explorers have already seen the 30%/50% comparison (size/time PR #11, mechanisms PR #12), so a QUESTION.md that uses it must say what was seen before the question was written. It is prospective only for observations not yet made.

## D3 (2026-10-21) Execution budget

Each session gets 30 minutes of cumulative elapsed numerical execution for e2, counted from its e2 start, at most two processes and one BLAS thread each. e1 usage stays as recorded in the e1 notes (coordinator about 25, mechanisms about 14, auditor about 23 minutes). Each session reports e1 and e2 usage separately. No paid compute.

## D4 (2026-10-21) Pinned base and shared code

- Base for all three workers: the `swarm/exploration` commit that adds this file (hash in [INDEX.md](INDEX.md) and in the launch messages). A later `swarm/exploration` commit is also a valid base while it changes no shared code.
- Shared code (`experiment.py`, `launch_runs.py`, `report_map.py`, `publish_run.py`, `instrumentation.py`, `pyproject.toml`, `uv.lock`, the shared tests) is unchanged since `fe284a5`; its identity for e2 is `fe284a5d90f741aab8c1273a55b11e5352ec38c5`. A shared-code change is proposed as a patch on a lane branch; once integrated it gets a new pinned hash recorded here, and dependent runs wait for it.
- Workers merge `swarm/exploration` into their designated branches (the branches stay the same; each PR targets `swarm/exploration`).

## D5 (2026-10-21) Run-ID claims

- One claim file: [research/e2/plan.json](../plan.json), same format as `research/plan.json` (`common` + `runs`; every e2 run sets its own `split_sha256`, and `train_fraction` when not .3). It starts empty.
- ID: `e2-<co|dy|me|au>-<condition>-w<width>-i<init seed>[-tf<share>][-d<data seed>][-s<steps>]-a<attempt>`, e.g. `e2-dy-add-w128-i30000-tf40-a1`.
- To claim: send the coordinator the plan entries (run ID, every field that differs from `common`, reason, expected minutes) through the thread, or push them as `research/e2/<lane>/REQUEST.md`. The coordinator adds them to `research/e2/plan.json`, pushes, and replies with the commit. Launch from a tree that contains that commit, with `launch_runs.py --plan research/e2/plan.json --registry research/e2/<lane>/registry.jsonl`. The launcher refuses IDs not in the plan.
- Local outputs under `runs/e2/`; publish with `publish_run.py` under `research/e2/<lane>/runs/<run_id>/`.

## D6 (2026-10-21) Common never-trained evaluation set (charter priority issue)

Checked, not inferred from the common data seed: the 30% training set is a subset of the 50% one (299 ⊂ 478; 179 added). The pairs trained on by neither are 483 pairs (248 unordered groups, transpose-closed, 15–16 per class), exactly the 50% split's cal ∪ audit, all inside the e1 audit split. Saved as `research/e2/data/common-never-trained-30-50.npz` (`index`, SHA-256 `1535a21a…4d3`). Scores of all six width-128 runs on it are in `common_set_scores.csv` with each run's own cal/audit accuracy alongside. Script: `common_set.py` (uses the shared `experiment.forward`, so it is not an independent check; the auditor's `nested_heldout.py` gives the same picture on the 364-pair subset).

Any 30%-vs-50% contrast in e2 reports this set next to each run's own whole-split metrics. What the set does not remove: the 50% share adds specific pairs (all 118 e1 calibration pairs and 61 e1 audit pairs), so amount and identity of training data are confounded (audit Part 4: AMBIGUOUS).

## D7 (2026-10-21) Stale claims reconciled at e2 start

| claim (source) | change | evidence |
|---|---|---|
| "a different partition of the same 961 pairs" (e1 INDEX follow-up 2; e1 INTERPRETATION) | The training sets nest; the held-out difference is removed by D6 | `common_set.json`; audit Part 4 |
| e1f2 "Not yet audited" (e1 INDEX) | Software PASS on data, recipe, metrics at every published checkpoint, onset replay to 5k for all seeds, full replay of i30001 (audit F10–F15). "50% generalizes where 30% does not" is SUPPORTED LOCALLY for one split, one width, three seeds; "the difference is the amount of data" is AMBIGUOUS. Not "confirmed" | audit Part 4 |
| e1f1 audit CE "peaks at 8.99 near step 27,000" (e1 INDEX) | 9.020 at 25,600 | audit Part 4 |
| e1f2 "at or below chance until about 3,000–4,000 updates" (e1 INDEX) | 3,900 / 5,500 / 4,800 by seed | audit Part 4 |
| width ≥ 128 "stationary by about 20,000–25,000 updates" (size/time interpretation 1) | Norm and AdamW balance yes; train CE at width 128 still falls 3.9× from 25k to 50k | audit Part 4 |
| "The packet-1 alternative is confirmed in its first half" (mechanisms INTERPRETATION, follow-up 2) | Per-unit concentration is an observation on unaudited outputs; its link to the computation is a candidate explanation | this log |

Unresolved: the restricted-rank statistic comes from the auditor's `research/audit/heldout_rank.py` (the coordinator's follow-up figures call it), reimplemented in the mechanisms lane's `relational.py` and as a loop in its `crosscheck.py`, which agree on the checkpoints they share. All of them follow the same written definition, so the definition itself (mask from training pairs only, operand positions, ties, set sizes, chance reference, like-for-like across shares) and a null that keeps the mask are the auditor's e2 target.
