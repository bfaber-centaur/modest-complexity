# Independent audit, expedition e1: Priority 1 (first preview)

Auditor session `au-cloud-1`, 2026-10-21. Branch `claude/project-thread-id1tmm` (this session's designated branch, used instead of `swarm/audit`), started from `c7c1c2ad8ac0bb23c9d0f84f9f1c9438764d03c6` (shared code identical to `ebd874e`). Published runs read from `swarm/exploration` at `67a10d8d282fd1d57ae6b3aba5a992f8d040203d`.

**Checked scope:** the Stage 0 diff `e0a6e0f..ebd874e` and the three preview runs `e1-co-add-w{32,128,512}-i30000-a1` with their registry, figures and index table. Nothing from the size/time or mechanism lanes yet. A software PASS means the stated check passed. It does not confirm any scientific claim.

**Independence:** `independent.py` imports no project code. It rebuilds the split from the written rule, uses one-hot matrix products instead of indexing and `np.add.at`, `np.logaddexp.reduce` for cross-entropy, an explicit other-logits matrix for margins, and AdamW in PyTorch's update order. `shared_checks.py` and `divergence.py` import `experiment.py` only as the code under test.

## Figure: held-out true label vs labels never seen with either operand

![held-out rank](figures/heldout-rank.png)

Look before reading: in modular addition, the true label of a held-out pair (a, b) never appears with a or b in any training pair. Does the network treat it like the other labels that never co-occur with a or b, or does it prefer it?

Source: `results/preview-1-heldout-rank.jsonl`. The figure uses only the published checkpoints. Vector version: `figures/heldout-rank.svg`.

## Software checks

Tolerances were fixed before running. Recomputed metrics must match to 1e-12 relative, with accuracies identical. Replays must match to 1e-8 relative (weights, both moments, logged loss and margins), with accuracies identical.

| # | Check | Result | Evidence |
|---|---|---|---|
| 1 | Shared unit tests at `c7c1c2a` | PASS | `test_exploration`, `test_instrumentation`: 9 tests OK |
| 2 | Cross-session handoff | PASS | Every published file (11 or 12 per run) matches `manifest.json` in SHA-256 and size, `published` flags match the files present, and everything loads |
| 3 | Split identity | PASS | Rebuilding from data seed 31000 and control seed 32000 reproduces `research/data/e1-split.npz` byte for byte. The split hash `aa8f…1f24`, addition target `6f15…2046` and shuffled target `80a9…d35f` all equal the plan |
| 4 | Split structure | PASS | The 961 pairs split 299/118/544 with no overlap and nothing missing. Every split is transpose-closed and holds exactly 5/2/9 unordered groups per residue (train/cal/audit). Labels are (a+b) mod 31. The shuffled table keeps each split's class counts and changes 96% of labels |
| 5 | Same data in every run | PASS | Each run's `data.npz` equals the frozen split (inputs, both label arrays, split indices) |
| 6 | Seed separation and initialization | PASS | Configured seeds equal the plan. Step-0 weights at every width exactly equal an independent draw from init seed 30000, and both moments are zero |
| 7 | Recipe, code and environment identity | PASS | Width, steps, lr 1e-3, wd 1, p 31, cadences, `diagnostics none` and task all equal the plan. Recorded SHA-256 of `experiment.py`, `launch_runs.py`, `instrumentation.py`, `publish_run.py` and `report_map.py`, plus the lock files, equal the `ebd874e` blobs. One BLAS thread; CPython 3.12.14 and NumPy 2.3.5 |
| 8 | Logits, accuracy, cross-entropy, margins and norm | PASS | Recomputed at every published checkpoint (7 or 8 per run, all three splits). Accuracies are identical, cross-entropy and margins agree to 4e-16 relative, the norm to 2e-16, with no ties |
| 9 | Gradients | PASS | Central differences (120 coordinates, eps 1e-6) at w32@1000, w128@0 and w512@100 agree within 1e-7 of the largest gradient. Shared `lossgrad` equals the independent gradient to 4e-15. At w512@50000 (train CE 1.8e-6) rounding limits FD (error 7% of a 1.6e-8 gradient), so that point is uninformative |
| 10 | AdamW and logged update, short replays | PASS | Independent AdamW was run between consecutive published checkpoints up to step 10,000 at all three widths. Weights and both moments agree to at most 3.4e-11 relative. At every logged row, accuracies are identical, CE agrees to 2.7e-14, margins to 2.6e-13, the adaptive and decay displacement norms to 2.3e-13 and the cosine to 6e-15. So `optimizer_update` is the actual displacement of the logged step, not an interval average |
| 11 | Long replays (10k→20k, 20k→50k) | PASS for same-host determinism; bitwise agreement is not expected | After about 17k updates the weights differ by 0.1 to 1 relative, yet logged accuracies stay within 1 example (10k→20k) and at most 6 (w32 20k→50k; 0 at w128 and w512), with CE within 7% at w32 and 0.35% at w128 and w512. Control: on this host the shared code reproduces checkpoint 20000 bitwise from 10000 (w32, w128). Moving every weight by **one ulp** makes the shared code diverge on the same schedule (w32: 1e-3 by step 18.5k; w128: 0.14 by step 21.3k; `results/div-*.json`). This is rounding amplification, not a formula difference |
| 12 | Checkpoint round-trip and contents | PASS | All published checkpoints load, are float64 and finite, have v ≥ 0, and their `step` field matches the filename. Continuing from them reproduces the next checkpoint (rows 10 and 11) |
| 13 | Checkpoint schedule | PASS | Each manifest lists steps 0 to 50000 every 1000 plus the first-fit step: 51 checkpoints at w32 (first fit at 1000), 52 at w128 (300) and w512 (100) |
| 14 | Logging semantics | PASS | 501 rows at steps 0 to 50000 by 100, no gaps. `optimizer_update` is null at step 0. `runtime.json` shows complete at 50000. `first_interpolation` equals the first logged train accuracy ≥ .99, and it stays ≥ .99 afterwards |
| 15 | Run accounting | PASS | Each of the 3 attempted runs has one `running` and one `complete` registry event. The terminal event matches `runtime.json`, and the launch command matches the plan. All 3 appear in `status.csv`, `curves.csv` and both figures. No run failed or was capped |
| 16 | Figure and index provenance | PASS (presentation notes below) | `curves.csv` equals `metrics.jsonl` on every column of all 1503 rows. `status.csv` matches. Every number in the INDEX.md observation table recomputes from `metrics.jsonl`. The exported PNGs were inspected |
| 17 | Stage 0 diff review | PASS (no defect affecting these runs; notes below) | Read `experiment.py`, `launch_runs.py`, `publish_run.py`, `report_map.py`, `test_exploration.py`, `analysis.py` and `run_study.py` |
| – | Size/time and mechanism outputs, the add0 and shuffled controls | NOT CHECKED | Not produced yet |
| – | Cross-host reproducibility | NOT CHECKED | Only one host type was used: 4 vCPU Xeon 2.1 GHz with the OpenBLAS 0.3.30 Haswell kernel |
| – | Rescaling invariance, offline vs inline diagnostics, capped/failed launcher path | NOT CHECKED | Mechanism-lane deliverable; e1 has no inline diagnostics; no run was capped |
| – | Unpublished checkpoints | NOT CHECKED | Listed by hash only |

Notes (none blocks a run, metric or figure):

- **Long-horizon sensitivity (affects all e1 runs).** Beyond about 15k updates, a trajectory depends on rounding at the 1-ulp level. Replays on another host (BLAS kernel, macOS) should reproduce logged accuracy and loss closely, but not the weights. Weight-level analyses of late checkpoints (unit identities, frequency rankings) are specific to the trajectory and the host that produced them. Run IDs: `e1-co-add-w{32,128,512}-i30000-a1`.
- `launch_runs.py` checks for run-ID reuse only against its own lane registry and output root. It does not consult other lanes' registries or the plan's `owner`. The risk is low because IDs carry the session prefix.
- Only the split hash is checked before training. The target hash is recorded but not checked, so the shuffled-label runs need a post-hoc check of `target_sha256` (`audit_preview.py` does this).
- `publish_run.py`: `producing_tree_dirty` is always true while the tracked registry is modified, so it carries no information. `published_from_commit` is the commit checked out during copying (`c7c1c2a`), not the commit that contains the files (`67a10d8`).
- INDEX.md says `git_status` lists only the untracked registry. The w32 config also lists untracked `research/coordinator/WORKER_PROMPTS.md`, which is documentation; the source hashes still match `ebd874e`.
- `loss.png`: the train and held-out rows share a log y-axis from 1e-6 to about 10, which squeezes the held-out rise (about 3.5 to 10 nats) into the top decade. The figure's prompt ("does held-out loss move with training loss") is hard to answer from it, so a separate held-out panel scale or a linear inset would help. `accuracy.png` starts at step 100, by which point width 512 has already fitted (step 0 is in `curves.csv`).

## Interpretations (seed 30000, one split, three widths)

| Statement (source) | Status | Basis |
|---|---|---|
| All three widths fit the 299 training pairs within 1000 updates and stay fitted (INDEX) | SUPPORTED LOCALLY | Checks 8 and 14 |
| The near-zero held-out accuracy is a label, split or evaluation defect (alternative, INTERPRETATION.md) | CONTRADICTED | Checks 3 to 5 and 8. Positive control: an exact modular-addition network in the same bias-free ReLU architecture (one unit per input pair) scores 1.0 on every split under both the independent evaluator and the shared `forward`/`accuracy`/`objective_metrics` |
| No held-out pair's true label co-occurs with either operand in training (premise) | SUPPORTED | It follows from the addition table, and holds for 0 of 662 held-out pairs |
| Held-out accuracy falls below chance because predictions go to labels that co-occur with an operand in training | SUPPORTED LOCALLY | At 50k, the argmax is such a label for 96% / 99.8% / 100% of held-out pairs (w32/w128/w512), against 45-51% at step 0 |
| "Fitting the training rows and columns pushes the correct class down" | AMBIGUOUS | The correct class sits below the co-occurring labels. But among the labels that never co-occur, it rises over training (figure): mean normalized rank goes from 0.50 at step 0 to 0.43 / 0.31 / 0.18 at 50k, and the true label tops them for 12% / 18% / 37% of pairs, against 7.0% expected without preference |
| "Does not generalize within 50,000 updates" | SUPPORTED LOCALLY for argmax accuracy; AMBIGUOUS about learned structure | Held-out accuracy and CE do not show the structure-dependent preference in the figure, which grows with width and time. The delayed-generalization alternative is not excluded |

The figure's preference cannot come from co-occurrence statistics: every absent label, including the true one, has zero co-occurrence with both operands. It needs relational structure. A distinguishing check would be the same rank statistic on all 51 or 52 checkpoints and on seeds 30001 and 30002 (size/time lane runs), plus a run continued past 50k at width 512.

## Reproduce

```bash
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
git fetch origin swarm/exploration
git worktree add --detach PUB 67a10d8d282fd1d57ae6b3aba5a992f8d040203d
C=PUB/research/coordinator
uv run --locked python research/audit/audit_preview.py --published $C --figures $C/figures/preview-1 \
    --code-commit ebd874e17028f2c2f04de5ecea9fe4b51b77e425 --replay \
    --ids e1-co-add-w32-i30000-a1 e1-co-add-w128-i30000-a1 --out research/audit/results/preview-1-w32-w128.json
uv run --locked python research/audit/audit_preview.py --published $C --figures $C/figures/preview-1 \
    --code-commit ebd874e17028f2c2f04de5ecea9fe4b51b77e425 --replay \
    --ids e1-co-add-w512-i30000-a1 --out research/audit/results/preview-1-w512.json
for a in "e1-co-add-w32-i30000-a1 1000" "e1-co-add-w128-i30000-a1 0" "e1-co-add-w512-i30000-a1 100" "e1-co-add-w512-i30000-a1 50000"; do
  uv run --locked python research/audit/shared_checks.py $C/runs/${a% *} ${a#* }; done   # results/preview-1-shared-checks.jsonl
uv run --locked python research/audit/divergence.py $C/runs/e1-co-add-w32-i30000-a1 10000 20000 research/audit/results/div-w32-10k-20k.json
uv run --locked python research/audit/divergence.py $C/runs/e1-co-add-w128-i30000-a1 10000 20000 research/audit/results/div-w128-10k-20k.json
uv run --locked python research/audit/divergence.py $C/runs/e1-co-add-w128-i30000-a1 10000 30000 research/audit/results/div-w128-10k-30k.json
uv run --locked python research/audit/heldout_rank.py $C/runs/e1-co-add-w{32,128,512}-i30000-a1 > research/audit/results/preview-1-heldout-rank.jsonl
uv run --locked --group analysis python research/audit/plot_heldout_rank.py research/audit/results/preview-1-heldout-rank.jsonl research/audit/figures/heldout-rank
```

Numerical execution in this session was about 12 minutes elapsed, with at most two processes. Each full replay covers 50,000 updates per width (w512 takes about 3.5 minutes).
