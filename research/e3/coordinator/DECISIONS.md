# e3 decision log

Charter: [charter/](charter/README.md). Launch messages: [DISPATCH.md](DISPATCH.md). Run claims: [../plan.json](../plan.json). Newest entries last; earlier entries are not edited, only superseded by later ones.

## D1 (2026-10-22) Open e3, keep e2 as it is

e2 closed at `swarm/exploration` `4783a47` (e2 D11), which was the branch head when e3 setup began. e1 and e2 artifacts, notes and run directories stay where they are and remain reusable; e2 documents get a dated pointer, not rewrites. "Simplicity" is retired as the organizing hypothesis (charter). e3 asks two questions: does the data-size pattern survive changing the training examples (dynamics), and what does training do to a network after the frequency-filter edit (mechanisms). The auditor first measures operand-label preference with scale-free pairwise rankings on existing artifacts, then checks both experiments.

The `walkthrough/rescaling` branch (`b22f7ce`, unmerged) is unrelated walkthrough work and stays separate.

## D2 (2026-10-22) Sessions and ownership

The existing threads keep their e2 roles. There are no new threads.

| e3 role | session (thread) | owned directory | branch | registry / session label |
|---|---|---|---|---|
| coordinator | "Exploratory swarm launch" | `research/e3/coordinator/`, `research/e3/plan.json`, `research/e3/data/`, shared code | `swarm/exploration` | `research/e3/coordinator/registry.jsonl` / `e3-co-cloud-1` |
| dynamics explorer | "Width and training-time map" | `research/e3/dynamics/` | `claude/project-thread-j8mgkm` | `research/e3/dynamics/registry.jsonl` / `e3-dy-cloud-1` |
| mechanisms explorer | "Frequency retention and rescaling" | `research/e3/mechanisms/` | `claude/project-thread-psralm` | `research/e3/mechanisms/registry.jsonl` / `e3-me-cloud-1` |
| independent auditor | "Correctness audit" | `research/e3/audit/` | `claude/project-thread-id1tmm` | `research/e3/audit/registry.jsonl` / `e3-au-cloud-1` |

All three workers had pushed a QUESTION.md before this setup existed: dynamics `b800339`, mechanisms `5bfa578` and audit `af9c01e`. Each was written from e2 results only; no e3 run had been made. The mechanisms lane also ran a label-free feasibility check of its sham rule on weights alone, and its QUESTION.md describes it.

## D3 (2026-10-22) Pinned base and shared code

- The e3 base for all workers is the `swarm/exploration` commit that adds this file. Its hash is in [INDEX.md](INDEX.md) and in the launch messages. A later `swarm/exploration` commit is equally valid as a base while it changes no shared code.
- Shared code is `experiment.py`, `launch_runs.py`, `report_map.py`, `publish_run.py`, `instrumentation.py`, `pyproject.toml`, `uv.lock` and the shared tests. Its identity for e3 is that same commit. The only changes from `fe284a5` are these:
  1. `data()` raises `ValueError` when any split is empty, closing e1 audit Part 4 note F10. Splits that are not empty, which includes every historical split, are unchanged: `test_historical_data_unchanged` and `test_first_map_split` still pass.
  2. `--init-weights FILE.npz` with `--expect-init-sha256 HASH` starts from given weights with a fresh optimizer, and `--init-age N` is recorded only. D5 has the details.
  3. `--checkpoint-steps 10,30,100,300` adds checkpoints at listed update counts. Each must be a multiple of `--log-every`.
  4. `launch_runs.py` passes `init_weights`, `init_weights_sha256`, `init_age` and `checkpoint_steps` from a plan entry when present.
  5. Two new tests: `test_continuation_from_given_weights_restarts_the_optimizer` and `test_continuation_refuses_a_different_weights_file`, plus `test_empty_split_is_rejected`.

  `uv run --locked python -m unittest test_exploration test_instrumentation` gives 13 tests OK, and `experiment.py --selftest` passes. Without the new flags a run takes exactly the same path as before.
- The mechanisms lane proposed `--start-weights` and `--start-age`. They are adopted under the names `--init-weights` and `--init-age`. Any further shared-code change is proposed as a patch on a lane branch, gets a new pinned hash here, and the runs that depend on it wait for that hash.

## D4 (2026-10-22) Dynamics initial experiment: three new ladders

**Seeds.** The data seeds are 41001, 41002 and 41003. No run in e1 or e2 uses them: the only "4100x" strings in the tree are inside hex hashes, and 31001 appears only in a test. The initialization seeds are 30000 and 30001. `init()` does not depend on the data seed, so each init seed gives identical starting weights in every ladder and at every share. The control seed is 32000 and the diagnostic seed 33000, both unused by `--diagnostics none` and the addition task.

**Crossed design.** The existing data-seed-31000 ladder already holds width-128 runs with inits 30000 and 30001 at all four shares, so it is a fourth ladder that needs no new runs:

| share | runs |
|---|---|
| 30% | `e1-co-add-w128-i30000-a1`, `e1-st-add-w128-i30001-a1` |
| 40% and 45% | `e2-dy-add-w128-i3000{0,1}-tf{40,45}-a1` |
| 50% | `e1f2-co-add-w128-i3000{0,1}-tf50-a1` |

It was seen before the e3 question was written, and its checkpoint cadence is sparser, so it is shown as the e2 reference and not counted as a fourth prospective ladder.

**Splits.** These were checked by the coordinator with `experiment.data` at the e3 code, using [ladders.py](ladders.py), whose output is [ladders.json](ladders.json). For every data seed:
- the 30% ⊂ 40% ⊂ 45% ⊂ 50% training sets nest;
- every split is transpose-closed;
- train, cal and audit partition the 961 pairs.

Actual training-pair counts are not the nominal shares, because the diagonal group of each class holds one ordered pair instead of two:

| data seed | 30% | 40% | 45% | 50% | evaluation pairs | evaluation SHA-256 |
|---|---|---|---|---|---|---|
| 31000 (e2 reference) | 299 | 359 | 417 | 478 | 483 | `1535a21a…` (= e2 common set) |
| 41001 | 303 | 363 | 422 | 482 | 479 | `59086f28…` |
| 41002 | 303 | 363 | 420 | 481 | 480 | `d5a7c820…` |
| 41003 | 299 | 360 | 421 | 480 | 481 | `2d2d8748…` |

**Evaluation pairs.** A ladder's evaluation pairs are the pairs outside its 50% training set, which is the 50% split's cal ∪ audit. They are untrained at every share of that ladder. Each set has 248 unordered groups, is transpose-closed, and has 15–16 pairs per class. They are saved as `research/e3/data/eval-d<seed>.npz`, with key `index` indexing the `data()` arrays. Ladders have different evaluation pairs and are shown separately; pairs within a ladder share operands and labels, so they are not independent replicates.

**Runs.** There are 24 runs, `e3-dy-add-w128-i<init>-tf<share>-d<data seed>-a1`, at width 128 for 50,000 updates with the e1 recipe (AdamW lr 1e-3, wd 1, full batch, float64, `--diagnostics none`, metrics every 100 updates, checkpoints every 1,000). Each plan entry carries its full split hash and actual training count. The execution order is the plan's `queue_position`, a seeded permutation (`numpy.random.default_rng(20261022)`) that interleaves ladders, shares and init seeds. The lane may reorder for practical reasons if it records why. Launch: `launch_runs.py --plan research/e3/plan.json --registry research/e3/dynamics/registry.jsonl --session-label e3-dy-cloud-1 --workers 2 --timeout 600`.

## D5 (2026-10-22) Mechanisms initial experiment: continuing from edited weights

**Sources.** The three failed 30% width-128 runs at 50,000 updates, all on split `aa8f7504…1f24` with 299 training pairs:

| source run | checkpoint SHA-256 |
|---|---|
| `e1-co-add-w128-i30000-a1` | `326cfc1e…11b4` |
| `e1-st-add-w128-i30001-a1` | `a96bc6e2…45ad5` |
| `e1-st-add-w128-i30002-a1` | `cb7dd55b…3b62` |

The full hashes are in the plan entries.

**Arms.**
- A keeps U, V and W unchanged.
- B applies the e2 filter `filter_units(w, dominant(w)[0])` from `research/e2/mechanisms/unitfilter.py`, unchanged.
- C is the non-dominant-frequency sham of `research/e3/mechanisms/QUESTION.md` (`5bfa578`). Each unit keeps non-dominant frequencies in a seeded random order until it holds λ times its dominant input energy. One global λ is bisected so that total kept input energy equals B's. The sham seeds are 37000, 37001 and 37002, which no earlier run uses.

The rule is label-free and fixed before any held-out evaluation, so it is accepted. It matches total input energy and input displacement (0.74–0.76 for B and C), but it does not match:
- W displacement (C 0.71–0.75, B 0.78–0.80);
- per-unit displacement in the 37–46 units whose dominant frequency outweighs the rest;
- gate flips;
- immediate damage.

C is therefore called "input-energy-matched", not "matched", and all these mismatches are reported at update 0. The e2 single-random-frequency null (displacement about 0.97) is not added to the initial nine. It remains a possible follow-up.

**Continuation implementation.** The implementation is the smallest one that does the job, and its contract is explicit.
- `--init-weights FILE.npz` loads U, V and W in place of `init()`. The keys must be exactly those, float64, with the run's shapes.
- `--expect-init-sha256` stops the run before training if the file's hash differs; `launch_runs.py` always passes it from the plan. `config.json` records `init_weights_sha256` and an `init_rule`.
- Adam moments m and v start at zero, and the update count restarts at 1. This is the same loop and bias correction as a fresh run, so update 1 has t = 1.
- `--init-seed` is still required by the command line but does not set any weights. The plan gives the source's seed for the record.
- **Checkpoint age and continuation updates are kept apart.** The age (50,000) is in the plan entry (`init_age`, `source_run`, `source_checkpoint_sha256`) and in `config.json` (`init_age`). Every `step` in metrics and checkpoints counts continuation updates from 0.
- **Post-edit evaluation.** The step-0 metrics row and `checkpoint_000000.npz` are written from the loaded weights before any update, so every branch has a pretraining evaluation of its edit.
- Tests check five things:
  - the step-0 weights equal the start file;
  - the step-0 moments are zero;
  - step-0 train accuracy equals a forward pass of the loaded weights;
  - 100 updates equal a manual replay with zero moments and t = 1…100;
  - those updates differ from the uninterrupted trajectory.

  A wrong hash fails before any checkpoint is written.
- Smoke check on this host: continuing A for `e1-co-add-w128-i30000-a1` reproduces the e1 step-50,000 row exactly at step 0 (train CE 3.650925e-06, norm 60.3379365). 2,000 updates with evaluation every 10 take 5.2 s.
- Weight decay, data, targets, learning rate and schedule are unchanged. There is no schedule, so the only state reset is Adam's.

**Cadence.** Metrics are logged every 10 updates for 20,000 updates. Checkpoints are written every 1,000 updates plus at 10, 30, 100 and 300; update 0 is always written.

**Start files and gate.** The lane's edit script writes `research/e3/mechanisms/starts/<run_id>.npz` with `np.savez`, keys U, V and W only. For A the arrays are byte-identical to the source checkpoint. The script also writes an update-0 report containing:
- displacement and kept energy per matrix;
- gate flips over the 961 pairs;
- RMS pair-centred logit change;
- train loss and accuracy;
- accuracy on the 483-pair common set.

The lane pushes these files and sends the nine SHA-256 values. The auditor checks them against its charter list: identical sources, A unchanged, B reproducing e2, no held-out data in C, and the continuation code. After that check the coordinator fills in `init_weights_sha256` and moves the nine entries from `reserved` to `runs` in the plan. `launch_runs.py` refuses an ID until then.

The nine IDs are `e3-me-<keep|filt|sham>-w128-i<30000|30001|30002>-a1`. Launch: `launch_runs.py --plan research/e3/plan.json --registry research/e3/mechanisms/registry.jsonl --session-label e3-me-cloud-1 --workers 2 --timeout 300`.

## D6 (2026-10-22) Shared measures, and empty or singleton sets

**Definitions.** All lanes use the auditor's [`research/e3/audit/pairwise.py`](../audit/pairwise.py) from PR #22:
- `preference(z, A, B)`, the per-pair P(A over B) with ties counted as 1/2, which is NaN when a set is empty;
- `ladder_sets`, the S, D and N sets of a nested ladder.

Each lane reports:
- P(S over N);
- P(a+b over N);
- P(a+b over S);
- accuracy and cross-entropy on the ladder's evaluation pairs;
- the margin of a+b over N in units of the pair's wrong-label logit standard deviation;
- the max-based diagnostics alongside.

Summaries average pairs with equal weight, and pairs with an undefined value are dropped and counted. Raw logit differences are not read as growing preference.

**Empty and singleton sets, checked before any e3 run** ([ladders.json](ladders.json)). On every e3 ladder's evaluation pairs, N under the 50% mask (smallest N, a+b excluded) has at least 1 label (31000, 41001) or 2 (41002, 41003). N is never empty, and it has 2 singletons in each of 31000 and 41001. S under the 30% mask has 4–6 labels at minimum. D can be empty, so P(D over ·) drops and counts. A singleton N gives a per-pair value of 0, ½ or 1; it is defined, but the lanes report the count.

The divide-by-zero in `research/audit/heldout_rank.py` and the mechanisms `relational.py` (e2 audit G3) needs a pair whose a+b is its only absent label. That does not occur on any e3 evaluation set. Those files stay as the e1 and e2 record. An e3 use of them on any other set must first add the drop-and-count patch that `offset_profile.py` uses. The dynamics lane's `ladder.py` (`27b034b`) already drops and counts.

## D7 (2026-10-22) Execution budget (e3, calibrated; not inherited)

Calibration was done on the coordinator host: 4 CPUs, two concurrent processes, one BLAS thread, data seed 41001 at width 128.
- 5,000 updates take 11.8 s at 30% and 16.8 s at 50%, so 50,000 updates take about 120–170 s.
- In e2 the dynamics host measured 137–151 s at 40–45%, and the mechanisms host 72–140 s per 50k.
- A 20,000-update continuation with evaluation every 10 updates takes about 52 s.

Each session's budget below is cumulative elapsed numerical execution counted from its e3 start. Each session runs at most two processes with one BLAS thread each, on CPU in these containers only, with no paid compute. e1 and e2 usage are reported separately.

| session | budget | covers |
|---|---|---|
| dynamics | 70 min | initial 24 runs ≈ 57 process-min ≈ 30 min elapsed with two workers, timeout 600 s each; analysis; a claimed follow-up of at most 25 min elapsed |
| mechanisms | 25 min | edits and update-0 reports; nine 20k continuations ≈ 8 process-min ≈ 5 min elapsed, timeout 300 s each; analysis; a claimed follow-up of at most 10 min |
| auditor | 30 min | pairwise preference on existing artifacts (about 0.5 min used); dynamics and mechanisms checks including short replays |
| coordinator | 15 min | calibration (about 1 min used), split checks, spot checks |

A lane that projects a launch past its remaining budget stops and asks; it never launches over budget. Capped runs are kept and reported. A follow-up claim states its expected minutes.

## D8 (2026-10-22) Workflow and claims

- Each explorer gets one initial experiment, D4 or D5, and one adaptive follow-up. The coordinator reviews each explorer's first figure before approving the follow-up. A follow-up claim names what it distinguishes, lists its run entries (ID, fields that differ from `common`, reason, expected minutes), and is pushed as `research/e3/<lane>/REQUEST.md` or sent through the thread. Runs launch only from a pushed plan commit that lists them. A follow-up that needs shared code waits for a new pin (D3).
- Run IDs follow `e3-<co|dy|me|au>-<condition>-w<width>-i<init>[-tf<share>][-d<data seed>]-a<attempt>`. IDs are never reused, outputs are never overwritten, and failures are never omitted. A replay is a reproducibility check, not a new seed. Local outputs go under `runs/e3/`. `publish_run.py` publishes under `research/e3/<lane>/runs/<run_id>/`.
- Workers merge `swarm/exploration` into their designated branches. PRs target `swarm/exploration`, and the coordinator merges them after checks. Nothing is merged upstream or into either main branch.
- The context zip's papers and glossary stay uncommitted (e2 charter README).

## D9 (2026-10-22) Departures from e2, in one place

| e2 | e3 | where |
|---|---|---|
| one data seed (31000) | data seeds 41001–41003 added; 31000 kept as the reference ladder | D4 |
| init seeds 30000–30002 | 30000 and 30001 for the dynamics ladders; the three 30% runs (30000–30002) as mechanisms sources | D4, D5 |
| common set: 483 pairs of the 31000 ladder | one evaluation set per ladder (479–483 pairs); 31000's equals the e2 set | D4 |
| training from `init()` only | `--init-weights` continuation, fresh Adam, recorded age | D3, D5 |
| checkpoints every 1,000 | adds `--checkpoint-steps` for dense early checkpoints | D3, D5 |
| empty split possible at small p | `data()` raises | D3 |
| code identity `fe284a5` | the e3 base commit | D3 |
| 30 min per session | 70 / 25 / 30 / 15 min, calibrated | D7 |
| two inquiry cycles per explorer | one initial experiment and one adaptive follow-up, with first figure reviewed | D8 |
| restricted rank and σ-unit gap as the main normalized measures | pairwise P(S over N), P(a+b over N), P(a+b over S) as primary; max-based measures kept | D6 |

## D10 (2026-10-22) Mechanisms runs activated

The lane pushed the nine start files and the update-0 report on `claude/project-thread-psralm` at `42ca626` (`research/e3/mechanisms/starts/`, written by `edits.py` at `371ed43`). The coordinator recomputed the nine SHA-256 values; they match the lane's list. A's arrays are identical to the source checkpoints. B's relative displacement is 0.74–0.76 for U and V and 0.78–0.80 for W; C's is 0.74–0.76 for U and V and 0.71–0.75 for W.

The audit gate passed on every item, with evidence on `claude/project-thread-id1tmm` at `a8d796d` (`research/e3/audit/results/continuation-check.json` and `start-check.json`):
- sources;
- A unchanged;
- B equal to the e2 `filter_units`;
- C weights-only and rebuilt independently;
- optimizer reset and recipe: a 300-update arm-A continuation matches an independent zero-moment AdamW replay to 3.6e-11;
- post-edit row;
- visible limits.

The nine entries moved from `reserved` to `runs` with `init_weights_sha256` filled in and `activated` set. The start files reach `swarm/exploration` with the lane's PR.

## D11 (2026-10-22) Mechanisms first figure reviewed; follow-up claimed (weight decay 0)

PR #24 was integrated at `d0a0657`; it is not yet audited. The coordinator spot-checked it with separate forward code on published checkpoints and matched `results/checkpoints.csv`, and the update-0 moments are zero. Reviewed figures: `research/e3/mechanisms/figures/continuation.png` and `structure.png`.

**Observations, all three seeds alike:**
- A does not move under the reset.
- B rises from 0.43–0.58 at update 0 to 0.74–0.86 around updates 70–100, while train accuracy returns to 1.
- B then falls to 0.22–0.27 by 1,000 updates and to near chance by 5,000.
- During the fall B's P(S over N) climbs back to about 0.94, A's level.
- B's P(a+b over N) stays near 0.87 against A's 0.70–0.74.
- 90–92% of B's units keep their update-0 dominant frequency.
- C stays at chance on held-out pairs, and its removed frequency partly regrows.

Two parts of the rise and fall are separate: the edit supplied the update-0 level, and the rise after the edit came from training. "Erasure" is an observation of the held-out readout, not evidence about a hidden algorithm.

**Follow-up approved, the lane's proposal.** The question is whether decay or the cross-entropy drive on 299 pairs removes B's held-out gain. The fall happens between about 300 and 2,000 updates, which overlaps both the decay timescale 1/(lr·wd) = 1,000 updates and the phase in which train CE falls from about 0.1 to about 1e-3. Design:
- Continue the same B and C start files, with the same hashes and already gated, at `--wd 0`.
- Everything else is unchanged.
- Six runs, `e3-me-<filt|sham>-wd0-w128-i<30000|30001|30002>-a1`.
- About 2 minutes elapsed, within the lane's follow-up allowance (D7).

The lane writes its expectation in QUESTION.md before launching. A is not repeated at wd 0, because the converged memorizer's held-out accuracy is not in question. Optimizer-state retention is left as a possible later question.

## D12 (2026-10-22) Mechanisms follow-up integrated; lane's cycles used

PR #25 was integrated. The lane pushed its expectation at `ad6e923` before running, and all six weight-decay-0 runs completed. The coordinator spot-checked one B and one C run at updates 0, 1,000 and 20,000 with separate forward code, and the configs record wd 0, the gated hashes and age 50,000. Neither packet is audited yet.

**Observation.** From the same start files:
- At wd 0, B keeps 0.69–0.81 common-set accuracy at 1,000 updates and 0.56–0.62 at 20,000. At wd 1 the same branches are at 0.22–0.27 and 0.01–0.03.
- The early rise is the same at both settings.
- At wd 0 the fit runs about 1,000× further (train cross-entropy about 3e-8), and the weight norm doubles.
- C stays at chance at both settings.

**Reading, labelled as such.** Within 20,000 updates, decay is necessary for the fast erasure. The cross-entropy drive alone erodes the gain only slowly, by 0.20–0.25 from the peak. Why decay erases it is open. One candidate explanation, untested: under decay at this training share, the operand-label memorizer is the cheaper solution.

The mechanisms lane has used its initial experiment and its one follow-up (D8). Its budget is about 5.5 of 25 minutes.

## D13 (2026-10-22) Audit of mechanisms packet 1 integrated; one audit probe approved

PR #23 was integrated. It contains audit investigations 2 and 3: the gate, the dynamics splits, and mechanisms packet 1.

**Packet 1 software checks: PASS.** All 104 checkpoints were recomputed, and the replays agree within 5e-11.

**Packet 1 interpretations:**
- The rise and fall of B are robust. At one-update resolution the peak is 0.743, 0.797 and 0.872, reached 18–37 updates after train accuracy first reaches 1.
- The edit supplies 0.43–0.58. Training then adds 0.26–0.31 and removes it again in accuracy, but not in ranks: at 20k, P(a+b over S) is 0.39–0.45 for B against 0.14–0.18 for A and C.
- "Return toward the memorizer" is CONTRADICTED as a return to A. B's argmax agreement with A is 0.42–0.50, and its distance from A grows.
- It is SUPPORTED LOCALLY as B becoming a fitted network that prefers operand labels again, with P(S over N) at 0.92–0.93.

**Audit probe approved.** The probe asks whether B's early rise comes from fitting frequency-concentrated features or from fresh Adam's first steps of size about ±lr. It replays B's first 300 updates from the same three start files at lr 1e-4, and separately with plain gradient descent.
- The probe uses the auditor's own code, takes a few seconds per replay, and counts against the audit budget (D7).
- These are diagnostic replays, not plan runs, so they get no run IDs. Their outputs and the decision rule, stated before running, go under `research/e3/audit/`.
- They are listed in the run accounting as audit probes.

## D14 (2026-10-22) Dynamics first figure reviewed

PR #28 was integrated. All 24 runs completed in 28 min elapsed, from a clean worktree containing the pin. The coordinator spot-checked three runs with separate forward code and matched Table 1. Split hashes and counts equal the plan. Not yet audited.

**Reviewed figures:** `research/e3/dynamics/figures/ladders_accuracy.png`, `ladders_trajectories.png` and `ladders_measures.png`.

**Observations:**
- In all 8 ladder × init sequences, every measure improves with each larger nested set.
- Accuracy falls in the same bands as on the 31000 ladder: 0.000–0.004 at 30%, 0.010–0.092 at 40%, 0.40–0.54 at 45% and 0.88–0.91 at 50%.
- The interpolated accuracy-0.5 crossings at 50k are 417–434 pairs.
- At 45%, the spread between ladders (0.076) is smaller than the largest init difference within a ladder (0.142).
- At 40%, 41002 leads on the σ-margin for both inits, but not on accuracy.
- The 45% runs are still rising at 50k.

These observations fit the lane's pre-stated pattern "ordering and location both travel", at four rungs, two inits, one width and one horizon.

**Follow-up.** Per the charter, if the ladders agree the follow-up is a small, explicitly selected continuation of intermediate runs. The lane proposes the selection and what it distinguishes. Two implementations are possible:
- An uninterrupted continuation past 50k needs a resume with the saved Adam moments and update count. That is a shared-code change, which requires a test and a new pin first (D3).
- A longer run from initialization needs no code change, and its first 50k replays the claimed run.

Either must fit the lane's remaining budget, about 41 min, with the follow-up limited to 25 min (D7).

## D15 (2026-10-22) Dynamics follow-up: resume past 50k (shared code re-pinned)

The follow-up is the lane's proposal, research/e3/dynamics/QUESTION.md Q2 (`cd12d03`), written before any follow-up run. It continues runs past 50,000 updates to separate the 50k step location and init gaps from timing, and to see whether 41002's graded lead at 40% becomes an accuracy lead.

**Runs.** The selection was fixed before any continuation.
- **45%:** all six e3 runs, continued to 150,000 updates.
- **40%:** the two ladders with the highest and lowest mean σ-margin at 50k (41002 and 41001), both inits, continued to 100,000 updates.
- IDs follow `e3-dy-add-w128-i<init>-tf<share>-d<seed>-c<final updates>k-a1`, with queue positions 25–34 (seeded, `default_rng(20261023)`).
- Expected time is about 19 min elapsed with two workers, inside the 25-minute follow-up allowance (D7).

**Shared-code change (new pin).** `--resume CHECKPOINT.npz` loads U, V and W, the Adam moments and the update count from a checkpoint, and continues training uninterrupted. `--steps` is the final update count, and step numbers continue the source's numbering.
- `--expect-init-sha256` now checks whichever start file is given, `--init-weights` or `--resume`.
- `config.json` records `resume_sha256`.
- `--resume` excludes `--init-weights` and requires `--diagnostics none`, because the inline readout diagnostic compares against initial weights, which a resumed run does not have.
- `launch_runs.py` passes `resume` and `resume_sha256` from the plan.

**Checks.**
- New test `test_resume_continues_training_exactly`: resuming a short run at update 100 reproduces its checkpoints at 200 and 300 bitwise (weights and moments) and its metrics rows except wall-clock fields. The resumed step logs no optimizer update.
- 14 tests OK, and the selftest passes.
- On this host, resuming `e3-dy-add-w128-i30001-tf45-d41001-a1` from its published 50k checkpoint reproduces that run's step-50,000 metrics row exactly.
- Every resume hash equals the checkpoint hash in the source manifest.
- A resume on a different host from the one that produced the source may differ in floating-point summation order. That is recorded, not prevented.

The new shared-code identity is the commit that adds this entry. The 15 mechanisms runs and the 24 dynamics runs were produced at the earlier pin. Their code paths are unchanged: the edit only adds a branch for `--resume`, and the existing tests still pass.

## D16 (2026-10-22) Dynamics follow-up integrated; launch deviation accepted

PR #30 was integrated at `3289ef1`. All ten D15 resumes completed in 19.1 min, and the lane's total is about 49 of 70 min.

The coordinator spot-checked two runs with separate forward code at 150k and 100k, and both matched Table 4. The resume and split hashes equal the plan.

**Launch deviation.** The lane launched from its main checkout instead of a clean worktree at the pin. Every run records commit `2f0aa7a`, which differs from `72cfb5d` only in `research/e3/dynamics/QUESTION.md`; the shared code, plan and lock files are identical. Each run also has a dirty status: the launcher's own registry append. The deviation was accepted rather than relaunched under `-a2`, because the runs are exact resumes whose start hashes are recorded and can be replayed. It is stated in the lane README and in [RUNS.md](RUNS.md).

**Correction.** Packet 2 withdraws packet 1's "the 45% runs are still rising at 50k" and the audit's matching rule-3 verdict, on every-100-update evidence. Both lanes have now used their initial experiment and their one follow-up.

Audit investigation 5 (PR #29, `a62e308`) found:
- software PASS;
- the ordering and the P(S over N) fall SUPPORTED LOCALLY;
- 41002's 40% lead as a training-set effect CONTRADICTED.

The audit of packet 2 is the last open item.
