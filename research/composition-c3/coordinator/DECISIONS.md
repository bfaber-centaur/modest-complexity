# Composition follow-ups c3: decisions

Bobby's packet is in [charter/](charter/), verbatim (uploaded 2026-10-22 04:57Z). This log supersedes c2's assignments and budgets for c3 only; every c2 and r1 artifact stays as it is.

## D1. Base, branch, directory, parent

- **Gate.** c3 began after c2's review was complete and its handoff pinned: c2's reviewer passed every lane and the synthesis (c2 `review/REVIEW.md`, PRs #50, #53, #56–#60), and the draft integration PR #61 to `swarm/exploration` is open for Bobby.
- **Branch.** `swarm/composition-c3`, created from `swarm/composition-c2` at `7fe75e6`, the head of c2's draft integration PR. This packet authorizes no c2 merge, and none was made.
- **Directory.** `research/composition-c3/`, with `common/`, `coordinator/`, and one directory per lane: `correction/`, `scaffolding/`, `review/`.
- **PRs.** Worker PRs target `swarm/composition-c3`.
- **Final PR.** A draft to the actual parent for Bobby, which the coordinator does not merge. The parent is `swarm/composition-c2` if c2's PR is still open at the end. If Bobby has merged c2 into `swarm/exploration` by then, the parent is `swarm/exploration`, after merging it into c3.

## D2. What reviewed c2 evidence says about the two questions

Sources: c2 `coordinator/SYNTHESIS.md` and the lane NOTEs, with `review/REVIEW.md`. Nothing in c2 refutes either premise, so both lanes run as the packet specifies. Points that sharpen or bound them:

**Correction.**
- **The motivating result holds** (c2 mechanisms O4/O5; reviewer PASS on numbers, CONCERN M1 adopted). Replacing the depth-2 state of held-out programs with the training class mean for its true symbol, then applying op 3, scores as follows:

  | arm | class mean | natural state |
  |---|---|---|
  | per-step readout | 1.00 on all 32 (symbol, op 3) pairs | 0.52–0.63 |
  | tied/final | 0.88 / 0.88 / 0.97 | 0.27–0.32 |

  Shrinking the residual raises accuracy monotonically.
- **The overlap is total at depth 2** (checked from c2task, D6). All 128 held-out (start, op1, op2) states occur in training, and those 128 are every depth-2 state a length-3 program can reach: 16 op prefixes × 8 starts, each prefix with 3 training suffixes and 1 held-out suffix. So on the main evaluation, any decoder that fits training h2 exactly reproduces the oracle arm E. In the per-step-readout arm, the network's own V2 already reads training h2 at 1.0. An adapter-held-out prefix in a grouped split is still a state the base network was trained on.
- **What part of the class-mean gain is the symbol.** Reviewer CX-M1: at cell (2, 2), nearest-centroid donors recover 10–59% of the gain.
- **The weak arms improve with time** (c2 time). The per-step readout closes by 50k; tied/final reaches 0.90–0.96 by 100k and is still rising. So a correction at 10k is a finite-time rescue. The packet's 10k point stands: c2 found no correctness defect (D10 is harmless; C1 was remedied by procedure, D9).

**Scaffolding.**
- **Both reference arms already exist exactly.**
  - Keep: r1 tied/inter 0–10k, then `c2-time-tt-inter-*` 10k–50k, with checkpoints at 1k, 10k and 20k–50k.
  - Final-only from init: r1 tied/final 0–10k, then `c2-time-tt-final-*` 10k–100k.
  - The joins and recipes are exact (c2 reviewer §2).
- **Optimizer regime** (c2 reviewer T1). In every c2 arm, median Adam √v̂ falls below eps at about 24–25k, and every coordinate by about 28–30k. Branches that run to 50k enter that regime. Dropping the intermediate losses changes the gradient scale, so it may move the onset. Branches log `--diag`.
- **Length generalization was still forming at 10k** (c2 generalization, reviewer G6). tied/inter's length-6 accuracy is 0.56–0.86 at 10k, and the decline is gone by 20k. So a switch at 10k tests acquisition of length generalization without intermediate labels, as well as retention at length 3.
- **High length-3 accuracy without length transfer is a known outcome** (c2 reviewer T2/CX-T4). Final-only from init reaches 0.90–0.96 held-out at length 3 by 100k, with lengths 4–6 at 0.06–0.14 and the readout reading h1 and h2 at chance. A branch can therefore keep its length-3 score and lose length transfer, so lengths 4–6 are evaluated at the same fixed times in every arm.

## D3. Shared code (coordinator-owned) and pin

[`common/`](../common/), importing c2's pinned `common/` (`b3e8132`) unchanged:

| file | what it is |
|---|---|
| `c3train.py` | c2train.py plus a step-weight schedule: switch step S, after-weights, taper D. On `--resume` it checks the source's config, manifest and recipe (task, scheme, width, lr, wd, Adam constants, and the step weights of every earlier update). |
| `c3correct.py` | `corrected_forward`: run the frozen base to depth d, apply a correction to h_d alone, and continue the unchanged steps. Also `frozen` (write-protected parameters) and `params_sha256`. |
| `test_c3.py` | 8 tests (run in its own process): schedule boundaries and label exposure; gradient linear in the weights; r1/c2 recipe parsing; no-change mode equals c2train bitwise; drop branch equals c2train final-only from the same state; mid-taper resume bitwise; recovered r1 checkpoints resume only under r1's recipe; identity correction equals the plain forward, and the base stays frozen. |

- **Verification.** `c3-co-verify-keep-s51200-a1` resumes r1's tied/inter `ckpt-1000` to 3,000 in no-change mode. Every r1 key equals r1's published row at all 41 logged steps.
- **Pin.** `be41d16`: `common/*.py` as of that commit, unchanged on the branch head. The reviewer checks it before any worker training or fitting. Later coordinator commits change only `coordinator/` and docs unless this log records a re-pin.
- **Not in the pin.**
  - An optimizer reset. A follow-up that needs one gets a coordinator re-pin and new attempt IDs.
  - The correction module and its fitting, which are lane code.

## D4. Inventory of states

Hashes are in c2's `recovered/RECOVERY.json` and each run's `MANIFEST.json`. No checkpoint is selected by its outcome: each lane uses every seed of the arms named below.

| source | states | location (c2) |
|---|---|---|
| r1 tied/inter, tied/final, per-step-readout/inter, seeds 51200–51202 | 0, 1k, 10k | `recovered/` |
| c2 time continuations (all nine) | 20k, 30k, 40k, 50k | `time/runs/` |
| c2 time F1 (tied/final ×3) | 75k, 100k | `time/runs/` |
| c2 sharing UT and UU ×3 | 10k | `sharing/runs/` |
| c2 generalization new tasks (16) | 10k | `generalization/runs/` |

- **Correction:** `r1-rev-comp-rnn-untied-*` and `r1-comp-rnn-final-*` `ckpt-10000`, all three seeds.
- **Scaffolding:** branches from `r1-comp-rnn-inter-*` `ckpt-1000` and `ckpt-10000`, all three seeds.

## D5. Scaffolding constants (fixed before any c3 result; pilot-informed)

**Switch times.** S = 1,000 and S = 10,000, the same for every seed. These are pilot-informed choices, made from r1's logged records before any c3 measurement:

| seed | training at 1k (all three steps) | held-out at 1k (N / ¬N) | held-out at 10k (¬N) | label at 10k |
|---|---|---|---|---|
| 51200 | 1.0 | 0.34 (0.50 / 0.00) | 1.000 (1.00) | near-perfect |
| 51201 | 1.0 | 0.35 (0.52 / 0.00) | 0.961 (0.88) | partial |
| 51202 | 1.0 | 0.36 (0.52 / 0.02) | 1.000 (1.00) | near-perfect |

At 1k every seed is partial.

**Arms**, for each seed and each S, all branching from the same full-state checkpoint:

| arm | what it is | source |
|---|---|---|
| A, keep [1, 1, 1] | the existing r1 + c2 trajectory | reused, not rerun |
| B, drop | `--switch-step S --switch-weights 0 0 1` (taper 1) | new run |
| C, taper | the same, with `--taper-updates 2000` | new run |
| D, final-only from init | the existing r1 + c2 tied/final trajectory | reused |

- In arm C, the update at S + k uses intermediate weight 1 − (k+1)/2000, which reaches 0 at the update S + 1999.
- Each switch checkpoint is also evaluated with no further training.

**Run constants.**
- **Horizon.** 50,000 total updates for every branch.
- **Optimizer.** Moments and step are carried in every primary branch, and the final weight stays exactly 1. Total loss scale and label exposure are in each run's `config.json` (`exposure.labels_*`).
- **Size.** 12 new runs: 6 × 49k and 6 × 40k updates, 534k in all.
- **Checkpoints and logging.** Checkpoints at fixed steps, the same in every arm, declared in QUESTION.md. `--diag` is on.

## D6. Correction constants

- **Base.** The D4 checkpoints, frozen, with a parameter hash recorded before and after every fit.
- **Correction point.** One correction at depth 2, then the unchanged step 3.
- **Centroids.** Fit on training states only, with each unique (start, op1, op2) state weighted once rather than once per suffix (each appears with 3 training suffixes).
- **Map.** A small map with a nonnegative output. Its dimensions and optimizer are fixed after a training-only calibration and declared in QUESTION.md. Arms B (centroid target) and C (reconstruction) use the same architecture, initial draw and update exposure.
- **Evaluations.** Two, both required:
  1. **The original held-out programs.** These are seen depth-2 states followed by a new suffix.
  2. **A grouped-prefix split for the adapter's own fit, validation and evaluation.** Identical prefixes stay together, and symbol coverage is balanced using training labels. These are adapter-held-out prefixes, not unseen states (D2).

  The base network's split is unchanged.

## D7. Run IDs and seeds

| lane | ID prefix | new seeds |
|---|---|---|
| coordinator | `c3-co-` | 70000–70999 |
| correction | `c3-corr-` | 71000–71999 |
| scaffolding | `c3-scaf-` | 72000–72999 |
| reviewer | `c3-rev-` | 73000–73999 |

- The original init seeds 51200–51202 are reused on purpose for pairing. They are not new seeds.
- Calibration uses 900200 and up, with the lane in the run ID.
- IDs are never reused. A retry gets a new `-aN` suffix, and failed or capped attempts stay in the registry.

## D8. Calibration and budgets

**Calibration on the coordinator host** (`c3-co-calib-{keep,taper}-s900200-a1`): 2,000 updates with `--diag` and evaluation every 50, two concurrent jobs, one BLAS thread each. These took 16.1–16.6 s, about 8 ms per update including evaluation, at a peak RSS of 46 MiB. c2's time host ran 4–6 ms per update. Each lane calibrates on its own host first and reports ms per update before launching.

**Budgets.** These are elapsed wall time with at least one job running, using `run_jobs.py --workers 2`. They are caps, not targets.

| lane | initial | follow-up | sized for |
|---|---|---|---|
| scaffolding | 40 min | 20 min | 534k updates (about 22–36 min at 5–8 ms per update, two workers); a matched reset-vs-carried follow-up in both keep and drop arms at one switch |
| correction | 10 min | 5 min | small maps on 384 rows × 6 checkpoints; frozen evaluation |
| reviewer | 20 min | | independent recomputation and one counterexample per lane |
| coordinator | 10 min | | about 1 min used (tests, verification, calibration) |

A lane that reaches 80% of a budget tells the coordinator before going further.

## D9. Gates and reporting

- **Before any result.** QUESTION.md comes first, with exact interventions, controls, predictions, evaluation sets and the first figure. Adaptive choices get dated entries in ADAPTIVE.md.
- **Training.** No worker training or fitting before the reviewer's PASS on the pin, which the coordinator relays.
- **First figure.** One primary figure comes before any expansion.
- **Follow-up.** One per investigator, proposed in QUESTION.md and approved by the coordinator.
- **Reviewer.** Commits at least one counterexample prediction per lane before inspecting that lane's result tables.
- **Milestones.** Reported in one line to the coordinator thread "Four-regime expedition".
- **Final packet.** `coordinator/SYNTHESIS.md` covers what an oracle-free correction achieved, what temporary supervision left behind, and the limitations, with at most two next questions.
