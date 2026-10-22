# c2 decisions (coordinator, "Four-regime expedition" thread)

Bobby's launch packet is committed verbatim in [charter/](charter/) (`composition-round2-prompts.zip`, SHA-256 `80bf2384…4391`, uploaded 2026-10-22 03:42Z). Where the charter and this file differ, the charter wins. Every lane reads its own role file there plus this file before executing.

## D1. Base, branch, directory (2026-10-22 03:48Z)

- **Integration branch.** `swarm/composition-c2`, branched from `swarm/exploration` at `5c0129b` (the PR #49 merge).
- **Directory.** All c2 files live under `research/composition-c2/`. r1 (`research/regimes/`), e4 and everything else are read-only for c2.
- **Pull requests.**
  - Worker PRs target `swarm/composition-c2`. The coordinator merges them after the reviewer has looked.
  - The final integration PR is a draft against `swarm/exploration`, for Bobby. The coordinator never merges it.
  - Nothing goes to `main` or upstream.
- **Lane directories.**

  | lane | directory | role file |
  |---|---|---|
  | time | `time/` | `charter/02-time.txt` |
  | sharing | `sharing/` | `charter/03-sharing.txt` |
  | generalization | `generalization/` | `charter/04-generalization.txt` |
  | mechanisms | `mechanisms/` | `charter/05-mechanisms.txt` |
  | reviewer | `review/` | `charter/06-reviewer.txt` |

## D2. r1 checkpoints were recovered, not replayed

The original RNN checkpoint files were still on the two containers that wrote them. Both threads hashed them against r1's published manifests, and every file matched. The coordinator re-hashed them here.

- **What is committed.** [`recovered/`](../recovered/) holds `ckpt-0`, `ckpt-1000` and `ckpt-10000` for nine runs, with weights, Adam moments and step count. [`recovered/RECOVERY.json`](../recovered/RECOVERY.json) gives each file's SHA-256, the manifest it matched, and its source.
  - `r1-comp-rnn-inter-s5120{0,1,2}-a1`: tied transition, tied readout, intermediate supervision.
  - `r1-comp-rnn-final-s5120{0,1,2}-a1`: tied/tied, final-only.
  - `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`: tied transition, per-step readouts, intermediate supervision. This is the reviewer's r1 arm.
- **What stays only on the mount.** The r1 MLP checkpoints and the reviewer's tied replay are hash-verified copies under `/mnt/project-files/c2-recovered/`. They are not committed and c2 does not use them.
- **Cross-host evidence.**
  - The reviewer's r1 replay of `rnn-inter-s51200` was trained on a different container. Its `ckpt-10000` arrays, weights and both Adam moments included, are identical to the original's.
  - On a third host, `c2train.py` resumed each original `ckpt-1000` to 10,000 updates (`c2-co-verify-{inter,final,untiedro}-s51200-a1`). Every logged r1 metric and every array of `ckpt-10000` is identical to the original ([verify_resume.out](verify_resume.out)).
  - The `.npz` *files* still differ byte-wise, because they carry run metadata and zip timestamps. Compare arrays, not file hashes.
- **Consequence.** No replays are assigned. Lanes start from the recovered files and cite them by SHA-256. A run from them is a *continuation of the original run*, not a new seed and not a replay.

## D3. Shared-code pin

[`common/`](../common/) contains `c2task.py`, `c2rnn.py`, `c2train.py`, `c2probe.py` and `test_c2.py`. It imports r1's `task.py` under the module name `r1_comp_task`, and `research/regimes/common/nets.py` unchanged. Pin commit: `b3e8132`. The reviewer checks the pin before any new training.

**Model and recipe.** These are r1's defaults. Every one can be set, and every run's `config.json` records it.
- **Model.** Width 128, ReLU. One start embedding (`Ws`, `bs`) shared by every scheme.
- **Transition group.** `A`, `B` and `bh`, where `B` is the op embedding.
- **Readout group.** `V` and `bv`.
- **Initialization.**
  - r1's `rnn_init` draw: He-normal `Ws`, `A`, `B`; LeCun-normal `V`; zero biases.
  - An untied group becomes three independent copies of the tied draw, so all four schemes start as the same function.
- **Loss.** Sum over steps of `weight_t × mean cross-entropy` on 384 training rows (48 programs × 8 starts).
  - `--step-weights 1 1 1` is r1's `inter`. `0 0 1` is r1's `final`.
  - The weights are explicit and equal for intermediate supervision.
- **Optimizer.** Full-batch Adam: lr 1e-3, b1 0.9, b2 0.999, eps 1e-8.
- **Decay.** Decoupled weight decay of 0.1 on every parameter: biases, start and op embeddings, and every untied copy. An untied scheme therefore decays more parameters.
- **Gradients.**
  - A tied parameter receives the sum of its per-step gradient terms.
  - Untied copy k receives step k's term only.
  - Tests check that the copies' gradients sum to the tied gradient at init.

**Parameter counts** (width 128):

| transition | readout | parameters |
|---|---|---|
| tied | tied | 19,208 |
| tied | per-step | 21,272 |
| per-step | tied | 53,256 |
| per-step | per-step | 55,320 |

These are not parameter-matched and must not be described as matched.

**Evaluation.** `c2rnn.forward` runs any length, and can continue from any hidden state.
- An untied group needs an explicit `step_map` past step 3 or mid-program. Each lane states the map it uses.
- `c2train.py` logs r1's metric keys, identical to r1 in the tied/tied scheme.
- It also logs `held_subset`: held-out final accuracy on N rows (the answer is a one-op-slot training neighbour's label) and ¬N rows (42 of 128 on the original task).
- `--diag` adds gradient norm, update norm, Adam √v̂ against eps, and the train margin.

**Resume.** `--resume CKPT --resume-sha256 H` continues a run with its full optimizer state. It refuses a hash mismatch or a different scheme. Weight-only restarts are not offered.

**Tests.** `uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py` runs 10 tests in its own process, about 20 s. They cover:
- rows and IDs against r1, and longer rows against a plain loop;
- the one-function start and independent copies;
- gradient splitting, and finite differences in all four schemes with unequal step weights;
- any-length forward and continuation from a hidden state;
- probe sanity;
- the tied/tied no-change mode, which is bitwise r1 (`train.py`, and r1's published metrics);
- per-step readouts, which are bitwise the reviewer's r1 arm;
- bitwise continuation, including diagnostics, and refusal of a wrong hash.

Run every lane's tests in its own process, as the charter says.

**Import hygiene.** Lane files carry the lane prefix: `time_*.py`, `share_*.py`, `gen_*.py`, `mech_*.py`, `rev_*.py`. They reach r1's task functions through `c2task.r1`. No c2 process puts `research/regimes/composition/` on `sys.path`; run r1's `train.py` only as a subprocess.

## D4. Run IDs, seeds, claims

**Prefixes.** Each lane claims its own IDs in its own `plan.json` and registry (`research/regimes/common/run_jobs.py`, append-only). IDs are never reused; a retry gets a new `-aN` suffix.

| lane | ID prefix |
|---|---|
| time | `c2-time-` |
| sharing | `c2-share-` |
| generalization | `c2-gen-` |
| mechanisms | `c2-mech-` |
| reviewer | `c2-rev-` |
| coordinator | `c2-co-` |

**Original task and seeds.**
- Original task: `ops51000-ls51100` (task seed 51000, split seed 51100).
- Original init seeds: 51200–51202.
- Time, sharing and mechanisms reuse 51200–51202 on purpose. They pair conditions on identical initial functions and are *not* new seeds.

**New seeds.** Ranges are by role.

| lane | range |
|---|---|
| coordinator | 60000–60999 |
| time | 61000–61999 |
| sharing | 62000–62999 |
| generalization | 63000–63999 |
| mechanisms | 64000–64999 |
| reviewer | 65000–65999 |

- Inside each range: task draws x000–x099, splits x100–x199, inits x200–x299, other x300 and up (probes, pair sampling, label shuffles).
- Calibration uses seeds 900100 and up, with the run ID naming the lane.
- The generalization lane's new tasks are `ops630xx-ls631xx`. It declares the draw-acceptance criteria before drawing and logs every rejected draw.

## D5. Decoder-across-position table (sharing and mechanisms)

Both lanes use `c2probe.decode_table` unchanged.
- **Fit.** One linear softmax probe per position i in 0..3, fit on h_i of training-program rows.
- **Score.** Each probe is scored on h_j of held-out-program rows against the true symbol at j.
- **Probe recipe.** r1's: no hidden layer, Adam lr 1e-2, 1,000 updates, no decay. Use probe seed 64300 for tables that the two lanes compare.

The table is descriptive. A low off-diagonal entry does not show that the symbol is absent, and decodability is not use.

## D6. Calibration and budgets

**Coordinator's calibration.** On the coordinator host (`c2-co-calib-*`, 2,000 updates, two concurrent jobs, width 128, one BLAS thread), an update takes:

| scheme | ms per update |
|---|---|
| tied/tied | 7.5 |
| tied/per-step | 7.4 |
| per-step/tied | 8.5 |
| per-step/per-step | 8.5 |

- Peak RSS is about 45 MiB.
- r1's host ran the same tied code at 3.9 ms per update. Hosts differ by about 2×.
- Each lane runs one short calibration on its own host first, and reports ms per update before planning.

**Budgets.** These are elapsed wall time with at least one job running, using `run_jobs.py --workers 2`. They are caps, not targets.

| lane | budget | what it is sized for |
|---|---|---|
| time | 40 min | 9 continuations × 40k updates; one follow-up |
| sharing | 20 min | 6 new 10k runs, optional c2-native replays of the 6 reused runs, one follow-up |
| generalization | 25 min | frozen evaluations, 16 replication runs at 10k, one follow-up |
| mechanisms | 15 min | analysis of existing checkpoints, one follow-up |
| reviewer | 25 min | |
| coordinator | 10 min | about 3.5 min used: tests, verification, calibration |

A lane that reaches 80% of its budget tells the coordinator before going further. Capped and failed runs stay in the registry.

## D7. Assignments, reuse and dependencies

**Time** (thread "Composing small programs", the r1 R1 worker).
- Continue all nine recovered runs, each from its `ckpt-10000` to 50,000 updates, with `c2train.py --resume … --diag --checkpoint-steps 20000 30000 40000 50000`. The nine are tied/inter, tied-transition/per-step-readout/inter, and tied/final, at seeds 51200–51202.
- Commit `ckpt-50000` of all nine as soon as they exist. They are downstream inputs; the intermediate checkpoints are optional.
- `c2-co-verify-*` already shows that resumed and uninterrupted runs are bitwise identical on this code path. The lane may cite it, or repeat it cheaply on its own host.
- Threshold: held-out final ≥ 0.95 for five consecutive logged evaluations, logged every 50 updates. A run that never meets it is censored at its last update.

**Sharing** (new thread).
- The 2×2 grid, all cells with intermediate supervision and step weights 1 1 1, at 10k and seeds 51200–51202.
- Two cells exist already, identical to `c2train.py`: tied/tied is `r1-comp-rnn-inter`, and tied/per-step is the reviewer's `rnn-untied`.
- New runs: per-step transition with a tied readout, and per-step transition with per-step readouts. That is six runs.
- Re-running the two reused cells with `c2train.py` to get c2-native logs is allowed. Label those runs replays.
- The time lane owns longer training.

**Generalization** (new thread).
- First, frozen evaluation of the recovered tied/inter and tied/final `ckpt-10000` at lengths 1–6, with no new training. Length 3 must reproduce r1's published held-out numbers. Add the time lane's 50k states once they are committed.
- Then the replication: 2 new draws × 2 splits × 2 init seeds × {tied/inter, tied/final} = 16 runs at 10k. Draws and acceptance criteria are logged before outcomes.

**Mechanisms** (new thread).
- Start from the recovered 10k checkpoints: tied/inter, tied/per-step/inter and tied/final, three seeds each. `ckpt-0` and `ckpt-1000` are also available.
- Use the time lane's 20k–50k states for any emergence trace. Do not retrain them.
- Per-step readouts in cross-position tests use the step map the lane declares. That is an explicit distribution shift.

**Reviewer** (thread "Regime expedition review").
- The pin (D3) first, then each lane's QUESTION.md and results.
- The reviewer wrote the r1 per-step-readout arm. Its runs are inputs to the time and sharing lanes, so the reviewer says so wherever it reviews claims that rest on them.

**Order of work.**
- Every lane commits QUESTION.md before taking measurements.
- Frozen analyses of recovered checkpoints may start once QUESTION.md is pushed.
- New training waits for the reviewer's PASS on the pin. The coordinator relays it.
- The single adaptive follow-up per lane is proposed in writing, as a "Follow-up" section appended to QUESTION.md and committed, and approved by the coordinator before launch.

## D8. Reporting

**Each lane delivers:**
- QUESTION.md;
- a README with exact commands, calibration, budget tally and run manifest, including failed and capped runs;
- one primary figure as PNG + SVG + CSV, with its regeneration command, visually inspected;
- NOTE.md, which separates observations from interpretation and bounds each claim by task, recipe and budget;
- a reviewer handoff list.

**Milestones** go to the coordinator thread in one line each: M0 QUESTION.md, M1 calibration and oracle checks, M2 first figure and PR, M3 follow-up, M4 NOTE.md. The coordinator logs them in [MILESTONES.md](MILESTONES.md), which is the reviewer's feed.
