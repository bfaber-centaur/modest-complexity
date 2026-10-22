# e4 decision log

Charter: [charter/](charter/README.md). Literature: [LITERATURE.md](LITERATURE.md). Launch messages: [DISPATCH.md](DISPATCH.md). Run claims: [../plan.json](../plan.json). Newest entries come last. Earlier entries are not edited, only superseded by later ones.

## D1 (2026-10-22) Open e4, keep e3 as it is

e3 closed at `swarm/exploration` `6b8d6c1` (e3 D17). That was the branch head when e4 setup began. e1–e3 artifacts and run directories stay where they are and stay reusable. They get dated pointers, not rewrites.

e4 asks one question: **between the post-filter accuracy peak and the collapse, what changes in the computation, and which of those changes cause the loss of correct answers?**
- The **dynamics** lane asks how decay strength changes when and whether the gain is lost.
- The **mechanisms** lane asks what changes in the network.
- The **auditor** checks both and builds the strongest cheap control.

The `walkthrough/rescaling` branch (`b22f7ce`, unmerged) stays separate.

## D2 (2026-10-22) Sessions and ownership

The existing threads keep their e3 roles. There are no new threads.

| e4 role | session (thread) | owned directory | branch | registry / session label |
|---|---|---|---|---|
| coordinator | "Exploratory swarm launch" | `research/e4/coordinator/`, `research/e4/plan.json`, shared code | `swarm/exploration` | `research/e4/coordinator/registry.jsonl` / `e4-co-cloud-1` |
| dynamics explorer | "Width and training-time map" | `research/e4/dynamics/` | `claude/project-thread-j8mgkm` | `research/e4/dynamics/registry.jsonl` / `e4-dy-cloud-1` |
| mechanisms explorer | "Frequency mechanisms lane" | `research/e4/mechanisms/` | `claude/project-thread-psralm` | `research/e4/mechanisms/registry.jsonl` / `e4-me-cloud-1` |
| independent auditor | "Correctness audit" | `research/e4/audit/` | `claude/project-thread-id1tmm` | `research/e4/audit/registry.jsonl` / `e4-au-cloud-1` |

Two lanes had pushed before this setup existed. Both pushes are on `6b8d6c1`, and neither made an e4 run.
- **Dynamics** pushed `e5fd1c2`: a QUESTION.md (`e4e5688`) and `decay.py` (`9257c00`). It measured the existing wd 0 and wd 1 runs and then fixed per-run predictions and a decision rule for wd 0.1 and 0.3, before any new run.
- **Mechanisms** pushed `d584af7`, a QUESTION.md written before any measurement.

## D3 (2026-10-22) Pinned base and shared code

The e4 base for all workers is the `swarm/exploration` commit that adds this file. Its hash is in [INDEX.md](INDEX.md) and in the launch messages. A later `swarm/exploration` commit is equally valid as a base, as long as it changes no shared code.

Shared code is `experiment.py`, `launch_runs.py`, `report_map.py`, `publish_run.py`, `instrumentation.py`, `pyproject.toml`, `uv.lock` and the shared tests. It is **unchanged** since the e3 re-pin `72cfb5d` (e3 D15). The e3 mechanisms runs were made at `9041a48`, and `72cfb5d` only added `--resume`.

Comparability was checked by the coordinator at `6b8d6c1`. Replaying `e3-me-filt-w128-i30000-a1` from its start file reproduces every published checkpoint array **bitwise** at updates 0, 10, 30, 60, 100, 300, 1,000, 2,000 and 5,000. All metric rows to 5,000 are equal except the two wall-clock fields. The existing wd 0 and wd 1 runs are therefore valid endpoints for the e4 decay-strength runs.

The replay was a scratch check with no run ID and is not published. It used the D7 dense checkpoint list. A lane may repeat such a check, unclaimed and labelled diagnostic.

**Continuation semantics** are e3's, restated here:
- `--init-weights FILE --expect-init-sha256 H --init-age N` sets the Adam moments to zero, restarts the update counter at 1, and records the age only. The step-0 row is the post-edit evaluation.
- `--resume CKPT` carries over weights, moments and update count exactly.
- `--checkpoint-steps` adds checkpoints at listed multiples of `--log-every`.

Any further shared-code change goes like this:
1. It is proposed as a patch on a lane branch, with a test.
2. The coordinator pins a new hash here.
3. Only then are the runs that depend on it claimed.

## D4 (2026-10-22) What exists: checkpoint and optimizer-state inventory

[inventory.py](inventory.py) produced [inventory.json](inventory.json). It covers the 15 e3 mechanisms runs (keep, filt and sham at wd 1; filt and sham at wd 0).

**What is stored**
- Every checkpoint holds the full optimizer state: `U, V, W, mU, mV, mW, vU, vV, vW, step`.
- Metrics are logged every 10 updates.
- The filt runs have checkpoints at 0, 10, 30, 100, 300, 1,000, 2,000, 5,000, 10,000, 15,000 and 20,000, plus one extra checkpoint at 60 (i30000) or 50 (i30001); i30002 has none.

**Fit times.** The first logged update with training accuracy exactly 1 is:

| arm | i30000 | i30001 | i30002 |
|---|---|---|---|
| filt | 70 | 50 | 40 |
| keep | 10 | 10 | 10 |
| sham | 110 | 100 | 100 |

The filt fit times are the same at wd 0 and wd 1.

**The wd 0 and wd 1 runs from the same start file** share the start-file hash. Their configs differ only in `wd` and bookkeeping.

**The peak is not checkpointed.** The logged held-out peak (dynamics, `e5fd1c2`: 0.75–0.88 at 70–80 updates at wd 1) falls between published checkpoints. A checkpoint near the peak is not the peak, so D7 adds exact states.

**Optimizer state at the start.** Every e4 start is a fresh-Adam continuation. The start files carry no moments, because the filter edit makes the source run's moments meaningless. Fresh moments at the edited point are part of the e3 recipe, and e4 keeps them.

## D5 (2026-10-22) Time references, windows, evaluation sets and measures

**Clock.** Time is updates since the edit. Update 0 is the post-edit state.

**Label-free reference.** t_fit is the first logged update with training accuracy 1: 70, 50 and 40 for i30000, i30001 and i30002 (D4).

**Windows on the update axis.** These are defined for wd 1:

| window | updates |
|---|---|
| rise | [0, t_fit] |
| peak | [t_fit, 300] |
| decline | [300, 2,000] |
| late | [2,000, 20,000] |

The dynamics lane compares other decay strengths on the decay-exposure axis E(t) = −Σ log(1 − lr·wd_t), which is −t·log(1 − lr·wd) at constant wd, and also on the update axis.

**Fixed states.** States are compared at 0, t_fit, 100, 300, 1,000, 2,000, 5,000 and 20,000. All of these exist in the e3 runs and the D7 replays.

**Peak time.** The held-out peak time t_peak may be reported per run as an observation. It uses held-out labels, so it is not used to choose which states an analysis compares unless that choice was fixed in advance.

**Evaluation sets**
- The primary held-out set is the 483 common never-trained pairs, `research/e2/data/common-never-trained-30-50.npz`.
- Each run's own cal and audit pairs are secondary.
- The 299 training pairs are reported separately.

**Measures**
- Accuracy and cross-entropy.
- The e3 pairwise measures from `research/e3/audit/pairwise.py`: P(S over N), P(a+b over N) and P(a+b over S).
- The σ-margin, (logit of a+b − max over N) / σ of the 30 wrong logits.

A new measure is defined in the lane's QUESTION.md before it is computed on a new run.

**Decline definition.** The dynamics lane declared its own definition in `e4e5688`/`e5fd1c2`, before any new run. It has two thresholds: held-out accuracy below its update-0 value, and below 0.10. A run that does not cross a threshold is censored at its last update and is not given a crossing time.

## D6 (2026-10-22) Dynamics initial experiment: decay strength, six runs

**Runs.** Six claimed runs, `e4-dy-filt-wd0p1-w128-i{30000,30001,30002}-a1` and `e4-dy-filt-wd0p3-w128-i{…}-a1`:
- each starts from a gated e3 filtered start file `research/e3/mechanisms/starts/e3-me-filt-w128-i<seed>-a1.npz` (hashes in plan.json);
- fresh Adam, init_age 50,000;
- the original 299 pairs (data seed 31000, split `aa8f7504…`);
- lr 1e-3, 20,000 updates, metrics every 10;
- checkpoints every 1,000, plus 10, 30, 100, 300, 500, 1,500 and 2,500.

The extra checkpoints after 300 bracket the expected wd 0.3 decline. Everything else equals `e3-me-filt[-wd0]-w128-i<seed>-a1`, which are the wd 1 and wd 0 endpoints.

**Naming.** The `wd0p1`/`wd0p3` IDs follow the lane's suggestion.

**Queue.** The order is the seeded permutation in plan.json (`queue_position`, `default_rng(20261024)`), so wd and seed are not confounded with queue position. Launch is from a clean worktree at the pinned base. The expected cost is about 1 min per run and 3–4 min elapsed with 2 workers.

**What is labelled separately.** Endpoint interventions (changing wd only at a declared update) and staged decay schedules are a different kind of experiment from constant-wd runs and are labelled separately. They are follow-up candidates and need a shared-code change for a schedule (D8).

## D7 (2026-10-22) Dense diagnostic replays for the mechanisms lane

**Runs.** Six claimed replays, `e4-me-filt-wd{1,0}-replay-w128-i{30000,30001,30002}-a1`:
- purpose `diagnostic-replay`, with `replay_of` set to the e3 run;
- the same start file and settings as `e3-me-filt[-wd0]-w128-i<seed>-a1`;
- 5,000 updates;
- checkpoints every 10 to 300, every 50 from 350 to 3,000, and every 500 from 3,500 to 5,000.

These are replays, not new seeds. They are dependent observations of the published runs, which they must equal at every shared checkpoint: arrays bitwise, and metric rows except wall-clock fields. The coordinator's check in D3 used this exact list and passed for i30000 at wd 1.

**Why replays and not resumes.** The lane asked for `--resume` continuations from update 100 to 3,000 with checkpoints every 50. A replay from update 0 holds those states too, and it also holds the rise and the peak at 10-update resolution, at the same cost (about 10 s per run).

**Storage**
- About 89 checkpoints per run is about 25 MB per run, so **the checkpoint files are not committed**.
- The lane commits `config.json`, `manifest.json`, `metrics.jsonl`, `runtime.json` and a SHA-256 list of every checkpoint file. It may also commit the D5 fixed states.
- Anyone can regenerate the rest with the registry command at the pinned code.

**Analysis of published artifacts.** The mechanisms lane's first figure may use the published e3 checkpoints now, as its QUESTION.md (`d584af7`) proposes, and the replays add density afterwards.

## D8 (2026-10-22) Workflow, follow-ups and interventions

**Initial experiments**
- Each explorer has one initial experiment (D6, D7 plus the lane's analysis) and **one** follow-up.
- A follow-up is granted only after the coordinator has seen that lane's first figure.
- No large sweeps.

**The mechanisms intervention.** The mechanisms lane gets one intervention after its first figure. It proposes the intervention with IDs, settings, the measurement and a pre-stated reading.

**Shared-code changes.** An intervention or a staged schedule that needs one (for example a frozen block, a per-matrix wd, or a wd schedule) follows D3. A frozen-parameter experiment states whether the frozen block gets neither updates nor decay, or only no updates. A decomposition states its object (weights, activations or logits), its basis rule and its reconstruction error.

**The auditor**
1. Audits the decomposition and the readings.
2. Builds the strongest cheap control for the main explanation.
3. Runs dynamics and intervention checks.

The auditor's runs are claimed like anyone's.

**Claims and outputs**
- A lane launches only IDs listed in `research/e4/plan.json` at a pushed commit, using `launch_runs.py --plan research/e4/plan.json`, its own registry, and `--workers` ≤ 2. `launch_runs.py` sets one BLAS thread per process.
- Outputs go to `research/e4/<lane>/runs/<run ID>/`.
- Run IDs are never reused. Failed or capped runs are kept and reported.

**Integration.** PRs target `swarm/exploration`, and the coordinator integrates them. A scientific packet is merged after the auditor has reviewed it, or with the audit marked as pending in INDEX.

## D9 (2026-10-22) Calibration and budgets (set for e4, not inherited)

**Calibration on the coordinator's host**, with 2 concurrent processes and 1 BLAS thread each:
- 2,000 updates from a filt start at wd 0.3 took 4.0 s, so 20,000 updates take about 40 s;
- the dense 5,000-update replay took 10.4 s with 89 checkpoints.

The dynamics lane estimates 45–60 s per 20k run on its host.

**Disclosure.** That 2,000-update calibration was a scientific configuration of the D6 design: filt i30000 at wd 0.3. The coordinator saw its final audit-split accuracy. The value is not used for any decision. It is not stated here, so that it cannot inform the lanes, and it will be listed in the review packet. The dynamics lane's predictions were pushed (`e5fd1c2`) before this log.

**Budgets** are in elapsed numerical execution. Analysis counts only when a script takes over a minute.

| session | budget | basis |
|---|---|---|
| dynamics | 20 min | about 4 min for D6, an equal or smaller follow-up, analysis |
| mechanisms | 20 min | about 1 min for D7, analysis over about 540 states, one intervention |
| auditor | 25 min | replays, controls and checks of both lanes |
| coordinator | 10 min | about 0.5 min used for calibration and the D3 replay |

A lane that will exceed its budget stops and asks. No paid compute is used.

## D10 (2026-10-22) What changed from e3

- There is no gate. Every start file was audited in e3 (e3 D10).
- Budgets were recalibrated and are lower.
- Dense replay checkpoints are not committed (D7).
- A single question spans all three lanes, so each lane's packet names the windows of D5 it speaks to.
- The literature table ([LITERATURE.md](LITERATURE.md)) records retrieval gaps. Only Varma et al. (body) and Janati et al. (main text, appendices A–I) were read beyond the abstract.

## D11 (2026-10-22) Audit control: random-feature starts through the B protocol (reserved)

**Proposal.** The auditor proposed a control before training anything. It runs the e3 B continuation protocol from a **random-feature start**, built per init seed 30000–30002 in four steps:
1. U and V are the untrained values from that init.
2. W is a ridge readout fitted to the 299 training labels.
3. The e2 per-unit dominant-frequency filter is applied, as for B.
4. U and V are rescaled jointly, and W separately, to the matrix norms of the B start file. This is a scale-only change, so the update-0 argmax, ranks and σ-margins are unchanged.

**Runs.** The runs are at wd 1 and wd 0, everything else as in D6/D7.

**What it tests.** It asks whether the peak-then-collapse belongs to "filter plus training" in general, or needs what the failed runs contain. That question decides how to read any mechanism found between B's peak and its collapse.

**Reserved IDs.** Six IDs are reserved in plan.json under `reserved`: `e4-au-rf[-wd0]-w128-i{30000,30001,30002}-a1`, with init_age 0 and `init_weights_sha256` PENDING. The coordinator moves them to `runs` and fills the hashes after the auditor has pushed three things:
1. the start files under `research/e4/audit/starts/`;
2. the script that writes them;
3. QUESTION.md with the decision rule stated before any run.

**Decision rule.** The rule uses the dynamics lane's declared decline thresholds (D5) for "collapse". A rise between 0.1 and half of B's rise is reported as inconclusive. The cost is about 2 min of the auditor's 25.

## D12 (2026-10-22) Audit control activated

The auditor pushed three things on `claude/project-thread-id1tmm`:
- `QUESTION.md` (`33bf731`), with the decision rule stated before any run and the D11 edits included;
- `rf_starts.py` (`c7ef7d0`);
- the start files with their update-0 report (`aa09dd4`).

The coordinator checked the start files as fetched from that branch:
- **Hashes.** Each file matches its SHA-256 (i30000 `898a65f8…`, i30001 `3d52f618…`, i30002 `e15b1e44…`).
- **Contents.** Each holds only float64 U, V and W of the expected shapes.
- **Norms.** The joint (U, V) norm and the W norm equal those of the B start file for the same seed to the last digit or so.

The six D11 IDs are moved from `reserved` to `runs` with these hashes. The start files reach `swarm/exploration` with the auditor's PR. Until then the auditor launches from its own branch merged with this commit.
