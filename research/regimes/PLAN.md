# Regime expedition r1: shared execution plan

Opened 2026-10-22 by Bobby (project thread "Four-regime expedition"). Four small CPU investigations into learning, composition, reuse and training history, plus one independent reviewer. The aim is informative observations and trustworthy figures, negative results included. It is not to show a preference for "simplicity", to establish novelty, or to make confirmatory statistical claims.

This plan is the coordinator's. Each lane's own design lives in its QUESTION.md. Role prompts are in [coordinator/prompts/](coordinator/prompts/), and launch texts and thread/branch assignments are in [coordinator/DISPATCH.md](coordinator/DISPATCH.md). Decisions made after launch are appended to [coordinator/DECISIONS.md](coordinator/DECISIONS.md).

## 1. Branches and isolation

- **Integration branch:** `swarm/regimes` in `bfaber-centaur/modest-complexity`, created from `swarm/exploration` at `7c4389c` (e4 open). Only the coordinator integrates into it.
- **e4 is separate.** It runs on `swarm/exploration` with its own three threads. Nothing in this expedition pushes to `swarm/exploration`, either `main`, or upstream, and nothing touches e4's directories (`research/e4/`) or the shared top-level files.
- **Workers** develop on the branch their session designates (`claude/project-thread-…`), branched from `swarm/regimes`. Each owns one directory and writes nowhere else. PRs target `swarm/regimes`. If PR creation fails (the GitHub connector has been unreliable), push the branch and tell the coordinator its name and head commit.

| role | directory | run-ID prefix | seed range | session label | budget |
|---|---|---|---|---|---|
| R1 composition investigator | `research/regimes/composition/` | `r1-comp-` | 51000–51999 | `r1-comp-cloud-1` | 25 min |
| R2 shared-subcomputation investigator | `research/regimes/shared/` | `r1-share-` | 52000–52999 | `r1-share-cloud-1` | 25 min |
| R3 training-history investigator | `research/regimes/history/` | `r1-hist-` | 53000–53999 | `r1-hist-cloud-1` | 25 min |
| R4 competing-routes investigator | `research/regimes/routes/` | `r1-route-` | 54000–54999 | `r1-route-cloud-1` | 25 min |
| independent reviewer | `research/regimes/review/` | `r1-rev-` | 55000–55999 | `r1-rev-cloud-1` | 20 min |
| coordinator | `research/regimes/coordinator/`, `common/`, this file | `r1-co-` | 900000+ (calibration) | `r1-co-cloud-1` | 12 min |

**Seeds.** Within a lane's range, use 5x000–5x099 for task construction (for example the permutations or a teacher network), 5x100–5x199 for data and split sampling, 5x200–5x299 for initialization, and 5x300 and above for anything else (probes, interventions, minibatch order). The principal comparisons use three **paired** initialization seeds, 5x200, 5x201 and 5x202, shared across conditions. Every run record lists each seed by role, so initialization and dataset randomness stay separable. Seeds 900000 and above are calibration only and never pooled with research runs.

## 2. Shared code (small on purpose)

| file | what it is |
|---|---|
| `research/regimes/common/nets.py` | NumPy ReLU network with named softmax heads on the last hidden layer; unlabeled rows (`-1`); `forward_from(p, layer, H)` for patched or ablated activations; Adam with decoupled decay, explicit state (`adam_state` = a reset), and a `trainable` list (frozen names get no update and no decay); `save`/`load` of weights + optimizer state + metadata; `provenance()` = `instrumentation.inventory` plus hashes of the lane's own files |
| `research/regimes/common/test_nets.py` | finite-difference gradients, unlabeled rows, head-init independence, Adam against an independent calculation, frozen parameters, bitwise continuation after a checkpoint round-trip |
| `research/regimes/common/run_jobs.py` | `launch_runs.py`'s registry rules for arbitrary lane scripts: append-only `running` / `complete` / `failed` / `capped` events, no reused run IDs, a hard per-run timeout, one BLAS thread, ≤ 2 concurrent jobs |
| top-level `instrumentation.py`, `experiment.array_sha256` | runtime inventory and dataset hashing, reused as they are |

Lanes may use `nets.py` or write their own model where the regime needs it (for example a recurrent network). Either way the model gets its own finite-difference check. **Do not edit `common/` or any top-level file.** Propose a change to the coordinator as a patch with a test; the coordinator pins it and records it in DECISIONS.md. Keep each lane's code to a task/oracle module, a training script and an analysis/figure script, or fewer. No general experiment framework, no new dependencies.

## 3. Environment and execution

```bash
# Cloud containers ship uv 0.11.32, which cannot fetch CPython 3.12.14. Install 0.12.23:
pip download --no-deps uv==0.12.23 -d /tmp/uvwheel && python3 -m zipfile -e /tmp/uvwheel/uv-0.12.23-*.whl /tmp/uvwheel/x
mkdir -p ~/.local/bin && cp /tmp/uvwheel/x/uv-0.12.23.data/scripts/uv ~/.local/bin/ && chmod +x ~/.local/bin/uv && export PATH=~/.local/bin:$PATH
uv sync --locked --group analysis
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
uv run --locked python -m unittest -v research/regimes/common/test_nets.py
# Every numerical run goes through the runner (local output root outside git):
uv run --locked python research/regimes/common/run_jobs.py --plan research/regimes/LANE/plan.json \
    --root runs/regimes --registry research/regimes/LANE/registry.jsonl --session-label LABEL --workers 2 --ids ID ...
```

- **Plans.** Each lane owns `research/regimes/<lane>/plan.json`: `{"runs": [{"run_id", "argv", "timeout"}]}`, where `"{out}"` in argv becomes the run directory. Commit and push a plan entry before launching it, so the record shows the plan came first. There is no central claim step. IDs are globally unique by prefix, and use the form `r1-<lane>-<condition>-s<init seed>-a<attempt>`.
- **Timeouts.** Each plan entry sets a timeout of about three times the calibrated expectation, at most 900 s. A capped run is kept and reported. Rerunning it is a new attempt ID, never a silent overwrite.
- **Resume.** The runs here last seconds to a few minutes, so a run is one complete job. There is no automatic resume. Branching from a saved checkpoint (R3's phases) is explicit: it loads `nets.save` state, records the source file's SHA-256, and states whether the optimizer state was carried over or reset. A lane that adds a mid-run resume first checks it bitwise against an uninterrupted short trajectory.
- **Malloc.** `run_jobs.py` sets `MALLOC_MMAP_THRESHOLD_` and `MALLOC_TRIM_THRESHOLD_`. On the calibration host this sped up small full-batch updates about 1.6×; it changes allocation only, not results. Set them by hand for direct timing runs.
- **Run records.** Each run directory has `config.json` (every setting, each seed by role, data/split hashes, `nets.provenance`), `metrics.jsonl`, `runtime.json` (status, wall/CPU time, peak RSS) and the checkpoints the analysis needs. Publish small files into `research/regimes/<lane>/runs/<ID>/` with a SHA-256 manifest. Do not commit more than about 2 MB of checkpoints per lane; list the hashes of the rest.

## 4. Calibration and budgets

Coordinator calibration (`coordinator/calibrate.py`, random labels, full-batch Adam, 1 BLAS thread, 4 visible CPUs, Xeon 2.8 GHz, CPython 3.12.14, NumPy 2.3.5, OpenBLAS 0.3.30; records in [coordinator/calibration/](coordinator/calibration/)). Attempt a2, after the backward-pass copy fix and the malloc settings:

| shape (rows × inputs → hidden → heads) | updates/s, serial | updates/s, 2 concurrent |
|---|---|---|
| 384 × 20 → 128,128 → 3 heads of 8 | 288 | 297 / 298 |
| 1024 × 10 → 64 → 2 heads of 2 | 667 | 672 / 624 |
| 1024 × 11 → 128 → 2 heads of 2 | 389 | 376 / 382 |
| 512 × 10 → 128 → 1 head of 2 | 1069 | 941 / 877 |
| 1024 × 20 → 512,512 → 1 head of 8 | 18 | 18 / 18 |

Peak RSS was 69 MiB per process. Two concurrent jobs did not slow each other on this host. That is a 40 s calibration, not a forecast for long runs; each lane recalibrates on its own host with its actual script before setting timeouts.

**Budgets** are elapsed wall time with at least one numerical job running (two concurrent jobs count once). Analysis counts only when a single script takes over a minute. Each lane keeps a running tally in its README. The investigator budget of 25 min covers calibration, positive controls, the initial comparison, one follow-up and analysis. At these sizes 10,000 updates take about 10–35 s, so the budget allows on the order of 100 such runs at two at a time, far more than the initial comparisons need. Wide networks are about 20× slower per update. A lane that will exceed its budget stops and asks the coordinator. No paid compute.

## 5. The execution contract (Bobby's, with logistics)

**M0 — QUESTION.md, before any main result.** It covers the exact question; task, architecture, objective and split; two or more plausible outcomes and the observation that distinguishes them; controls and known confounders; the proposed first figure; the initial run budget and stopping rules. Push it. Then send the coordinator one message: branch, commit, a one-line design summary.

**M1 — Oracle, splits, positive control, calibration.**
- Validate the oracle and splits with an independent check: a second, differently written computation or exhaustive enumeration, not the same function called twice.
- Run a positive control showing the task and evaluation pipeline can succeed. A hard-coded oracle validates the task, not trainability. A trainable positive control is, for example, a model trained on the full domain, or an easier split.
- Calibrate on your host and record versions, hardware, thread settings, peak memory and elapsed time.

**M2 — The first useful figure, before expanding.**
- Three paired seeds per principal comparison. Checkpoints, examples and branches from one run are not independent replications.
- Show every seed, failures, capped runs and missing results. Prefer trajectories and effect sizes to significance tests.
- Include training performance next to held-out performance, so failed optimization is distinguishable from failed generalization.
- Push it with its CSV and command, open the PR into `swarm/regimes`, and send the coordinator one message.

**M3 — One adaptive follow-up.**
- Write its motivation and prediction into QUESTION.md under "Follow-up" and push that before running it.
- Keep its results visibly separate from the initial comparison.
- No broad architecture × width × optimizer × duration grids.
- You need not wait for the coordinator, but you must have pushed M2 first.

**M4 — Handoff.**
- `NOTE.md`: question, design, observations, limitations, next question.
- One primary figure with machine-readable data.
- Exact commands, plan, registry and run manifest including failed and capped runs.
- Update the PR and send the coordinator one message.

**Figures** follow the minimum standard in [../RED_TEAM.md](../RED_TEAM.md):
- static Matplotlib;
- individual seed traces, with no confidence ribbons over three seeds;
- missing and capped portions as gaps;
- comparable axes, with chance and baseline references drawn;
- a CSV and the regeneration command beside each figure;
- PNG plus SVG;
- open the exported PNG and inspect it before handoff.

## 6. Independent review

The reviewer checks, per regime:
- oracle and split correctness;
- loss and metric calculations;
- data and compute exposure across controls;
- intervention semantics;
- checkpoint and optimizer-state handling;
- whether figures support the written claims;
- at least one attempted counterexample to the leading interpretation.

It also checks `common/`.
- **Blocking.** A finding blocks only the affected runs, metrics or claims. Review status per regime is recorded in `research/regimes/review/REVIEW.md` and mirrored in the coordinator's INDEX.
- **Fixes.** The reviewer never edits a lane's files. It reports and may propose a patch.
- **Merging.** The coordinator merges a lane packet after review, or with "review pending" stated in INDEX.

## 7. Communication

Cross-thread messages are capped at about 20 an hour per thread, so they are for milestones only, M0, M2 and M4, plus real blockers. The branch is the record: anything long goes in a file, and the message points to it. If a send is refused for the rate limit, commit the message to your directory as `OUTBOX.md` and retry later. Workers do not message each other, except the reviewer replying to a lane about a finding. The coordinator forwards lane milestones to the reviewer.
