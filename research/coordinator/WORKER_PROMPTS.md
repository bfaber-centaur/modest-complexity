# Worker prompts for expedition e1

Each worker is a separate Claude session. Give it the common block plus its role block. The role text from `research/LAUNCH_PROMPTS.md` is repeated verbatim; the assignment lines below it are the coordinator's specifics.

## Common block (all three workers)

Repository `bfaber-centaur/modest-complexity`. Fetch `swarm/exploration` and create your owned branch from the base commit named in your launch message (it contains this file; its code is the Stage 0 shared code `ebd874e17028f2c2f04de5ecea9fe4b51b77e425`). Record the commit you started from. Do not push to `main` or to `swarm/exploration`, and do not change shared code (`experiment.py`, `launch_runs.py`, `report_map.py`, `publish_run.py`, `instrumentation.py`, the plan or the lock). If shared code is wrong, report it with a minimal proposed patch on your branch; the coordinator integrates.

Read AGENTS.md, README.md (section "Exploratory map"), research/SWARM.md, research/RED_TEAM.md and research/coordinator/README.md.

Environment: `uv sync --locked --group analysis`, then `uv run --locked ...`. Set `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1` for any direct numerical run (`launch_runs.py` does this for its children). If uv says no interpreter for CPython 3.12.14, the installed uv is too old: `python3 -m pip download --no-deps uv==0.12.23 -d DIR`, extract the wheel with `python3 -m zipfile -e`, and put `uv-0.12.23.data/scripts/uv` first on PATH. Run `uv run --locked python -m unittest test_exploration test_instrumentation` before relying on the code and record the result.

Fixed data for the first map: data seed 31000, control seed 32000, `split_sha256` `aa8f7504bf7d925eb90c69183f10daf5778cce9aad8e460b2c907811526d1f24` (frozen arrays in `research/data/e1-split.npz`), addition target `6f1576d4…a2046`, shuffled target `80a9a845…3d35f` (full hashes in `research/plan.json`).

Cross-session artifacts: a path in another session is not available to you. Published runs live in Git under `research/<lane>/runs/<run_id>/` with `manifest.json` (SHA-256 for every local file, published or not). The coordinator publishes the first preview runs (`e1-co-add-w{32,128,512}-i30000-a1`) under `research/coordinator/runs/` on `swarm/exploration`; fetch the branch tip for them, verify hashes, and record which commit you read them from.

Handoff: push your owned branch and open a PR targeting `swarm/exploration` (if PR creation is unavailable, the pushed branch and commit SHA are enough). Then tell the coordinator thread ("Exploratory swarm launch") the branch, commit, what changed, checks run, and anything blocking. Keep prose short; figures first, interpretations in a separate file. Budget: at most 30 minutes of elapsed numerical execution in your session for the first map, then report completed and pending work. No paid compute.

## Size/time explorer

Read AGENTS.md and research/SWARM.md. You own research/size-time/ on a dedicated branch. Use the coordinator's exact shared-code commit and dataset hash; claim assigned run IDs before execution. Run the width/time map and assigned controls with separate initialization seeds. Do not modify shared training code independently.

Explore fit, generalization, margins, plateaus and reversals across individual trajectories. Record all attempted configurations and adaptive decisions. Preserve non-fitting/non-generalizing/capped runs. Checkpoints are repeated measurements, not independent seeds. Compare matched steps and, separately, comparable training loss. Suggest up to two bounded follow-ups addressing specific ambiguities; coordinate ownership before running them.

Publish checkpoints as they become available for the mechanism lane. Return up to three compact evidence packets, each with a neutral figure/prompt before interpretation, a counterexample or missing check, and exact reproduction instructions. Do not optimize for a positive grokking story.

Assignment:
- Owned branch `swarm/size-time` (or your session's designated branch; say which). Session label `st-cloud-1`. Registry `research/size-time/registry.jsonl` (commit it).
- Your runs are the 21 plan entries with `owner: size-time`, queue positions 4-24. Claim them by committing the registry's first events, then run them in queue order with two workers: `uv run --locked python launch_runs.py --root runs/e1 --registry research/size-time/registry.jsonl --session-label st-cloud-1 --timeout 1500 --workers 2 --ids <IDs in queue order>`. Run the launcher in chunks that fit your tool's time limits (for example by run ID batches), never as shortened runs: each run is a complete 50,000-update job. Expected cost, measured in the coordinator session: about 0.5 s (width 32), 1.3 s (128) and 4.8 s (512) per 1,000 updates, so roughly 13-16 minutes for all 21 with two workers.
- Commit `report_map.py` outputs for each condition (`--condition add`, `add0`, `shuf`) under `research/size-time/figures/`, and publish checkpoints with `publish_run.py ... research/size-time/runs` for at least one run per width plus the controls, as they finish.
- The coordinator's three preview runs complete the add/decay map at seed 30000; include them in your figures by reading `research/coordinator/runs/` and `research/coordinator/registry.jsonl`.

## Mechanism/measurement explorer

Read AGENTS.md, research/SWARM.md and PRELAUNCH_REVIEW.txt. Own research/mechanisms/ on a dedicated branch. Reuse the size/time lane's registered checkpoints and exact data. While waiting, develop independently checkable analyses on synthetic examples and calibration checkpoints; label those outputs correctly.

Study frequency retention/removal curves, checkpoint-local versus retrospective component selection, rescaling sensitivity of compression/rankings, and targeted interventions with controls. Verify function-preserving rescalings before interpreting metric changes. Work on analysis copies; do not change training dynamics. Optional operand-subspace patching needs whole-operand and no-op sanity checks.

Keep parameter count, parameter norm, codec size, frequency concentration, prediction accuracy and margins distinct. A failed proxy is a useful result. Save counterexamples and supply up to three figures-first evidence packets with alternative explanations and reproduction commands. Do not claim compositional reuse from the relabeled-output readout probe.

Assignment:
- Owned branch `swarm/mechanisms` (or your session's designated branch; say which). You run no training jobs from the plan.
- First deliverable (research/RED_TEAM.md section 4): one frequency-retention/removal curve over K=1..15 on one late checkpoint, using checkpoint-local selection, with accuracy, absolute loss and margins; plus one example testing whether a function-preserving positive hidden-unit rescaling (logits verified unchanged) changes the historical frequency ranking and the pruning/quantization codec. Start with `e1-co-add-w128-i30000-a1` (final checkpoint) once it appears under `research/coordinator/runs/`; until then use a synthetic or calibration checkpoint and label it as such.
- Normalization conventions, retrospective tracing and patching come after the first deliverable.

## Independent auditor

Read AGENTS.md and research/SWARM.md. Own research/audit/ on a dedicated branch. Review the coordinator's common foundation before substantive training, then audit selected outputs. Verify separate data/init seeds, fixed dataset hashes, disjoint splits, gradients/AdamW, loss/margins, checkpoint round-trips, and logging semantics. Independently recompute selected results and short traces; identical imports or duplicated formulas are not independent validation.

Audit at least one ordinary and one surprising result. Check run accounting and figure provenance. Separate software PASS/FAIL/NOT CHECKED from whether an interpretation is locally supported, ambiguous or contradicted. Report actual checked scope; do not issue a blanket 'validated' badge. Flag concrete defects immediately and link affected run IDs. Preserve original failures. Return a compact audit suitable for Bobby's review, with reproduction commands and unresolved concerns.

Assignment:
- Owned branch `swarm/audit` (or your session's designated branch; say which). Session label `au-cloud-1` for anything you run.
- Priority 1, the first milestone (do this first and report it as soon as it exists): review the Stage 0 diff `e0a6e0f..ebd874e` and the first preview runs `e1-co-add-w{32,128,512}-i30000-a1`. Complete the cross-session handoff check (fetch the published files, verify manifest hashes, load them). Using your own code (not imports of `experiment.py` formulas): confirm split disjointness, transpose grouping and the split hash; recompute logits, accuracy, cross-entropy and margins from published checkpoints and compare with `metrics.jsonl`; replay the first 1,000 updates (or more) of at least one preview run with an independently written forward/backward/AdamW and compare with the step-1000 checkpoint (state the tolerance; bitwise equality across hosts is not promised); check that the registry, `runtime.json` and figures account for every attempted run; check the preview figures against `curves.csv`.
- Then audit selected size/time and mechanism outputs, including one ordinary and one surprising result.
- Write `research/audit/AUDIT.md` with the PASS / FAIL / NOT CHECKED table and, separately, SUPPORTED LOCALLY / AMBIGUOUS / CONTRADICTED for any interpretation; include reproduction commands.
