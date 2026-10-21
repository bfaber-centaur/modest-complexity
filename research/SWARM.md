# Exploratory swarm: modest-complexity

Design for review, 2026-10-21. No workers or research runs launched by this document.

## Purpose and posture

Build intuition about how width and training duration change learned modular-addition behavior. Inspect how fit, generalization, margins, approximate computation, and parameter compressibility evolve together. Return reproducible observations, competing explanations, and promising questions for Bobby to analyze.

This is exploratory science. Adaptive choices, negative results, abandoned hypotheses, and unexpected examples are welcome. No preregistered success criterion, significance gate, or composite verdict is required. Record what was tried and why the next experiment was chosen. Later confirmatory work uses fresh seeds and a separately frozen protocol.

This charter supersedes the historical protocol's pilot/main gating and composite verdict **for this exploration only**. Preserve those files as history; do not execute `run_study.py main` or present its composite decision as a scientific conclusion. Correctness is required; validation of every possible complexity measure is not a prerequisite for training.

## Organization: four concurrent sessions at most

| Role | Owns | Primary deliverable |
|---|---|---|
| Coordinator / infrastructure | shared code, environment, run registry, integration and synthesis | one reproducible implementation and a navigable review packet |
| Size and training explorer | `research/size-time/`, assigned runs | width-by-time behavior maps, representative trajectories and exceptions |
| Mechanism and measurement explorer | `research/mechanisms/`, analyses of shared checkpoints | explanations tested by interventions; sensitivity of complexity proxies |
| Independent correctness auditor | `research/audit/` | checked implementation and result provenance, with explicit unresolved defects |

The coordinator works on centaur/main. Workers use separate branches/worktrees and their owned directories; only the coordinator integrates shared-code changes. The auditor may propose a patch but never silently replaces an explorer's output. Publish integrated work as a PR to bubskee/main; Bobby merges upstream. No fifth research lane until an existing lane has returned something reviewable.

Workers are separate Claude sessions where available. Subprocesses within a session share its resources. Do not equate more agent sessions with proven linear compute scaling.

## Stage 0: make the current prototype usable for exploration

The coordinator implements a small common foundation; the auditor checks it while explorers prepare analyses. Do not build a platform or a dashboard first.

1. Separate `data_seed`, `init_seed`, `control_seed`, and diagnostic randomness. Freeze and hash one generated dataset/split for the first map. The existing single `seed` currently changes both initialization and split, so it is not sufficient for a fixed-dataset width comparison.
2. Allow a manifest to identify each run: ID, purpose=exploratory, task, width, all seeds, optimizer settings, horizon, measurement cadence, code commit, environment hashes and host/session label. Start with exploratory initialization seeds 30000–30002, separate from original pilot/main and benchmark seeds. Different widths with the same seed are not identical initial models.
3. Keep the current training rule: p=31, one-hidden-layer ReLU, float64, full batch, AdamW lr=.001, beta=(.9,.999), epsilon=1e-8, decay=1. Inputs initialize with SD .2; output weights with SD 1/sqrt(width). Describe this as varying width under a fixed recipe, not equal optimization difficulty or equal regularization effect.
4. Preserve lightweight loss/accuracy/margin/update logging every 100 steps. Save weights and optimizer state at step 0 and every 1,000 steps plus the last step. Ensure the config and exact data stay with the checkpoints. Permit expensive analyses to run offline; identify which diagnostic schedule each run used.
5. Make a compact status registry: planned, running, complete, failed, or capped, with completed steps and reason. Never overwrite outputs, reuse IDs, or silently omit failures. Write status and checkpoints incrementally. Do not claim resume support unless implemented and tested against an uninterrupted short trajectory.
6. Disable broad composite conclusions in the exploratory reporting path; keep historical analysis explicitly labeled. Readout accessibility is the correct name for the permuted-output ridge probe. Weight norm, Fourier concentration and pruning/quantization code length remain distinct measurements.

Essential checks: disjoint grouped splits; unchanged dataset across widths/initializations; finite-difference gradients; AdamW against an independent calculation over multiple steps; checkpoint round-trip and offline-vs-inline measurement agreement. A representative short job must include checkpoint I/O. These are code checks, not tests of the simplicity hypothesis.

## Stage 1: first map, intentionally modest

Starting allocation, adaptable with a logged reason:

- Widths 16, 32, 64, 128, 256, 512.
- Three initialization seeds at each width, same dataset/split: 18 addition runs.
- Up to 50,000 updates per run, subject to the measured execution budget. Evaluate all saved times from the same trajectory instead of retraining for each duration.
- Six anchor controls: width 128, the same three initialization seeds, (a) addition without decay and (b) a fixed shuffled-label table with decay. The shuffled table preserves split-wise class counts but is noncommutative; it is a memorization diagnostic, not a perfectly matched complexity control.
- Proposed fixed data seed 31000 and shuffled-label seed 32000. Record resulting array hashes before training. A second data split is an optional follow-up, not mixed into the first map without a label.

The 24 runs are a starting map, not a statistical power calculation or minimum sample size. Preserve non-fitting and non-generalizing cases. Early successful generalization is interesting too; delayed grokking is not the admission criterion.

Mix widths in the execution order. If more than one host trains models, spread conditions across hosts rather than assigning all narrow models locally and wide models remotely. Record host as a potential nuisance factor. Do not pool a local/cloud rerun of the same seed as an independent replicate.

Training length is an axis within trajectories. Checkpoints are not independent replicates. Plot individual seed traces; any aggregate should keep their spread visible. The first map is conditional on its chosen dataset split.

## Lane questions

### Size and training explorer

Start with accuracy, cross-entropy, signed margins, and the time courses of the measured optimizer updates. Inspect raw and width-normalized parameter norms as descriptive quantities only. Ask:

- Which widths fit, generalize immediately, generalize late, or remain unsuccessful within the horizon?
- Does extra width buy earlier learning, lower eventual loss, larger margins, or a different trajectory?
- Are apparent plateaus stable across initializations? Are there reversals or late spikes?
- What changes at matched steps? Separately, what changes at comparable training loss? Report unmatched runs instead of discarding them.

Produce a width-by-time overview and individual trajectories, with first and sustained threshold crossings distinguished descriptively. Thresholds used for visualization must be stated and can be revised openly. Do not report only successful-run mean grokking times. Timing is censored at the run horizon; a capped job is not evidence of impossibility.

After the first map, choose up to two bounded follow-ups: extend a plateaued case and its neighboring width, add seeds near a qualitative boundary, or probe a smaller width. Explain what ambiguity each follow-up addresses. Optimizer retuning or architecture changes belong to a separately labeled comparison.

### Mechanism and measurement explorer

Reuse the first lane's checkpoints. Sample early fitting, pre-generalization, transition and late checkpoints, including a failed or contrary example. Keep their selection rule and whether it used future outcomes visible.

- Plot frequency retention/removal curves across all possible K=1..15 rather than only the old top-four threshold. Report accuracy, absolute loss and margins. Full retention is a reconstruction control, not evidence of a compact mechanism.
- Separate frequencies selected using the current checkpoint from final-checkpoint frequencies traced backward. Both views are useful and answer different questions.
- Apply positive hidden-unit rescalings to checkpoint copies, compensating outgoing weights. First verify logits are unchanged within numerical tolerance; then measure changes in the raw codec and frequency ranking. This is an invariance audit, not another training condition.
- Try a documented normalization convention on analysis copies and repeat the invariance check. It addresses that specific symmetry, not every equivalent representation or minimum program length. Never normalize the live training model without declaring a new experiment.
- Compare targeted perturbations with matched shams. Explore counterfactual operand-subspace patching if feasible; verify whole-operand replacement and no-op controls first. An intervention that changes output as predicted is stronger evidence than a pretty Fourier plot, but does not establish reuse across tasks.
- Distinguish a compact core from small collective contributions. Retaining more neurons may improve approximation or margin without introducing a different high-level algorithm.

Raw metrics may be reported before gauge issues are resolved, with sensitivity attached. Do not claim architecture-independent simplicity from them. Nulls that do not actually match intervention magnitude/rank must be identified rather than called matched.

### Independent correctness auditor

Check implementation, provenance and interpretation separately. Recompute a small selection from exact stored data/checkpoints using independently written formulas where practical. Prioritize:

- Seed separation and fixed split identity; no train/audit leakage.
- Optimizer state and actual displacement diagnostics; current-step versus interval-average semantics.
- Logit/margin/loss computation, checkpoint integrity, and counts of completed versus attempted runs.
- Rescaling invariance of predictions and known non-invariance of weight-based proxies.
- A short rerun of one ordinary and one surprising case. Numerical backend differences may change long trajectories; validate short computations without promising cross-platform bitwise equality.
- Whether each proposed explanation has a real competing explanation and whether any intervention distinguishes them.

Issue an audit table with PASS / FAIL / NOT CHECKED for software checks, and SUPPORTED LOCALLY / AMBIGUOUS / CONTRADICTED for interpretations. A software pass is not scientific confirmation. Attach defects to the affected run IDs; preserve originals and mark replacements explicitly.

## Execution envelope

Use `uv sync --locked` / `uv run --locked`. The benchmark launcher sets BLAS threads to one; direct training launches must do so too. Start with at most two numerical processes per session. Use more only after measuring throughput with the actual diagnostic/I/O workload.

Observed context: ALM's two Claude sessions exposed 4 vCPUs, 13.36 GiB and ~30 GB; two-process FFT scaling was good, cross-session speed differed up to about 1.5x. This is not a guarantee for new sessions or matrix multiplication. Bobby's M4 short calibration took 2.02 s serial / .73 s with two workers for two 1,000-update jobs; the >2x ratio is too short/noisy to treat as sustained scaling.

Time a representative width-128 and width-512 job before committing the entire batch. Use bounded chunks, normal tool timeouts, and checkpointed progress; no keep-alive workaround. Initially budget at most 30 minutes of elapsed numerical execution per session for the first map, then report completed and pending work. This is a resource bound, not a scientific stopping rule. Adjust the allocation from observed rates; do not interpret resource-capped runs as failed hypotheses. No paid compute is needed or authorized by this packet.

Analysis can proceed as checkpoints arrive. Save compact metrics/configs/figures and selected checkpoints with hashes in lane directories; avoid committing every transient array. Before cleaning local state, ensure retained checkpoints needed for review are durably available and provide paths/hashes. Do not silently discard failed runs to save space.

## Review packet for Bobby

Each lane returns at most three observation packets for the first review. The full run registry stays available; the limit is on prose, not on preserving evidence.

For each packet:

1. A neutral question and a main figure, with seed-level traces and run IDs.
2. A short 'look before reading' prompt so Bobby can interpret the figure independently.
3. Observation: what the data directly show, with thresholds/selection described.
4. Interpretation: one candidate explanation and its strongest alternative.
5. Counterexample, failed control, or the most important missing check.
6. Next distinguishing experiment, without pretending it was predicted in advance.
7. Exact code/environment/data/checkpoint provenance and reproduction command.

Keep figures and prompts in a figures-first index; put interpretations in a separate linked file. The coordinator adds one short synthesis: repeated patterns, disagreements, defects, and at most three worthwhile next questions. No wall of agent transcripts, no ranking by dramatic effect size, no automatic claim promotion.

Useful references already reviewed:
- Burny's replication and failed key-frequency criterion: https://github.com/BurnyCoder/visualizing-grokking-modular-addition/blob/main/reports/exp-03-hypotheses.md
- Retrospective frequency definitions: https://github.com/BurnyCoder/visualizing-grokking-modular-addition/blob/main/docs/operationalizations.md
- Approximate CRT: https://arxiv.org/abs/2505.18266
- Quadrature interpretation: https://arxiv.org/abs/2412.03773
- ALM resource report: https://github.com/bfaber-centaur/artificial-life-museum/blob/ca260c1944bd531b0f882a517cad284c9547c3b1/research/reports/reef-environment-measurements-2026-10-20.md
