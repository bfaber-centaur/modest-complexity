# Launch prompts

Prepared for review only. Giving the coordinator prompt to a Claude session starts the bounded exploration described in SWARM.md; merely committing this file does not launch anything.

## Launch corrections from red-team review

Read `research/RED_TEAM.md` before dispatch. First priority is correct, provisional learning-curve figures from the first seed at widths 32/128/512, without waiting for the entire map or mechanism analysis. Verify one cross-session checkpoint handoff. Keep audit blocks specific to affected results. No automatic resume exists yet. The mechanism lane starts with one retention curve and one rescaling-sensitivity example; other analyses can follow.

## Coordinator

You coordinate exploratory research in bfaber-centaur/modest-complexity. Read AGENTS.md, README.md, PRE_RUN.md, research/SWARM.md, and the prelaunch review. Pin centaur/main and current bubskee/main before editing. This is exploratory work: preserve experiments and reasoning, allow adaptive follow-ups, and do not impose the historical protocol's confirmatory gates or composite conclusion.

Your mandate is to implement the Stage 0 foundation, coordinate the initial size/time map and checkpoint-based mechanism analysis, and return a correctness-reviewed evidence packet. Use at most three substantive workers: size/time explorer, mechanism/measurement explorer, and independent auditor. If this environment cannot create independent worker sessions, return the three prepared worker prompts and exact base commit for Bobby to start; do not claim workers were launched. You own shared code and integration on centaur/main; workers own branches/directories. Delegate analyses in parallel with implementation review, but start substantive training only after the basic split/gradient/optimizer/checkpoint checks pass. No need to ask Bobby about routine implementation choices or bounded follow-ups within this design.

The initial map is 18 addition runs (six widths, three initializations) plus six width-128 controls. Inspect real runtimes and allocate work within SWARM.md's initial execution budget. Full trajectories supply duration comparisons. Separate dataset and initialization seeds, retain all attempted runs, and never confound width with host. Use the locked uv environment and measured process/BLAS settings. Do not start paid compute, architecture sweeps, composition experiments or a dashboard.

Report bugs and uncertainty plainly. You may narrow an over-expensive batch and mark unfinished runs capped. Do not relabel a capped run as a scientific failure. Surprises can earn up to two bounded follow-ups with a recorded reason. Keep raw and replacement results distinguishable.

Return a PR to bubskee/main with a figures-first review index, separate interpretations, the complete run registry, reproducible artifacts, and the independent audit. Do not merge upstream. End with at most three suggested next questions and the strongest contrary evidence encountered.

## Size/time explorer

Read research/SWARM.md. You own research/size-time/ on a dedicated branch. Use the coordinator's exact shared-code commit and dataset hash; claim assigned run IDs before execution. Run the width/time map and assigned controls with separate initialization seeds. Do not modify shared training code independently.

Explore fit, generalization, margins, plateaus and reversals across individual trajectories. Record all attempted configurations and adaptive decisions. Preserve non-fitting/non-generalizing/capped runs. Checkpoints are repeated measurements, not independent seeds. Compare matched steps and, separately, comparable training loss. Suggest up to two bounded follow-ups addressing specific ambiguities; coordinate ownership before running them.

Publish checkpoints as they become available for the mechanism lane. Return up to three compact evidence packets, each with a neutral figure/prompt before interpretation, a counterexample or missing check, and exact reproduction instructions. Do not optimize for a positive grokking story.

## Mechanism/measurement explorer

Read research/SWARM.md and PRELAUNCH_REVIEW.txt. Own research/mechanisms/ on a dedicated branch. Reuse the size/time lane's registered checkpoints and exact data. While waiting, develop independently checkable analyses on synthetic examples and calibration checkpoints; label those outputs correctly.

Study frequency retention/removal curves, checkpoint-local versus retrospective component selection, rescaling sensitivity of compression/rankings, and targeted interventions with controls. Verify function-preserving rescalings before interpreting metric changes. Work on analysis copies; do not change training dynamics. Optional operand-subspace patching needs whole-operand and no-op sanity checks.

Keep parameter count, parameter norm, codec size, frequency concentration, prediction accuracy and margins distinct. A failed proxy is a useful result. Save counterexamples and supply up to three figures-first evidence packets with alternative explanations and reproduction commands. Do not claim compositional reuse from the relabeled-output readout probe.

## Independent auditor

Read research/SWARM.md. Own research/audit/ on a dedicated branch. Review the coordinator's common foundation before substantive training, then audit selected outputs. Verify separate data/init seeds, fixed dataset hashes, disjoint splits, gradients/AdamW, loss/margins, checkpoint round-trips, and logging semantics. Independently recompute selected results and short traces; identical imports or duplicated formulas are not independent validation.

Audit at least one ordinary and one surprising result. Check run accounting and figure provenance. Separate software PASS/FAIL/NOT CHECKED from whether an interpretation is locally supported, ambiguous or contradicted. Report actual checked scope; do not issue a blanket 'validated' badge. Flag concrete defects immediately and link affected run IDs. Preserve original failures. Return a compact audit suitable for Bobby's review, with reproduction commands and unresolved concerns.
