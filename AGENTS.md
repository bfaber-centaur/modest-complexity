# Repository workflow

For this exploratory swarm, the coordinator integrates on `swarm/exploration` in `bfaber-centaur/modest-complexity`. Workers branch from the coordinator's pinned shared-code commit and open PRs in the same repository targeting `swarm/exploration`. Only the coordinator integrates worker changes after relevant checks. Do not push swarm work to either repository's main branch or merge upstream. Bobby's local work remains on main.

The coordinator returns the integrated review packet and may prepare a curated PR for Bobby's review against `bubskee/modest-complexity` main. Refresh upstream first and exclude unrelated or already-merged work. No upstream merge is authorized.

Read `research/LAUNCH_PROMPTS.md`, `research/SWARM.md`, and `research/RED_TEAM.md`. The current launch instructions supersede older main-branch routing and skill-import recommendations. Use the locked uv environment. Keep calibration, exploratory, and confirmatory runs distinct; read PRE_RUN.md before scientific work.

## Small, inspectable implementation

Optimize for a short path from experiment configuration to trustworthy figures. Reuse existing code and dependencies; prefer straightforward functions and explicit data structures. Add abstractions when they simplify concrete current uses. Defer speculative extensibility and experiment-management frameworks.

Test consequential scientific/numerical failure modes: incorrect labels, leakage, optimizer changes, misleading metrics, and corrupted checkpoints. Preserve readable names, useful errors, provenance, and correctness. No arbitrary line-count target or code golf.

Keep scientific rationale in research notes and code comments focused on assumptions, invariants, and non-obvious choices. At handoff explain changes and checks, justify new dependencies or substantial infrastructure, and identify obsolete code. The coordinator consolidates duplicate loaders, plot helpers, and configuration code before integration.

## Scoped skills and figures

No skill installation is required. Use the minimum figure standard in RED_TEAM.md: static Matplotlib first, individual seed trajectories, explicit missing/capped runs, comparable axes, source tables/run IDs, a regeneration command, PNG plus SVG or PDF, and visual inspection of the exported PNG. Preserve the project lock; only the coordinator changes dependencies.

Full scientific-visualization guidance is an optional reference for a concrete plotting issue. Do not load entire skill bundles into every worker. Do not introduce journal compliance, automatic statistical tests, power-analysis gates, or clinical evidence grading into this exploratory map.

The auditor checks provenance, numerical correctness, dependence, proxy validity, and competing explanations. Block only affected runs, metrics, or interpretations. Record adaptive decisions and preserve negative results.

## Local/cloud provenance

Give runs globally unique IDs including expedition/session and attempt, and record host/session labels separately from scientific configuration. A local replay of the same configuration and seed is a reproducibility check, not another independent seed. Never overwrite or pool it as an independent replicate.
