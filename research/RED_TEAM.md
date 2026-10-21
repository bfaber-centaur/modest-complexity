# Swarm red-team review — 2026-10-21

Scope: inspect the launch design and existing code for operational failure modes. No workers or experiments launched. The four-role organization and exploratory intent remain unchanged.

## Highest-priority corrections

### 1. Stage 0 can consume the expedition

Seed separation, a registry, offline diagnostics, independent checks, resource measurements and reporting could turn into a framework-building exercise. The plan also asks for offline-vs-inline agreement before training even though basic learning curves do not depend on offline analysis.

First-figure path: separate dataset/init randomness; retain the existing gradient/split checks; check optimizer updates and checkpoint round-trip; lock a minimal Matplotlib dependency group; then train the first assigned seed at widths 32, 128 and 512. Use their normal 50k maximum and render partial curves as logs arrive. Continue those same jobs; do not introduce shortened runs that require unimplemented resume. Show incomplete endpoints explicitly. Do not wait for 24 runs, gauge normalization, Fourier interventions, a full audit, or polished prose to publish a preview.

The first useful deliverable is one seed-level training/test accuracy figure and one cross-entropy figure, plus a status table. Put the other widths and seeds back into the mixed queue after the initial three. No grokking is required to make a useful figure.

If no preview appears within roughly 20 minutes after dependency setup, write a concrete blocker report and simplify the implementation path. This is an attention checkpoint, not a training endpoint or a reason to skip correctness checks. Offline analysis and its parity checks can follow.

### 2. Cross-session sharing is currently underspecified

A path in one Claude session is not available in another. Before relying on shared checkpoints, complete one end-to-end handoff: producer publishes a small checkpoint, config, exact data and hashes on its owned branch; consumer fetches that commit into its own workspace, verifies the hashes, loads the files and reproduces a basic metric.

For the first pass, use a small curated checkpoint set in Git with a manifest (run ID, step, producing code commit, artifact commit/path and checksum). The producer updates the manifest at useful milestones rather than committing every checkpoint. Bulk archives can use an explicitly agreed artifact store later. A hash without accessible bytes is not a handoff. Parent logs/configs must travel with model state. Integration of every data commit into main is not required before analysis.

### 3. Checkpointed is not resumable

The existing experiment saves Adam state but has no resume command. Instructions to use bounded chunks must not lead a worker to restart weights, reset optimizer state or concatenate independent trajectories as if continuous.

Prefer complete bounded subprocess jobs after measuring their expected runtime. If a job hits a resource limit, retain it as capped. Implement real resume only when it solves an observed execution need, and verify against an uninterrupted short trajectory before use. Do not assume shell background lifetime from ALM's untested timeout documentation.

### 4. The mechanism lane has too many simultaneous objectives

Retention curves, retrospective selection, compression, gauge normalization, shams and operand patching could each absorb a session.

First mechanism deliverable: one frequency-retention curve on one late checkpoint, plus a counterexample showing whether function-preserving rescaling changes a proxy. Current-checkpoint selection first. Normalization, retrospective traces and patching are subsequent choices, not prerequisites. Publish negative or uninformative results rather than expanding until the story looks interesting.

### 5. Audit can become either a rubber stamp or a global stop sign

The auditor should check split identity, gradients/optimizer and checkpoint integrity early. A real defect in those blocks the affected training or interpretation. An unresolved gauge convention blocks a simplicity interpretation, not an accuracy plot. A rendering problem blocks that figure, not all jobs. State the affected runs/metrics and continue unaffected work.

Use a small independent calculation or rerun with an explicit comparison, not 'tests pass' as evidence for every claim. Keep longer scientific criticism alongside provisional figures; do not wait for consensus before sharing observations.

### 6. Familiar narratives can steer the swarm

The fixed recipe may generalize immediately, never generalize within budget, or collapse; the agents must not tune silently until they obtain textbook grokking. Report the initial map first. A recipe rescue is a new labeled exploratory comparison with old failures retained. Observe loss and margins as well as accuracy. A minimum at the end of a short trace is not a convergence claim.

Likewise, do not spend the expedition collecting literature instead of inspecting available trajectories. Literature lookup should resolve a concrete interpretation question and return to the data.

## Figures: minimum useful standard

Use static Matplotlib figures first. Add it through the existing uv project as an analysis dependency group, lock it once, and use that environment for rendering. Do not copy a skill's standalone Python/version pins over the project's pins. No dashboard, browser automation, Kaleido or journal-format work is needed for the first review.

- One shared plotting script/style, consistent width colors, labeled seed traces or seed facets.
- Same axes/color normalization for directly compared panels; accuracy on 0..1 with a 1/31 chance reference when appropriate.
- Label log-loss axes. Handle zero/negative values explicitly; signed margins need a linear or disclosed symmetric-log scale.
- Do not smooth the first-pass trajectories. Show missing/capped portions as gaps, not zeros or interpolated continuations. Handle step zero explicitly on logarithmic time axes.
- With three seeds, prefer individual traces to apparently precise confidence ribbons. Do not use a plotting library's row-wise default confidence interval on repeated checkpoints.
- Every figure has underlying tabular data, run IDs, plotting command and a short neutral caption. Open the actual exported PNG to inspect labels/clipping before handoff.
- Save PNG for quick review and SVG or PDF for zooming. A local interactive Plotly HTML can follow if scrubbing or toggling traces answers a real question; it must not delay static output.

## Skills for the Claude sessions

Recommended: `scientific-visualization` for the two explorers and auditor; `scientific-critical-thinking` for the auditor. The visualization skill was inspected for this review: it covers trace/replicate handling, honest axes, missingness, consistent normalization, accessible encodings and exported-file inspection. Apply its general figure guidance; defer its publisher-specific workflow.

Optional later: `statistical-analysis` when a concrete comparison needs it. Do not turn its availability into a requirement to add p-values or significance claims to this exploratory map.

Import the portable skill packages into the environment where Claude actually runs; skills available in this ChatGPT session are not automatically available to those workers. Include referenced resources/helpers if used. No skills have been installed into Claude by this review. A skill is guidance, not a substitute for the project lockfile, readable plotting code or inspecting exported results.

Workflow attribution: the figure review used the Scientific Visualization skill from Kassis, Agarwal, He, Patel and Brueckner (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, https://doi.org/10.48550/arXiv.2609.00065 . This credits procedural guidance, not evidence that the swarm or scientific claims are validated.
