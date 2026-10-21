# Pre-run handoff — 2026-10-21

This commit imports the original NumPy prototype and adds operational measurements. It does not resolve all scientific review findings or authorize a confirmatory interpretation. Benchmark outputs are calibration only.

## Settled scope

Keep p=31, width=128, one hidden ReLU layer, float64, full-batch AdamW. Benchmark this workload before expanding width or architecture. The next scientific stage remains feasibility, including whether the random-label control can fit. Training steps/data exposure and elapsed compute are different budgets.

## Measurement additions

- End-to-end parent timing plus nonoverlapping training, evaluation, diagnostic and write phases.
- Runtime inventory: source and Git provenance, BLAS build/settings, CPU affinity, visible cgroup hierarchy, disk, process peak RSS.
- Split cross-entropy and signed margins alongside accuracy.
- Actual adaptive AdamW and decay update norms and alignment at sampled steps.
- Separate calibration seeds, per-child timeouts, partial-failure preservation, explicit process/BLAS concurrency.

## Scientific amendments still required

The original protocol/analyzer retain their historical definitions. Do not promote their composite decision to a conclusion about algorithm discovery. Before confirmation:

1. Replace fixed top-four sufficiency with prespecified retention/removal curves; address ReLU gauge dependence in frequency ranking and quantization.
2. Rename the relabeled-output transfer endpoint as readout accessibility; it does not establish compositional reuse.
3. Report endpoints separately; preserve fit/loss context for compression comparisons.
4. Include random-label fit feasibility and resolve/disclose the commutativity mismatch.
5. Distinguish checkpoint-local frequency selection from retrospective tracing of final frequencies.
6. Consider counterfactual operand-subspace patching as a mechanistic benchmark, not proof of cross-task reuse.

Burny's replication reports a strict key-frequency criterion passing only 6/11 grokking seeds; weak contributions can matter collectively. Its early restricted-loss trace uses final-checkpoint frequencies. Its raw-gradient/decay comparison motivates logging actual AdamW displacements here.
https://github.com/BurnyCoder/visualizing-grokking-modular-addition/blob/main/reports/exp-03-hypotheses.md
https://github.com/BurnyCoder/visualizing-grokking-modular-addition/blob/main/docs/operationalizations.md

## Execution evidence from ALM

Two Claude cloud sessions each exposed 4 vCPUs, 13.36 GiB process-cgroup memory and about 30 GB free disk. Two single-threaded FFT processes scaled nearly linearly; four-way evidence was limited. Same-harness FFT throughput varied 1.4–1.5x across sessions. These are observations of those sessions, not a service guarantee or a neural-training benchmark.

Separate project sessions may add capacity. Processes within a session share its resources. Benchmark the actual matrix-multiplication workload; cap BLAS threads before importing NumPy. Include diagnostic and checkpoint costs. On the M4, record its actual NumPy backend and measure locally rather than extrapolating FFT rates. Balance experimental conditions across hosts; do not confound architecture or width with host. Compare numerical correctness on short traces, not bitwise equality of long optimization trajectories across backends.

Pinned ALM report:
https://github.com/bfaber-centaur/artificial-life-museum/blob/ca260c1944bd531b0f882a517cad284c9547c3b1/research/reports/reef-environment-measurements-2026-10-20.md

## Locked runtime

Use `uv sync --locked` and `uv run --locked python ...`. CPython 3.12.14 and NumPy 2.3.5 are pinned; runtime inventory records hashes of the environment manifests. The lock includes platform-specific distributions, so backend/hardware differences remain part of provenance.
