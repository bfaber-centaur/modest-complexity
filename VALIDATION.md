# Implementation checks

The original validation below describes the earlier standalone package; its smoke artifacts are not imported into this repository. Current instrumentation evidence is recorded separately here.

## Instrumentation checks (2026-10-21)

- Existing self-test passes. Two new unit tests verify exact compatibility with the original AdamW update, displacement norms/alignment, zero-decay handling, and loss/margin calculations.
- Two 1,000-update calibration jobs ran serially and concurrently, with the standard 32 shams and complete diagnostics/checkpoint writes. Serial: 3.545 s; two processes: 1.822 s, approximately 1.95x aggregate throughput. These short timings are from the execution host, not an M4 or Claude session.
- All final weights and Adam moments match bitwise between serial and parallel execution for both calibration seeds. Timing phase sums are bounded by total runtime.
- A deliberately tiny subprocess timeout returns nonzero, marks calibration incomplete, and preserves the parent report.
- Summary, environment and source hashes: `benchmarks/pre-run-calibration.json`. Raw model checkpoints are excluded from Git.
- No pilot or confirmatory seeds were trained. Long-run timing, grokking feasibility and automatic resume remain unverified.

## Historical prototype checks

Only software checks and short smoke runs have been performed. There is no observed grokking result and no empirical conclusion about H1 through H4.

Verified using Python and NumPy 2.3.5 on the supplied CPU environment:

- Analytic backpropagation agrees with central finite differences for parameters in all three matrices.
- All pair indices appear in exactly one split, and reversed pairs never cross splits.
- Shuffled-label controls preserve exact label histograms in every split.
- Conjugating weights with input/output token permutations preserves semantic logits.
- The compression search produces candidates satisfying its training and calibration fidelity constraints.
- A short optimization run decreases training loss.
- Targeted and sham interventions execute and emit the declared number of measurements.
- A 100-update run at the full default model size exercises training, compression, transfer and checkpoint serialization.
- Timing and censoring logic, Wilson interval bounds, rejection of incomplete study runs, and rejection of failed audit fidelity were checked on constructed inputs.
- Sham row permutations preserve perturbation norm and singular values numerically; the launcher was checked in schedule-only mode.

An initial unstratified split left fewer than two training examples for one output class in the reuse assay. Before any full pilot or confirmatory data, the split was corrected to stratify unordered pairs by their sum. The original package’s retained smoke_verified run uses the final split. Its final training accuracy was approximately 0.702 and audit accuracy 0.0; at this duration these are pipeline checks, not hypothesis evidence. The earlier superseded smoke run is not included in the deliverable.

Remaining validation boundary: neither 50,000-step feasibility pilot has run. Whether this exact ReLU/AdamW configuration groks is unknown. The prototype does not prove the validity of the chosen complexity proxy, demonstrate subroutine reuse on a different operation, or implement a packed binary codec. Those limits are explicit in the protocol.
