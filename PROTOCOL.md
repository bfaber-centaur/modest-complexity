# Compact reusable computation in modular addition

Preregistration candidate v1 • 20 October 2026 • No confirmatory runs collected

This experiment tests whether a small network trained on modular addition develops a compact computation that generalizes to unseen pairs, survives a fixed compression procedure, and supports a new readout without retraining its hidden computation. It separately tests a specific Fourier mechanism. It does not equate low weight norm with a short program or claim that a positive result establishes general program synthesis.

The starting claim is Furman's distinction between observed algorithmic circuits and the stronger hypothesis that optimization reliably discovers simple programs [1]. The experiment makes a local version falsifiable: **under the architecture, data distribution, optimizer and budget below, structured addition should yield a more compressible and reusable solution than a matched memorization task, with identifiable Fourier components causally supporting performance.** “Causally” here means an intervention changes the trained model's behavior; it does not mean that those components caused gradient descent to discover the solution.

## Prior work and the remaining question

| Work | Established result or proposed account | Consequence for this experiment |
| --- | --- | --- |
| Power et al. 2022 [2] | Delayed generalization on small algorithmic datasets | Measure the whole training trajectory, not just the endpoint. |
| Nanda et al. 2023 [3] | A Fourier modular-addition circuit, with memorization, circuit formation and cleanup phases | Final compressibility and pre-grok structure are already known phenomena. The proposed combination of fixed coding, matched interventions and frozen reuse is a controlled extension, not a claim of first discovery. |
| Varma et al. 2023 [4] | Circuit efficiency in terms of logits per parameter norm; predictions including ungrokking and semi-grokking | Separate regularization from implicit optimizer bias. Norm efficiency is not description length. |
| Zhong et al. 2023 [5] | Different initializations and hyperparameters produce Clock, Pizza and more complex solutions | Failure of a four-frequency diagnostic is evidence against this particular mechanism, not every possible algorithm. |
| Mohamadi et al. 2024 [6] | Results on kernel-regime limitations and generalization of bounded-norm quadratic networks | There are already restricted theoretical accounts of discovery. This ReLU/AdamW experiment is not a test of all their assumptions or theorems. |
| He et al. 2026 [7] | Fourier feature emergence, phase alignment, diversification and a lottery-ticket account in two-layer models | The “why can gradient descent find it?” question has partial answers. This protocol tests robust observable consequences; it does not replace their dynamics analysis. |
| Sivasankar 2026 v3 [8] | Basis choice changes synchronization timing; the revision withdraws v1's causal and predictive claims | An early spectral signal alone is insufficient. Freeze basis selection and use perturbation controls. |
| Arora et al. 2018 [9] | Compression-based generalization bounds using explicit compressed representations | Count an executable representation's bits, but do not present this empirical codec as their theorem or a validated generalization bound. |
| Shah et al. 2020 [10] | Simplicity bias can favor less predictive features and harm robustness | “Simpler” is not synonymous with “correct” or “useful.” Report complexity and accuracy separately. |

## Hypotheses and distinct alternatives

H1, operational simplicity: at the fixed final budget, addition models are at least 20% shorter under the specified fidelity-preserving codec than their matched random-label models, conditional on both fitting training data. Compression must preserve model behavior on the untouched audit pairs as well as on calibration pairs.

H2, specific mechanism: four weight-selected Fourier frequency pairs suffice for at least 95% addition audit accuracy, and removing these components damages accuracy at least 10 percentage points more than the median matched sham perturbation.

H3, minimal reuse: a new linear readout of frozen addition hidden activations reaches at least 80% audit accuracy on a fresh output-label permutation from 62 examples, beating both untrained and shuffled-activation controls by at least 20 percentage points.

H4, temporal formation, is secondary: a retained Fourier computation reaches 90% audit accuracy at two consecutive scheduled audits while the unmodified network is still below 50% at the first of those audits. This supports formation of a useful component before overt generalization. Absence of H4 does not by itself reject H1–H3.

The alternatives are separable. Pair memorization predicts training success without held-out success or frozen reuse. A compact representation visible only after generalization is consistent with later cleanup. A spectral pattern that fails the causal contrast is an inadequate mechanistic explanation. A causal circuit without a coding advantage supports learned computation but not H1. Success only with decay supports the specified regularized training system, not unqualified “gradient descent prefers simple algorithms.”

## Experimental units and arms

The independent replicate is a seed block, not a pair, neuron, checkpoint or sham. Use eight confirmatory blocks, seeds 2000–2007, with four arms in each block:

| Target | AdamW weight decay | Purpose |
| --- | --- | --- |
| Addition | 1.0 | Primary hypothesis arm |
| Matched shuffled labels | 1.0 | Memorization control |
| Addition | 0.0 | Dependence on explicit regularization |
| Matched shuffled labels | 0.0 | Regularization by target interaction |

All four arms share initial weights, input pairs, split membership, learning rate, training steps and architecture. Shuffle labels independently within each split while preserving its exact class histogram. This destroys algebraic structure in expectation; it does not claim to sample an incompressible function. The random task is generally noncommutative, a remaining mismatch. Grouping swapped pairs in the same split prevents commutativity alone from leaking answers across splits. A future symmetric random-table control would require a separately frozen design.

Generate one randomized 32-run order with NumPy RNG seed 20261020 and execute sequentially on one machine. Record the order before launch. Match inference and optimization budgets, not achieved audit accuracy. Report the random arm's failure to interpolate as a control failure; do not silently extend its budget or replace its seeds.

## Data and model

Use p = 31 and all 961 ordered pairs. For each answer class c, enumerate the 16 unordered pairs satisfying a+b=c mod 31. Shuffle those groups, assigning five groups to training, two to compression calibration, and nine to audit. Both orientations stay together. Each training answer has nine or ten ordered examples, ensuring availability for the reuse probe. Exact counts vary with diagonal allocation and are saved in data.npz. All residues occur in training; this is pair generalization, not extrapolation to new numbers or a new modulus.

Use the two-layer ReLU network

`h(a,b) = ReLU(U[a] + V[b]); logits(a,b) = h(a,b) W`

with U,V of shape 31 by 128 and W of shape 128 by 31, no bias, no tied embeddings, no preinstalled Fourier features, no dropout and no normalization. Initialize U,V with independent N(0,0.2²) entries and W with N(0,1/128) entries. Initialization seed is block seed + 10000. The architecture itself composes operations, but this is not taken as evidence that the trained model discovered a useful algorithm.

Train full-batch cross entropy for 50,000 updates with AdamW, learning rate 0.001, betas (0.9,0.999), epsilon 1e-8, decoupled decay on all matrices, and float64 arithmetic. No scheduler, clipping, early stopping or tuning on audit performance. Full-batch AdamW is a gradient-based optimizer, not vanilla gradient descent or SGD. The experiment's conclusions retain that qualification.

## Pilot and freezing rule

Before confirmation, run addition with decay 1 on pilot seeds 1000 and 1001 using the full budget. The feasibility gate requires both to interpolate at 99% training accuracy and at least one to show delayed generalization: audit accuracy below 20% at first interpolation, followed at least 1,000 steps later by sustained 95% accuracy. If the gate fails, stop this protocol, report the failed gate, and write a new version before trying another modulus, architecture, split or training budget. Pilot seeds never enter confirmatory statistics.

Freeze this document, experiment.py, analysis.py, run_study.py, requirements.txt and their SHA256 manifest before pilot execution; publish/register a second immutable record after the gate recording “unchanged” or the new protocol version. If unchanged, run all 32 confirmatory jobs without inspecting their intermediate audit curves to change settings. A pilot is not a favorable seed selection procedure for the final sample.

The delivered package has only implementation smoke checks, including 100 updates on seed 1000, far short of the feasibility budget. These checks influenced a split correction to guarantee class coverage and are disclosed in VALIDATION.md. No grokking or hypothesis result is claimed. The 100-update run is not a confirmatory observation.

## Measurement schedule

Log unmodified training, calibration and audit accuracy, elapsed seconds and total parameter norm at step 0 and every 100 steps. First interpolation tM is the first logged training accuracy at least 99%. Grokking tG is the first of three consecutive 100-step logs with audit accuracy at least 95%, provided tG−tM is at least 1,000 steps. Also report ordinary generalization time without the delay criterion.

At step 0, every 1,000 steps, the first interpolation checkpoint, and step 50,000, save weights and Adam moments and run the three audits below. The extra tM checkpoint is included in complexity comparisons but excluded from H4's two-consecutive-scheduled-audits rule. Archive all metric rows, not just successful runs. Evaluate scheduled probes with fresh fixed RNG streams so they do not change training randomness.

## Operational complexity

Use a finite, executable model-description family, fixed before seeing outcomes. Rank hidden units by `(norm(U[:,j]) + norm(V[:,j])) * norm(W[j,:])`; ties use original order. This ranking is invariant to a positive reciprocal ReLU rescaling of each hidden unit. Retain k in {8,16,32,64,128}. Independently quantize each retained matrix using its maximum absolute value, symmetric integer rounding, and q in {4,8,16,64} bits; q=64 stores the original float64 values. No gradients, refitting, distillation, rotation search or task-specific code replacement are allowed.

The fixed decoder uses 32 bits each for p and k, 8 for q, three 64-bit scales, then 3pk fixed-width weights. Thus `L(k,q) = 264 + 3pkq` bits. The decoder and activation convention are common to all candidates and conditions. Reordering retained hidden units requires no index list: the decoded network contains only those units. This is a specified code length, not the zip/npz file size. The prototype evaluates the decoded networks and bit count; it does not emit a packed bitstream.

Choose the minimum L candidate satisfying, separately on training and calibration pairs, at least 99% agreement with the uncompressed model's argmax and at most 0.01 relative squared error in class-centered logits. Relative error is sum of squared discrepancies divided by the reference's summed squared centered logits, floored at 1e-30. Selection uses reference outputs, not target labels. Resolve ties by k then q. The dense float64 candidate is a fallback. Report audit fidelity after selection without revising the choice.

Let C(t) be that code length. Primary complexity comparison is `C_add(50000)/C_random(50000)` within each decay-1 block, requiring both models' training accuracy at least 99% and both compressed models to meet the same audit fidelity thresholds. Also report each arm's `C(50000)/C(tM)`, retained k and q separately, and raw model size. A quantization-only gain should not be described as removal of computational steps.

This is an upper bound within one restricted coding family. It is coordinate- and tolerance-dependent, not Kolmogorov complexity, minimal circuit size, or the description length of the mathematical target. A lookup table and `return (a+b)%31` describe the external function under other languages; replacing a model with either would not measure what its weights implement. No unseen-pair prediction is counted as an independent statistical replicate.

## Intervention and matched shams

Take the real discrete Fourier transforms of U and V along their token axis. Rank positive frequencies 1–15 by joint squared coefficient magnitude across both matrices and all hidden units. Select the top four, always retaining conjugate pairs through real FFT reconstruction. Selection never uses audit labels. This is a known task-aligned diagnostic, not blind circuit discovery.

For each input matrix, reconstruct delta from only those four frequency pairs. Retention replaces the matrix with its row mean plus delta; removal subtracts delta from the original. Keep W unchanged. Record baseline, retained and removed audit accuracies. No magnitude rescaling follows the intervention.

Construct 32 shams by independently randomly permuting the token rows of each delta and subtracting these shuffled deltas instead. Each sham has exactly the same perturbation Frobenius norm, rank, singular values and mean per matrix as the targeted perturbation. It destroys alignment with token identities. Its orientation relative to the original weights differs; that is part of the intervention, not a perfectly isolated biological-style lesion. Report every sham result. Use the same sham permutations across paired arms and checkpoints (seed + 30000).

Define selective damage D as median(sham audit accuracy) minus targeted-removal audit accuracy. H2 requires D at least 0.10 and retained accuracy at least 0.95. Apply the same diagnostic to random-label and untrained models as negative controls. Fourier concentration alone is never a success criterion. The shams test whether the particular alignment matters beyond perturbation size; they do not identify every individual arithmetic operation.

As an implementation control, conjugate U,V and W with a random permutation of input/output token IDs and verify identical logits in semantic coordinates. The included self-test performs this check. This exact relabeling is not an independent experiment; it prevents interpreting the numeric names of token IDs as an inductive advantage. Any Fourier analysis of an encoded version must first restore semantic order.

## Frozen reuse probe

Define a fresh output bijection pi and target pi((a+b) mod 31). At every audit, freeze U and V, discard W, and fit only a linear ridge readout on hidden activations. Sample two source-training pairs per sum class, 62 total, without replacement, with seed + 40000. Reuse the identical examples, output bijection and fit budget across all arms and checkpoints. Evaluate on source-audit pairs, which were never used for source training or the new readout.

Standardize features using only those 62 fitting examples, flooring standard deviations at 1e-6, append an intercept, and use ridge lambda=1 on all coefficients including the intercept. Solve the dual system exactly. No validation sweep. Controls are the initial network's hidden activations and a permutation of all pair rows of the trained activation matrix; row shuffling preserves its entire singular spectrum while breaking pair identity. Use the same preprocessing and solver. Random-label pretrained networks form a further paired control.

Define reuse gain R as trained accuracy minus the larger of initial and shuffled-control accuracies. H3 requires trained accuracy at least 0.80 and R at least 0.20. Freeze the representation even if this fails.

This is deliberately a **minimal reuse assay**: learning new names for sums, not learning subtraction or a new algorithm. A fully generalizing representation may make this task easy by construction. Its role is to reject purely pair-specific features and distinguish accessible reusable information from a post-hoc replacement algorithm. It cannot, on its own, establish discovery of modular subroutines or compositional transfer to a new domain.

## Analysis and decision rules

Primary endpoint is the fraction of the eight decay-1 blocks satisfying all of H1–H3 plus uncompressed final addition audit accuracy at least 95%. Count non-generalizing and non-interpolating blocks as failures for this all-block success rate. Mark missing/invalid complexity comparisons separately; never interpret those as evidence that the algorithm was complex. Report a Wilson 95% interval for the success fraction. Six or more successes is the predeclared pattern supporting the local account; two or fewer, if at least six blocks have interpolated in both arms, is a strong contrary pattern. Three through five is mixed. These practical cutoffs are not a powered significance test or a claim about universal frequency of discovery.

For continuous outcomes, report all seed values and the paired means of log2(C_random/C_add), D, R, and final accuracy. Use 10,000 block-bootstrap draws with RNG seed 20261022 for descriptive 95% percentile intervals; preserve all four arms when resampling blocks. With eight blocks the intervals will be imprecise. There is no formal null-hypothesis rejection claim, so no hidden multiplicity-adjusted “winner” is selected. Mark secondary timing and decay-interaction analyses as secondary.

For H4, report eligible seed counts and the actual scheduled times, including right-censoring at 50,000 steps. No discovery time for non-grokking runs is imputed as if it were observed. Report runtime and updates to generalization as finite-budget search costs; do not infer asymptotic efficiency from a single p. Smaller code does not imply faster training.

| Observed outcome | Interpretation committed in advance |
| --- | --- |
| High training accuracy, persistently poor audit accuracy | Pair memorization remains adequate within this budget; local reliable-discovery claim fails. |
| Good generalization but no coding advantage over matched interpolating random controls | Evidence against H1 under this code; algorithmic behavior may still exist. |
| Compact final model, no pre-grok retained computation | Compatible with later cleanup; does not establish early circuit discovery. |
| Fourier structure appears early but targeted damage is no larger than matched shams | Spectral timing is not sufficient mechanism evidence; H2 fails. |
| Fourier retention fails but generalization and frozen reuse succeed | Specific four-frequency account fails; do not relabel this “memorization.” |
| Frozen trained features do no better than initial or shuffled controls | H3 fails; do not claim measured reuse from compression alone. |
| Addition and random models compress equally at matched fit | No measured simplicity preference in this coding family. |
| Effects occur only with decay 1 | Supports a regularized-system account; cannot attribute the result to optimizer bias alone. |
| Random controls do not fit or compressed audit fidelity fails | Comparison is unresolved; retain the failures and amend only in a new study. |

Exclude only unreadable/corrupt files or verified implementation errors. Log hardware interruptions and rerun the same seed/config from scratch; do not substitute seeds. Nonfinite training is a recorded failure. Any implementation change after confirmation begins invalidates pooling across versions and requires rerunning all affected paired blocks. No early stopping for favorable effects and no extension beyond eight blocks based on intervals.

## Scope and next experiment

A positive conjunction supports a compact, causally relevant, minimally reusable computation discovered by this training system. It remains compatible with several mechanisms of optimization and several effective algorithms. A negative result identifies which link failed rather than rejecting all program-synthesis views.

The most useful follow-up, only after this experiment, would put two competing rules of different registered description lengths on the *same training observations*, then test which off-training continuation optimization selects. That would address simplicity preference more directly than comparing two target distributions. A separate modulus/width scaling study would be needed for claims about discovery efficiency. Neither is included in the present budget.

## References

1. Zach Furman. Deep Learning as Program Synthesis. January 20 essay supplied by the user. https://www.lesswrong.com/posts/Dw8mskAvBX37MxvXo/deep-learning-as-program-synthesis-1
2. Alethea Power et al. Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets. 2022. https://arxiv.org/abs/2201.02177
3. Neel Nanda et al. Progress measures for grokking via mechanistic interpretability. 2023. https://arxiv.org/abs/2301.05217
4. Vikrant Varma et al. Explaining grokking through circuit efficiency. 2023. https://arxiv.org/abs/2309.02390
5. Ziqian Zhong et al. The Clock and the Pizza: Two Stories in Mechanistic Explanation of Neural Networks. 2023. https://arxiv.org/abs/2306.17844
6. Mohamad Amin Mohamadi, Zhiyuan Li, Lei Wu, Danica J. Sutherland. Why Do You Grok? A Theoretical Analysis on Grokking Modular Addition. ICML 2024. https://proceedings.mlr.press/v235/mohamadi24a.html
7. Jianliang He, Leda Wang, Siyu Chen, Zhuoran Yang. On the Mechanism and Dynamics of Modular Addition: Fourier Features, Lottery Ticket, and Grokking. 2026. https://arxiv.org/abs/2602.16849
8. Achyuthan Sivasankar. Learned Structure Shows Up Earlier in the Task's Own Basis: A Controlled Look at a Synchronisation Precursor in Grokking. Version 3, October 11, 2026. https://arxiv.org/abs/2606.12966v3 . Earlier title and causal claims are superseded.
9. Sanjeev Arora, Rong Ge, Behnam Neyshabur, Yi Zhang. Stronger Generalization Bounds for Deep Nets via a Compression Approach. ICML 2018. https://proceedings.mlr.press/v80/arora18b.html
10. Harshay Shah et al. The Pitfalls of Simplicity Bias in Neural Networks. NeurIPS 2020. https://arxiv.org/abs/2006.07710

Design workflow used the Experimental Design skill from Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel and Aubrey M. Brueckner, Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents (2026), https://doi.org/10.48550/arXiv.2609.00065 . This is a workflow attribution, not scientific evidence for the hypotheses.
