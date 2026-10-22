# e4 mechanisms: question log

Written 2026-10-22 at about 02:15 UTC on `swarm/exploration` `6b8d6c1` (e3 closed, D17), before any e4 measurement and before the coordinator's e4 pin. Session label to be assigned by the coordinator. This is an exploratory log, not a preregistration.

## Question

Between the post-filter accuracy peak and the collapse, what changes in the computation, and which of those changes costs the correct answers?

## Prior observations (e3, audited; three sources, one split, width 128)

- **The edit.** The per-unit dominant-frequency filter lifts the three failed 30% networks from 0.000–0.004 to 0.43–0.58 on the 483 common pairs. It also flips 28–35% of ReLU gates (e2 audit G17), so it builds a new network rather than exposing an untouched one. Random features with a fitted readout, filtered the same way, reach only 0.14–0.30 (e2 audit cycle 3).
- **The rise.** Training adds 0.26–0.31, peaking at updates 68–87, 18–37 updates after train accuracy first reaches 1. It survives Adam at lr 1e-4 and plain gradient descent (audit investigation 4), so it comes from fitting from the edited point.
- **The fall needs decay.** At wd 1 the gain is gone in accuracy by 5,000 updates (0.010–0.031 at 20k). At wd 0 the same start files keep 0.56–0.62 at 20k, about 0.2 below their peak.
- **The ranks keep a trace.** At 20k, B at wd 1 has P(a+b over S) 0.39–0.45, against 0.14–0.18 for A. B ends as a different fitted network, with argmax agreement with A of 0.42–0.50, not as the original.
- **Fit is not the driver.** At wd 0 B fits the training pairs about 1000× harder by cross-entropy and still keeps its answers. Gradient descent with decay erases while its train cross-entropy rises (audit, post hoc).
- **Uniform shrinkage is not enough.** The network has no biases, so shrinking every weight by one factor changes no argmax. Decay can only act through its interplay with the gradient (audit investigation 4).

## Rival explanations and what each predicts

The object is B at wd 1, compared with the same start files at wd 0 and with A (the unedited memorizer) as reference.

| explanation | what it predicts between the peak and about 2k updates at wd 1, but not at wd 0 |
|---|---|
| E1. **The useful component shrinks.** | The held-out margin carried by the reference channel (below) falls in absolute size. The units' weights at their reference frequency shrink. Other channels do not grow much. |
| E2. **Competitors overtake.** | The reference channel's held-out margin holds roughly steady while the other channels' contributions against a+b grow past it. |
| E3. **Alignment or gating changes.** | Per-unit reference-frequency weight amplitudes hold, but the units' contributions to the a+b direction lose phase coherence and cancel. Or gates change, so the same weights compute differently. |
| E4. **A shared label offset masks the answer** (Janati et al. 2026, mean term). | The label-only part of the logits (the training-mean feature times the readout) grows, and removing it alone restores much of the accuracy. |

Combinations, or something else, are possible; nothing here has to win.

## Measurements (first figure)

The object is the logits of each checkpoint on the 483 common pairs, plus the weights. Each quantity was chosen because the explanations above predict different values for it.

- **Fixed reference.** Each unit's reference frequency k_j is its frequency at update 0, the one the filter kept. It is fixed for the whole trajectory and never re-selected per checkpoint. As a check on definition-dependence, the main measures are repeated with each checkpoint's own dominant frequency.
- **Channel decomposition (exact).** Write the logits as z = Σ_j h_j W_j. Here μ is the mean hidden activation over the 299 training pairs, and W_j splits into its part at frequency k_j along the label axis (W_j^ref) and the rest (W_j^off). The label-axis DC is dropped because it is constant across labels. Three channels:
  - **label offset** μ·W: shared by all pairs, Janati's mean term;
  - **reference** (h − μ)·W^ref;
  - **off-reference** (h − μ)·W^off.
  The channels must sum to the centred logits; the identity is checked numerically and its maximum error reported. The decomposition is applied to linear margins on the common pairs, which add exactly across channels: z_{a+b} minus the mean logit over S, and over N. Max-based measures do not decompose, so accuracy and the D6 pairwise measures are reported alongside but not split.
- **Weight level, at k_j.** The total amplitude of U and V and of W at each unit's reference frequency, and the total off-reference energy. Both are absolute, so shrinkage in the weights is visible.
- **Coherence.** For each frequency group, the per-unit contributions to the a+b mode e^{ik(a+b−c)} of the logit table are complex numbers that sum exactly to the group's total. The index is |Σ_j A_j| / Σ_j |A_j|: 1 means the units are aligned, near 0 that they cancel.
- **Gates.** The fraction of (pair, unit) ReLU gates that differ from update 100.
- **Masking check, evaluation-only.** Accuracy with the label-offset channel removed.

**Times.** The fixed reference state is update 100. It is the first published checkpoint at which train accuracy is 1 in all six runs, a label-free rule, and it sits 0.008–0.009 below the replayed held-out peaks. All published checkpoints are used: 0, 10, 30, first fit, 100, 300, 1k, 2k, 5k, 10k, 15k and 20k. The fall happens between 300 and 2,000 with only two published states inside it, so a minimal dense state record over 100–3,000 is requested from the coordinator (below).

**Not first.** These checks are left out of the first figure to avoid measuring everything:
- swapping features and readouts between times;
- training-only readout refits (decodability);
- full 3D Fourier family decomposition of the logit table.
Any of them may be added if the first figure leaves a specific question that they answer.

## How the comparison with Janati et al. is limited

They filter cached representations, the residual stream, in a transformer trained with Muon. They find collapse at the readout interface, with the information still linearly decodable. Our filter edits pre-ReLU weights, which changes gates and can create a new computation. Our network has one hidden layer and no biases, and is trained with AdamW at decoupled decay.

- **What is borrowed.** Only the diagnostic question: did the useful computation fail, or is it present but masked (for example by a shared class offset)?
- **Phase coherence.** It follows He et al. (2026) only descriptively. Their theory is for quadratic activations with one shared input vector per unit. Our units have separate U and V and use ReLU, so the alignment condition is measured, not assumed.

## Expectation (stated before measuring; held loosely)

- At wd 1, E1 and E2 combine. The reference channel's held-out margin falls in absolute size from about 300 updates as decay removes what the gradient no longer replenishes once the training pairs are fit. Meanwhile the off-reference channel's margins against a+b on held-out pairs grow.
- Coherence falls somewhat, but does not lead.
- The label offset is small and not the cause (E4 unlikely).
- At wd 0 the reference channel holds or grows.

## What would change the next step

The one adaptive intervention is proposed to the coordinator after the first figure, chosen from it. Examples:
- If E1 leads, restore the update-100 reference channel at a later state, with a matched sham channel.
- If E2 leads, freeze or remove the competing channel.
- If E4 holds, remove the offset during training.
Intervention strengths and selections will be fixed without held-out feedback.
