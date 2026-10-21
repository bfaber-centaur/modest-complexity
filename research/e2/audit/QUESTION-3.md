# e2 audit, cycle 3: question (written 2026-10-21, about 20:03 UTC, before any of the computations below)

Session `e2-au-cloud-1`, base `3b94f0f` (swarm/exploration after PR #18 and DECISIONS D10). Target: mechanisms packet 2 (`research/e2/mechanisms/unitfilter.py`, PR #17), and the coordinator's no-decay counterexample (`research/e2/coordinator/spot_unitfilter.py`, D10).

## Already seen when this was written

- **Packet 2.** I have read the README section, `unitfilter.py` and D10, but not the figure or the CSV rows. The claim: projecting every unit's U, V columns and W row onto its own dominant input frequency (keeping the DC of U and V) takes the 30% runs from 0.000–0.004 to 0.43–0.58 common-set accuracy.
  - The nulls stay near chance: random other frequency 0.064–0.078 (median), phases scrambled 0.032–0.037, step 0 0.029–0.064, shuffled-label runs 0.035–0.041.
  - The 50% runs filter to 1.000.
  - In the 30% runs the filtered accuracy is already 0.28–0.34 at step 1,000.
- **D10.** The no-decay runs filter to 0.466 / 0.325 / 0.416 (train 0.49–0.63), without nulls.
- **My cycle 2.** No-decay groups carry a weak aligned a+b−c term: 34/45 groups aligned, about one fifth of the decayed 30% relative amplitude, and a null restricted rank.
- **Not yet computed.** Nothing with any per-unit filter.

## Question

Is the filtered accuracy of the failed 30% runs evidence of a generalizing computation that weight decay built and that the rest of the network hides? Or is it what the filter extracts from any network fitted to a+b on these training pairs, a fit-induced interpolation in which the single-frequency projection supplies the prior?

## The mechanism I suspect, stated before testing it

A filtered unit j computes relu(A cos(ka+α) + B cos(kb+β) + DC) · C cos(kc+γ). One of its terms is cos(k(a+b) − kc + α+β−γ), which peaks at c = a+b when γ = α+β. A readout fitted to training pairs with c = a+b moves each unit's output row towards the label pattern of its own activations, which contains cos(k(a+b) + α+β). That sets γ ≈ α+β from the training labels alone, whatever U and V are. If so, the filter needs no learned input structure. A fitted readout on random features would be enough.

## Two explanations

- **H, a hidden computation.** Decay builds per-unit structure in the failed runs that carries the answer. Filtering needs that structure. A network with random inputs and a readout fitted to the same training pairs filters to near the step-0 null (0.03–0.06). The no-decay runs, with near-initial inputs (concentration 0.22), filter far below the decayed runs.
- **F, fit-induced interpolation.** The answer comes from fitting a+b on the training pairs plus the projection's prior. Random step-0 features with a readout fitted in closed form to the 299 training pairs filter to well above the null, comparable with the no-decay runs.

## Diagnostic

1. Reimplement the filter and its two weight-level nulls (random other frequency, phases scrambled) from the README definitions with my own code and seeds. Cross-check against `unitfilter.csv` on the packet's runs and steps: the filter rows exactly, the nulls as distributions.
2. Run the no-decay width-128 runs at 0, 1k, 5k, 20k and 50k with the same nulls, and check D10's three numbers.
3. **Constructed control, no optimizer and no training run.** For each init seed:
   - take the step-0 U and V of the width-128 runs and freeze them;
   - fit W in closed form (ridge) to the one-hot a+b labels of the 299 training pairs (30% split), with ridge λ = 1e-6 and 1e-2 times the mean eigenvalue of the feature Gram matrix, both fixed now;
   - score the readout unfiltered and filtered, with the same nulls.
4. Report the gate flips caused by the input projection (fraction of (pair, unit) gates that change) and the in/out dominant-frequency agreement, per run.

**Decision rule, fixed now.** F is supported if the constructed control's filtered common-set accuracy is at least 0.2 at either λ and above the maximum of its own phase-scrambled draws. H is supported if it stays within the step-0 null range (≤ 0.064).

## Failure modes

- The ridge readout on 128 random features may not fit all 299 training pairs. Its train accuracy is reported.
- Random-feature units are weakly concentrated (0.17), so the filter keeps little energy. That works against F, so a positive result is conservative.
- The accuracy of a filtered network is not binomial over pairs. A few summed frequency components with random phases decide it for all pairs at once, so nulls must be judged by their draw-to-draw spread, not by 1/31 ± a binomial error.
- The closed-form readout is not what gradient descent with weight decay produces. A positive result shows that the training labels suffice, not that the trained runs got their filtered accuracy this way.
