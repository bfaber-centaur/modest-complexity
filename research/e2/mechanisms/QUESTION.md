# e2 mechanisms: question log

Written 2026-10-21 at about 19:35 UTC, before any of the measurements below were made. This is a prospective exploration log, not a preregistration. Base `9e56faf` (e2 base `6aaa3a2`); shared code `fe284a5`. Session label `e2-me-cloud-1`.

## Already seen before writing this (e1 mechanisms lane, PR #12)

- Per-unit frequency concentration (`research/mechanisms/units.py`) for the six width-128 runs (30% share: e1 i30000–i30002; 50% share: e1f2 i30000–i30002). At 50k, units in the 50% runs put 0.83–0.86 of their input energy in one frequency, and input and output dominant frequencies agree for 0.99–1.00 of units. The 30% runs reach 0.40–0.44 and 0.66–0.75. All six cover all 15 frequencies. The shares separate by step 1,000.
- Aggregate-frequency retention and removal (input filtering) and the restricted rank readout on 50% i30000 at 5k, 10k, 20k and 50k. Held-out accuracy moves roughly linearly with the number of frequencies kept, and the top-energy frequencies are not special.
- I have not computed anything per unit group: no group logit contributions, no unit ablations, no common-set (483-pair) metrics of my own. The coordinator's common-set accuracies (`research/e2/coordinator/common_set_scores.csv`) are known to me.

## Question

Does the weak per-unit frequency structure in the failed 30% runs compute the same kind of thing as the strong structure in the successful 50% runs? Specifically: do the units grouped by dominant input frequency k contribute a logit term that depends on the inputs through a + b − c, the way an addition computation needs, and does removing a group affect correct answers (50%) and the restricted preference (30%) more than removing size- and norm-matched random unit sets?

## Explicit candidate computation and what it predicts

In this model a unit's pre-activation is U[a] + V[b], a sum of separate operand terms. For a unit with u_j(a) = cos(ka + α) and v_j(b) = cos(kb + β), the sum is 2 cos(k(a+b)/2 + (α+β)/2) cos(k(a−b)/2 + (α−β)/2). This is the "pizza" form, which Yip et al. 2024 derive from an integral over phases (§4.2–4.3, eq. 5–6) and which Zhong et al. 2023 call Pizza (§3, Model A, "Correct logits ... have a clear dependence on a − b", Fig. 3). A group of units at frequency k with spread phases then gives

logit_k(a, b, c) ∝ |cos(k(a − b)/2)| cos(k(a + b − c)).

The reference levels for an ideal group follow from this form:

- Over all 961 pairs, its share of variance (after removing each pair's mean over c) explained by a function of d = a + b − c mod p is E[|cos x|]²/E[cos² x] = 8/π² ≈ 0.81. A pure "clock" term cos(k(a+b−c)) would give 1. Structureless contributions would give about 1/(number of pairs).
- That function of d is a cosine at frequency k peaking at d = 0, the correct label.
- Removing one ideal group removes one term out of about 15 that add up at the correct label. Held-out accuracy should fall only a little, and a random unit set of the same size removes about the same total amount spread over all frequencies.

The reference values are recomputed on each evaluation set by evaluating the ideal term there (the 483-pair set's a − b distribution is not uniform).

## Two explanations

H1, a shared but partial computation: the 30% runs' groups also contribute pizza-like terms at their frequency, peaking at d = 0, only weaker or mixed with a non-equivariant part (the memorized table) that wins the argmax. Their restricted preference is carried by these terms.

H2, a different computation: the 30% runs' groups are frequency-concentrated in weight space without computing a function of a + b − c on held-out pairs. Their equivariant share sits near that of matched random unit sets and of the step-0 network, the d-profile does not peak at 0, and the restricted preference comes from elsewhere (for example, operand-associated labels).

## Cheap measurement that separates them

For each run at its 50k checkpoint (and the published early checkpoints of the 50% runs), and for each group G_k of live units with dominant input frequency k:

1. Contribution C_k(a, b, c) = Σ_{j∈G_k} relu(U[a,j] + V[b,j]) W[j,c], minus its mean over c for each pair (softmax-invariant). Report the share of variance explained by a function of d = a + b − c, separately on the training pairs and on held-out pairs: the common never-trained set (483 pairs) and each run's own cal + audit split. Also report the share explained by cos and sin of kd alone, and the sign of the d-profile at d = 0.
2. Ablation: zero the group's output rows. Report common-set and own-split accuracy, the restricted rank among absent labels (reusing `research/mechanisms/relational.py` with the held-out set swapped), train accuracy, and the size of the logit change.
3. Controls: no-op (nothing ablated), full ablation (all units: DC-free output), the step-0 network, and 16 random unit sets per group with the same number of units and total |u||w| norm within ±10%.

Expected under H1: in the 30% runs the groups' held-out equivariant share is clearly above the matched random sets and step 0, below the 50% runs, with the d-profile peaking at 0. Expected under H2: near the random-set and step-0 levels, no peak at 0.

## Known mismatches of the shams and failure modes of the diagnostic

- Gate changes: removing units from the single hidden layer changes no other unit's gate, unlike the e1 input filters, which change gates. Group ablation and random-set ablation are matched on this. Neither is matched to the e1 filter interventions.
- Functional disruption: groups and random sets matched on count and norm can still change the logits by different amounts. I report the logit-change size and compare at that, not only at equal count.
- Selection: groups are read from the analysed checkpoint (checkpoint-local). Using the 50k groups at earlier checkpoints is retrospective and labelled as such.
- Random sets are not structureless. When most units are single-frequency, a random set is a mixture of groups and carries equivariant terms too. The control tests whether grouping by frequency is the right partition, not whether structure exists.
- The equivariant share on training pairs is contaminated by memorization, because training labels equal a + b. Only held-out shares bear on H1 versus H2.
- In the 30% runs the groups are weakly defined (concentration about 0.4), so a group's contribution mixes frequencies. The d-share does not require frequency k; the cos/sin(kd) share does.
- Amount and identity of training data are confounded (DECISIONS D6), so any 30%-vs-50% contrast is about these two splits, not about data amount alone.

## Cycle 2, written 2026-10-21 at about 19:50 UTC, before any edit below was computed

Already seen: packet 1 (README.md). In the 30% runs every frequency group adds a cosine of a + b − c at its own frequency that peaks at the correct label on the common set, with about one fifth of the 50% runs' amplitude, under a held-out contribution that is about 99% not a function of a + b − c.

Question: can the frequency structure in the failed 30% runs' weights carry correct answers on pairs no split trains on, once it is isolated by an edit that uses the weights alone? An edit built from a + b − c would be correct on held-out pairs by construction (a masked readout), so neither edit below looks at labels, pairs or d.

Edits (`unitfilter.py`), applied at a checkpoint to every live unit j, with k_j the unit's dominant input frequency (largest share of U[:, j] and V[:, j] energy in rfft bins 1..15, the same definition as `units.py`):
1. **Per-unit filter.** Replace U[:, j] and V[:, j] (token axis) and W[j, :] (label axis) by their projections onto frequency k_j. The DC component of U and V is kept, because it sets the unit's gate; W's DC does not change any argmax.
2. **Scale aligned units.** Multiply W[j, :] by s ∈ {2, 4, 8} for the units whose dominant input and output frequencies agree, and also report keeping only those units (s → ∞). In the 50% runs nearly every unit agrees (0.99–1.00), so this edit is close to a no-op there; it is meant for the 30% runs (0.66–0.75).

Score: plain argmax over all 31 labels on the common set (483 pairs) and on the run's own cal + audit split, against a + b. Next to every result: train accuracy against the run's own training labels and the RMS size of the pair-centred logit change.

Nulls, each with the same edit:
- step 0 of the same run (step 300 where step 0 is not published), where k_j is arbitrary;
- for the filter, each unit filtered to a random non-dominant frequency k′_j ≠ k_j, 8 draws (seed 36000);
- for the scaling, random live-unit sets with the same count and total ‖u_j‖‖w_j‖ within 10% of the aligned set, 8 draws;
- the e1 shuffled-label width-128 runs (`e1-st-shuf-w128-i3000{0,1,2}-a1`, same split as the 30% runs, training labels a fixed random table that agrees with a + b on 2.7% of training pairs), at step 0 and 50k. Their fit cannot have put a + b structure in the weights.

Expected if the 30% runs' weights carry a partial addition computation (H1): the filtered 30% models answer common-set pairs correctly well above chance (1/31 ≈ 0.03) and above all nulls, and the filtered 50% models keep most of their accuracy. Expected if the aligned terms are too weak or mixed to carry answers (or are not a computation): the filtered 30% models stay near chance like the nulls. The coordinator's criterion is the first case; anything else is reported as a null result.

Known mismatches and failure modes:
- Filtering changes every unit's gate, as the e1 input filters do and the packet 1 ablations do not. The random-frequency null is matched on that but keeps much less of each unit's energy in the 50% runs (dominant share 0.83–0.86 against about 0.01), less so in the 30% runs (0.40–0.44 against about 0.04). The kept-energy fraction is reported.
- If filtering destroys the 50% runs' accuracy, the edit is too disruptive to test anything, and a negative 30% result says nothing.
- The Fourier basis of Z_31 is a prior about cyclic structure, the same one every frequency analysis here uses. It carries no information about which phases add up at a + b; only the learned phases decide that.
- The shuffled-label runs may be less frequency-concentrated than the 30% runs. Their concentration is reported, since a null with weaker structure is a weaker match.
- Amount and identity of training data stay confounded (DECISIONS D6).

## Cycle 2 close-out, written 2026-10-21 at about 20:03 UTC, before running `unitfilter.py` on the no-decay runs

Not prospective for the headline. The coordinator's spot check (DECISIONS D10, `research/e2/coordinator/spot_unitfilter.jsonl`) already found filtered common-set accuracy 0.466 / 0.325 / 0.416 at 50k for the e1 no-decay width-128 runs (`e1-st-add0-w128-i3000{0,1,2}-a1`). These runs have the same split and a+b training labels as the 30% runs, weight decay 0, input concentration 0.22–0.23 (0.17 at step 0) and a null restricted rank. What is new here: the full null set, the variants, the over-training points and the scaling edit on these runs, from the lane's own code.

What each outcome would mean:
- If the no-decay runs' filtered accuracy stands clearly above their own nulls, as the 30% runs' does, then the filter's held-out accuracy is mostly available from any network that fits a+b on these pairs. The packet 2 result then does not show that the decayed 30% runs hold a computation of their own; it supports the fit-induced alternative ("the projection supplies the prior").
- What would still favour structure specific to weight decay: filtered accuracy in the 30% runs clearly above the no-decay runs at comparable kept energy or train accuracy, at every checkpoint. A gap of the size seen at 50k (0.43–0.58 against 0.33–0.47, one split, three seeds each) would not be enough on its own.

Correction to packet 1 (e2 audit cycle 2, G13): the step-0 reference "34/75 groups aligned" counted byte-identical networks more than once, because runs with the same init seed start from the same weights. The three distinct step-0 networks give 21/45.
