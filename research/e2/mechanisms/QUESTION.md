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
