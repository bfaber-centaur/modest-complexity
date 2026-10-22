# e3 audit, first investigation: question (written 2026-10-22, about 00:10 UTC, before any of the computations below)

Session `e2-au-cloud-1`, now the e3 auditor. Base `4783a47` (swarm/exploration after the close of e2; no e3 pin exists yet). This investigation uses published artifacts only. It launches no training runs.

## Already known

- e2 audit cycle 4: on the 483 common pairs of the nested 30/40/45/50% ladder (data seed 31000, width 128), the S − N gap grows in logits during training. At 30% it doubles. In units of each pair's wrong-label standard deviation σ it stays near 2σ and is flat or falls. From 30% to 50% it falls 14% (σ) or 30% (within-set spread, post hoc) at training cross-entropy 1e-4, and 40% in logits. The two scales disagree on the size of that residual share effect, and both contain choices: σ includes the S − N offset itself.
- Set sizes: S 4–25 (mean 16.0), D 1–13, N 1–20; none is empty on the common pairs.
- Nothing pairwise has been computed. I have not looked at any per-label rank of S against N.

## Measure, fixed now

For a held-out pair with logits z and label sets A, B (disjoint, both excluding a+b):

P(A over B) = mean over a in A, b in B of 1[z_a > z_b] + 0.5 · 1[z_a = z_b].

It is undefined if A or B is empty; such pairs are dropped and counted. The run-level summary is the mean over the 483 common pairs of the per-pair value. Each pair weighs the same, whatever |A| · |B| is. The pooled alternative, which weighs pairs by |A| · |B|, is reported as a robustness column. Transposes (a, b) and (b, a) share their label sets and are not independent. The replicate unit is the init seed; pairs are never treated as independent replicates. S, D and N are e2's fixed sets, from the frozen 30% and 50% training sets. The primary quantity is P(S over N). The rank analogues of the max-based margins are P(a+b over N), the fraction of N labels below a+b, and P(a+b over S). They are reported alongside the cycle-4 max-based diagnostics (margin over N and gap in σ units, "beats every S"), not instead of them.

Invariance, checked numerically: the per-pair values must be identical after z → c·z for c = 0.37 and 1000, z → z + 5, and z → z + a random offset per pair. Ties are counted before and after.

## Questions and decision rules

1. **Time.** Does P(S over N) change during training once scale is removed? Per share and seed, Δ = value at 50k minus value at 1k. "Rises" if Δ > +0.02 in all three seeds, "falls" if Δ < −0.02 in all three, "flat" if |Δ| ≤ 0.02 in all three, otherwise "mixed". Values at 300 updates are reported too.
2. **Share.** Does P(S over N) fall with training share? At 50k and at training cross-entropy 1e-4 (interpolated in log cross-entropy between checkpoints, as in cycle 4): "ordered" if 30 > 40 > 45 > 50 without overlap between neighbouring shares' seed ranges, at both; "no share effect" if the 30% and 50% ranges overlap, at both; otherwise "ambiguous".
3. **Controls.** Step 0 (three inits), the shuffled-label 30% runs scored with the a+b sets, the no-decay 30% runs, and the closed-form memorizer of cycle 4 at the four shares. Expectation: step 0 and shuffled within 0.02 of 0.5.

## Expectations, stated before computing

Cycle 4 predicts that P(S over N) is well above 0.5 and roughly flat in time at every share, with at most a modest fall from 30% to 50%. The cycle-4 conclusion, "no growth once scale is removed", would be contradicted for this distributional measure if P(S over N) rises over time in all seeds. Because P(S over N) is set by the bulk of the labels and the max-based gap by the strongest ones, the two can disagree. A disagreement would mean that the strongest operand labels and the typical ones behave differently.

## Failure modes

- Ties: logits are float64 sums, so exact ties should be rare. They are counted, not assumed absent.
- Scaling by a constant that is not a power of two could, through rounding, create ties between nearly equal logits. Any change is reported.
- A rank measure discards how far apart labels are. The strongest competitor decides accuracy, so the max-based diagnostics stay in the report.
- One data seed and three init seeds per share. The e3 dynamics ladders will test whether this travels across data seeds.
