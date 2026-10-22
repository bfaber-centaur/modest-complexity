# e3 independent audit

Session `e2-au-cloud-1` (e3 auditor), branch `claude/project-thread-id1tmm`. Software checks are PASS / FAIL / NOT CHECKED. Interpretations are SUPPORTED LOCALLY / AMBIGUOUS / CONTRADICTED. Everything is exploratory: the e2 ladder pairs have guided analysis since e1.

# Investigation 1: operand-label preference without logit scale (existing artifacts)

Base `4783a47` (close of e2). The question, decision rules and expectations were committed before computing: [QUESTION.md](QUESTION.md), `af9c01e`. No training runs.

## Figure 1

![pairwise preference](figures/preference.png)

Look before reading: in the top left, does any trained curve rise after its first few hundred updates? In the top right, do the filled markers step down from 30% to 50%, and do the orange ones?

## Definitions (for reuse by the e3 lanes; `pairwise.py`)

For a held-out pair with logits z and disjoint label sets A, B: **P(A over B)** = mean over a in A, b in B of 1[z_a > z_b] + 0.5 · 1[z_a = z_b]. It is undefined, and dropped and counted, when A or B is empty. The run summary averages per-pair values with equal weight per pair. S, D and N come from `ladder_sets(a, b, smallest_train, largest_train, eval_pairs, p)`:

- S: labels seen with an operand in the ladder's smallest training set.
- D: labels seen only in the largest.
- N: the other wrong labels.

The evaluation pairs are those excluded from the largest training set. The e2 sets are reproduced exactly (asserted). Three quantities:

- **P(S over N)** measures operand-label preference.
- **P(a+b over N)** is the fraction of N below the correct label, the rank analogue of the margin over N and the e2 restricted rank's complement.
- **P(a+b over S)** is the rank analogue of the margin over S.

P(S over N) reflects the bulk of the labels, not the strongest competitor, so the max-based diagnostics of e2 audit cycle 4 stay alongside (bottom right).

## Results (width 128, 483 common pairs, ranges over three init seeds)

| share | P(S over N) at first fit (300 or 400) | at 1k | at 10k | at 50k | at train CE 1e-4 | change 1k→50k by seed | P(a+b over N) at 50k | P(a+b over S) at 50k |
|---|---|---|---|---|---|---|---|---|
| 30% | 0.970–0.978 | 0.956–0.964 | 0.950–0.961 | 0.940–0.948 | 0.936–0.948 | −0.016, −0.017, −0.018 | 0.69–0.74 | 0.14–0.17 |
| 40% | 0.964–0.973 | 0.946–0.955 | 0.929–0.936 | 0.911–0.919 | 0.911–0.920 | −0.032, −0.034, −0.036 | 0.90–0.91 | 0.47–0.50 |
| 45% | 0.965–0.968 | 0.934–0.941 | 0.915–0.918 | 0.902–0.903 | 0.890–0.899 | −0.031, −0.039, −0.034 | 0.99 | 0.84–0.89 |
| 50% | 0.955–0.962 | 0.930–0.936 | 0.900–0.904 | 0.888–0.891 | 0.866–0.881 | −0.047, −0.045, −0.041 | 0.998 | 0.97–0.98 |

**Controls.**
- Step 0: 0.479–0.501.
- Shuffled-label 30% runs, scored with the a+b sets: 0.453–0.474 at 50k.
- No-decay 30% runs: 0.952–0.964 at 1k and 0.927–0.938 at 50k, with P(a+b over N) 0.50–0.52.
- Closed-form memorizer (e2 cycle 4): 0.71–0.73 at 30% and 0.735–0.744 at 50% (λ 1e-6); 0.74–0.76 and 0.76–0.77 (λ 1e-2).
- P(D over N) is 0.47–0.49 in the 30% runs, where D and N are both outside the run's own set. It rises with share as D becomes the run's own: 0.62, 0.74–0.75, 0.84–0.85.

**Robustness.** The pooled (|S|·|N|-weighted), transpose-group and median summaries give the same ordering and the same conclusions (at 50k: pooled 0.938–0.947 at 30% to 0.888–0.893 at 50%; median 0.953–0.958 to 0.900–0.904). Across all 185 ladder checkpoints from 300 updates on, P(S over N) correlates at 0.96 with cycle 4's σ-unit S − N gap.

## Software checks

| # | Check | Result | Evidence |
|---|---|---|---|
| E1 | Invariance to scale and offset | PASS | At all 269 rows, every per-pair value is identical after multiplying the logits by 0.37 or 1000, adding 5, or adding a random offset per pair |
| E2 | Ties, empty sets | PASS | No tied comparison in any set pair at any row; S, D and N are nonempty for all 483 pairs (no pair dropped) |
| E3 | Label sets | PASS | `ladder_sets` reproduces e2's S, D, N exactly; a+b is in none (asserted) |
| E4 | Regeneration | PASS | `preference.csv`, `preference-matched-ce.csv` and the PNG regenerate byte-identically |

## Interpretations

| Claim | Verdict | Evidence and scope |
|---|---|---|
| The operand-label preference grows during training (raw-logit reading, e2) | CONTRADICTED | Pre-stated rule 1: flat at 30% (all seeds within ±0.02) and falling at 40%, 45% and 50% (−0.03 to −0.05 in every seed). It is highest at first fit (0.955–0.978) |
| The preference is smaller at larger training share | SUPPORTED LOCALLY | Pre-stated rule 2: ordered 30 > 40 > 45 > 50 without overlap at 50k and at matched fit. This settles e2 cycle 4's AMBIGUOUS σ-unit result: the effect is real but small. An operand label beats an unrelated one 94% of the time at 30% and 89% at 50%. One data seed |
| The preference comes from fitting the training labels, not from weight decay | SUPPORTED LOCALLY | The no-decay runs, which never rank a+b above N (0.50–0.52), prefer S over N as strongly as the decayed 30% runs (0.93–0.94 at 50k) |
| The share effect is generic memorizer dilution (e2 INTERPRETATIONS 2, alternative 2) | CONTRADICTED for the closed-form memorizer | Its P(S over N) does not fall with share; it rises by 0.007–0.016. Untested for gradient-trained memorizers |
| Share mainly acts on the correct label, not on the preference (e2 cycle 4) | SUPPORTED LOCALLY, now in ranks | From 30% to 50% at 50k, P(a+b over S) goes from 0.14–0.17 to 0.97–0.98 and P(a+b over N) from 0.69–0.74 to 0.998, while P(S over N) moves by 0.055 |

**Expectation missed.** I expected the shuffled-label runs within 0.02 of 0.5. They sit 0.03–0.05 below it, a slight avoidance of the a+b-associated labels. Their preference for their own shuffled training labels was not measured.

**Strongest surviving alternative.** The small share effect, and the slow fall of the preference during training at 40–50%, may be a generic property of gradient-trained networks fitting more pairs per operand. The no-decay runs fall during training too (−0.020 to −0.027 from 1k to 50k), without ever ranking a+b high. If so, the effect is not tied to finding the answer. The closed-form memorizer cannot speak to this, because it has no training dynamics.

**Cheapest next discriminating check.** First, score the e3 dynamics ladders (three new data seeds) with these three quantities: does the fall of P(S over N) with share travel across data seeds? That costs nothing extra. If it travels, shuffled-label runs at two nested shares, scored with their own label sets, would separate trained-network dilution from answer-finding.

## Reproduce

From the repository root:

```bash
E=research/e3/audit
uv run --locked python $E/preference_check.py $E/results/preference.csv
uv run --locked --group analysis python $E/plot_preference.py $E/figures/preference $E/results/preference.csv \
  research/e2/audit/results/margins-check.csv $E/results/preference-matched-ce.csv
```

## Budget

About 25 s of numerical execution, one process and one BLAS thread. No training runs. The e3 numerical budget has not been set yet, so this count stands on its own.
