# e3 audit, fifth investigation: the 24 dynamics runs and the ladder claims (written 2026-10-22, about 01:00 UTC, before computing anything below)

Session `e3-au-cloud-1`. Base `0d11fbb` (dynamics packet 1 integrated as PR #28; D14). Budget used so far: about 2.6 of 30 min.

## Already known

- Investigation 2 checked the design and the splits (Y1): every ladder split, hash, count, nesting, closure and evaluation set matches. D is empty for 3 evaluation pairs of the 41001 ladder.
- I have read the lane's README, its three figures and the measure definitions in `ladder.py` (docstring and `score`). I have not run `ladder.py` or opened `ladders.csv` / `ladders_final.csv`.
- The lane reports (Table 1, items 1–6, Table 3) that the ordering travels, that ladder spread is no larger than init spread except for the σ-margin and P(a+b over S) at 40%, where data seed 41002 is ahead on both inits, that P(S over N) falls by 0.051–0.060 from 30% to 50% in all 8 sequences, and that the 45% runs still rise at 50,000.
- The published checkpoints are at the first-interpolation update (300 in 21 runs, 400 in 3), 1k, 2k, 5k, 10k, 20k, 30k and 50k, with Adam moments. No step-0 checkpoint is published.

## Software checks (Y2–Y6 of QUESTION-2.md, adapted to what was published)

| # | Check | PASS if |
|---|---|---|
| Y2′ | Init pairing (changed: no step-0 checkpoint is published) | For all 24 runs, an independent replay from `independent.initial_weights(31, 128, init_seed)` on the regenerated training split, Adam lr 1e-3, wd 1, reproduces the first published checkpoint (300 or 400) within 1e-9 × the largest weight. This checks the init, the split and the recipe together |
| Y3 | Provenance and accounting | As stated in QUESTION-2.md; plus config fields against the plan, manifest hashes of every published file |
| Y4 | Replay, on the 41002 40% pair (both inits; the lane's suggestion) | Early: from the independent init to update 2,000, every published checkpoint within 1e-9 × the largest weight. Late (changed: only 30k and 50k are published around the end): from the 30k checkpoint with its moments, Adam step counter continuing at 30,001, 20,000 updates to 50k. Reported as found: largest weight difference, and the 50k evaluation accuracy and σ-margin of the replay against the published checkpoint |
| Y5 | Checkpoint predictions | My forward pass on every published checkpoint of all 24 runs: train, cal and audit accuracy exactly equal to `metrics.jsonl`; cross-entropy within 1e-12 relative (as pre-stated). I record the raw relative and absolute differences. If only the relative criterion fails on tiny cross-entropies, I report it as a mis-set criterion, as with C3 |
| T1 | Table 1 recomputed | My own implementation of the measures, from the definitions in the README and the `ladder.py` docstring (not its code), on all 24 e3 runs and the 8 reference 31000 runs at 50,000. Every value matches `ladders_final.csv` within 1e-9: accuracy, P(a+b over N), P(a+b over S), P(S over N), the σ-margin and the own-absent rank. The P measures come from `pairwise.py` (my definition). The σ-margin and the rank are written fresh |
| Y6 | Separation of variation | Review of the three figures (done before computing). Ladders are shown in separate panels with their evaluation-pair counts, init seeds as line styles within a ladder, and pairs are never pooled or counted as replicates. PASS |

## Interpretation rules

1. **"The ordering travels"** is SUPPORTED LOCALLY if, in my values, each of the 8 ladder × init sequences improves strictly at every rung on accuracy, P(a+b over N), P(a+b over S), the σ-margin and the own-absent rank (falling). If any rung fails, I name it.
2. **"P(S over N) falls with share in every ladder"** is SUPPORTED LOCALLY if it falls by at least 0.03 from 30% to 50% in all 8 sequences.
3. **"The 45% runs are still rising at 50,000"** is SUPPORTED LOCALLY if evaluation accuracy rises by at least 0.03 from 30k to 50k in all six e3 45% runs.
4. **"41002 at 40% is a ladder (training-set) effect."** The lane already calls it a hint that two inits cannot separate from chance. I test it four ways.
   - (a) **Null calibration of the lane's criterion.** The lane's criterion is that "the range of the three e3 ladder means exceeds all four within-ladder init differences". Under a no-ladder-effect model, each cell is the share mean plus independent Gaussian noise, so the criterion's probability P0 does not depend on the noise scale. I compute P0 by Monte Carlo. I also count how many of the 24 measure × share combinations meet the criterion (six measures, four shares; the combinations are not independent).
   - (b) **Two-way decomposition for the 40% σ-margin.** Ladder and init are crossed, because the same two init seeds are used on every ladder. With one value per cell, I take the ladder mean square over the ladder × init mean square: F(2, 2) for the three e3 ladders and F(3, 3) with 31000. I report both p-values.
   - (c) **Fixed pairs.** Each ladder is scored on its own evaluation pairs, so a ladder difference mixes training-set identity with evaluation-pair identity. I score the eight 40% runs (four ladders × two inits) at 50,000 on the 53 pairs that are outside all four ladders' 50% training sets. The primary measure is the σ-margin over each run's own N (lane definition). The secondary measures are P(a+b over own N) and a ladder-free one: the mean normalized rank of a+b among all 30 wrong labels.
     - The rule uses the primary measure. On the lane's own pairs, 41002 is above every other ladder's run of the same init for both inits.
     - "The lead is not an evaluation-pair artifact": SUPPORTED LOCALLY if the lead holds for both inits on the 53 pairs, CONTRADICTED if it holds for neither, and AMBIGUOUS if it holds for one.
   - (d) **Over training.** Descriptive: does 41002 lead for both inits at 20k and 30k as well as at 50k?
   - **Overall verdict.** SUPPORTED LOCALLY only if (b) gives p < 0.05 in both versions and (c) is supported. CONTRADICTED if (c) is contradicted. AMBIGUOUS otherwise.

## Expectations, stated before computing

- I expect the software checks to pass, with the late replay drifting at the rounding sensitivity seen before.
- I expect rules 1–3 to hold as the lane reports.
- For rule 4 I expect AMBIGUOUS.
  - The lane's criterion should fire often by chance. The range of three means of two has about the same expected size as one init difference.
  - The init effect switches sign between ladders (on the lane's 40% σ-margins, init 30000 is ahead in 41001 and 41003 and behind in 41002), so the interaction is large.
  - On 53 pairs, the median σ-margin is coarse.

## Failure modes

- The 53 fixed pairs are a small, non-random subset (pairs outside all four 50% sets). Their N sets still differ by ladder, and pairs share operands and labels. The ladder-free rank is there to remove the N-set difference.
- The Gaussian model in (a) and (b) is a convention, not a fitted model, and with two inits per cell, F tests have little power. A "not significant" result is weak evidence of no ladder effect.
- Within a ladder the shares nest, so a ladder's four rungs are not independent: one lucky group can lift several rungs.
