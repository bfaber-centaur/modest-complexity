# c2 time lane: note (M4)

**Question** (charter 02-time): are the original RNN gaps persistent within a longer budget, or mainly differences in learning speed?

- **Design, predictions and rules.** [QUESTION.md](QUESTION.md). Revisions R1 and R2 were written before any continuation record was read. Follow-up F1 was approved before it ran.
- **Commands, budget and manifest.** [README.md](README.md).
- **Sources.** Every number below comes from `runs/`, `checks.json` or the CSVs in `figures/`. The reviewer's independent checks are in [../review/REVIEW.md](../review/REVIEW.md) §2.

## Design in brief

**Runs.** Nine recovered r1 runs were continued from `ckpt-10000` to 50,000 updates with full Adam state. All use one task draw and split (`ops51000-ls51100`) and three init seeds (51200–51202). The recipe is r1's, unchanged:
- full-batch Adam with lr 1e-3, b1 0.9, b2 0.999 and eps 1e-8;
- decoupled weight decay 0.1 on every parameter;
- width 128 and 384 training rows.

Follow-up F1 then continued the three `tt-final` runs from 50,000 to 100,000 updates.

| arm | transition / readout | step weights | params | 10k source |
|---|---|---|---|---|
| `tt-inter` | tied / tied | 1 1 1 | 19,208 | `r1-comp-rnn-inter` |
| `tu-inter` | tied / per-step | 1 1 1 | 21,272 | `r1-rev-comp-rnn-untied` (the reviewer's r1 arm) |
| `tt-final` | tied / tied | 0 0 1 | 19,208 | `r1-comp-rnn-final` |

Each run is a continuation of its original run, not a new seed and not a replay. Three init seeds are exploratory coverage, not statistical power.

**Checks** (`checks.json`; the reviewer recomputed each independently).
- **On-host continuation.** Resuming each arm's seed-51200 `ckpt-1000` to 3,000 updates reproduced every published r1 key at all 41 logged steps.
- **Joins.**
  - Every continuation's first row equals the published row it continues: r1's step-10,000 row, or the 50k run's step-50,000 row.
  - N and ¬N, recomputed from the predictions, equal the logged `held_subset`.
- **Recipe.** Every run's recorded recipe equals its r1 source `config.json` (reviewer C1).
- **Code.** All 15 runs record identical hashes for the training code and the environment.

**Logging.** Every 50 updates. Time to threshold is the first of five consecutive logged evaluations with held-out final accuracy ≥ 0.95 (≥ 122 of 128 rows). It is known to 50 updates, and a run with no such window is censored at its last update.

## Observations

### Main comparison, 10k → 50k

Primary figure: [figures/time_main.png](figures/time_main.png).

| arm | held-out final at 10k → 50k | threshold step | ¬N (42 rows) at 10k → 50k | ¬N threshold step |
|---|---|---|---|---|
| `tt-inter` | 1.000 / 0.961 / 1.000 → 1.000 ×3 | 7,700 / 9,150 / 8,200 (in r1's record) | 1.00 / 0.88 / 1.00 → 1.00 ×3 | 8,650 / 11,200 / 9,000 |
| `tu-inter` | 0.633 / 0.523 / 0.516 → 1.000 ×3 | 24,900 / 32,300 / 33,550 | 0.24 / 0.07 / 0.05 → 1.00 ×3 | 30,250 / 34,100 / 39,350 |
| `tt-final` | 0.273 / 0.312 / 0.320 → 0.562 / 0.695 / 0.656 | censored ×3 | 0.00 ×3 → 0.12 / 0.26 / 0.14 | censored ×3 |

- **Fit and staying power.** Train final accuracy is 1.0 throughout the continuation in every run. Every crossing stays ≥ 0.95 through 50,000.
- **Pre-registered outcomes.**
  - `tu-inter` meets **"speed within 50k"**: all three seeds cross the threshold, and ¬N is 1.0 at each of the last five evaluations.
  - `tt-final` is **unresolved** at 50k. Held-out final rose +0.05 / +0.12 / +0.10 over 40k–50k, and 14–20 held-out predictions changed between 40k and 50k.
- **Reviewer's check of `tt-final` (CX-T1).** On rows still wrong at 50k, the median wrong-logit gap shrank 15–28% from 40k to 50k, and held-out CE fell from 3.6–4.0 to 2.7–3.3. So "unresolved" is the right reading, not "persists".
- **N before ¬N.** At 30k, `tu-inter` is already at 0.98–1.00 on N but at 0.67–0.93 on ¬N, which crosses at 30,250–39,350. `tt-final` at 50k is at 0.78–0.91 on N but only 0.12–0.26 on ¬N.

### Follow-up F1: `tt-final`, 50k → 100k

Figure: [figures/time_followup.png](figures/time_followup.png).

| seed | held-out final at 50k / 75k / 100k | threshold step | ¬N at 50k / 75k / 100k | N at 100k | 90k → 100k |
|---|---|---|---|---|---|
| 51200 | 0.562 / 0.781 / 0.898 | censored | 0.12 / 0.48 / 0.74 | 0.98 | +0.06, CE −35% |
| 51201 | 0.695 / 0.797 / 0.961 | 95,050 | 0.26 / 0.40 / 0.88 | 1.00 | +0.05, CE −52% |
| 51202 | 0.656 / 0.844 / 0.914 | censored | 0.14 / 0.55 / 0.74 | 1.00 | +0.02, CE −28% |

- **Outcome.** By the same rules applied over 90k–100k, `tt-final` is still unresolved at 100,000.
  - Every seed is still rising.
  - Held-out CE fell 28–52% over the last 10,000 updates.
  - The held-out predictions at 100k differ from those at 90k in 3–8 rows.
  - One seed crosses the threshold and two do not.

### Wrong predictions remain neighbour-compatible

From `time_error_compatibility.csv` and `time_followup_error_compatibility.csv`, at every checkpoint from 10k to 100k and in every arm with wrong rows:
- **Any neighbour.** Every wrong held-out prediction (100%) equals the final label of some one-op-slot training neighbour. Uniform wrong answers would give 64–77%.
- **Same prefix.** 86–100% equal the label of a same-prefix neighbour, the one that differs only in op 3, against 29–37% if uniform. At the 1,000-update resolution of the CSVs, the lowest value is 75%, when only a few rows are wrong.

So the remaining wrong rows keep giving the answer of a seen neighbour while their number falls. For example, in `tt-final` s51201 it falls from 88 at 10k to 39 at 50k and 5 at 100k. This compatibility does not show that the networks copy neighbours.

### Optimization diagnostics: the late phase runs in Adam's eps regime

These are logged from 10,000 on. r1 logged no diagnostics before that.
- **Timing.** In every run of every arm:
  - median √v̂ falls below eps = 1e-8 at 24,100–25,100 updates;
  - every coordinate is below eps from 27,800–29,650 on;
  - √v̂ stays there through 100k (median about 5–6·10⁻¹⁰ in `tt-final` at 100k).
- **Cause.** Once the training set is fit, the train final CE falls to about 10⁻⁸ and the global gradient norm to about 10⁻⁷ (3.5–19·10⁻⁸ at 50k).
- **What the update becomes.** Adam's update lr·m̂/(√v̂ + eps) then becomes about (lr/eps)·m̂. That is momentum gradient descent with an effective rate of 10⁵, balanced against decoupled decay.
  - The reviewer measured this at 50k (T1, CX-T1): the Adam part of the update is 0.88–0.89 of (lr/eps)·m̂, and only 1.02–1.06× the decay part.
- **What else changes at the same point.** The train final margin stops growing: its median settles at about 19.8 (`tt-inter`), 19.0 (`tu-inter`) and 18.2–18.5 (`tt-final`). The gradient norm flattens. r1's routes lane saw the same regime at 200k without decay ([../../regimes/routes/NOTE.md](../../regimes/routes/NOTE.md)).
- **How much of the gain falls in this regime** (held-out final, before and after the median crosses eps):
  - `tu-inter` made 61–87% of its 10k→50k gain before. But two of its three threshold crossings (32,300 and 33,550), and all three ¬N crossings, came after every coordinate was below eps.
  - `tt-final` made 56–70% of its 10k→50k gain after, and 75–86% of its 10k→100k gain. F1 runs entirely in this regime.

### Function against parameters

- **Stable functions, moving parameters.** `tt-inter` and `tu-inter` show no change in any held-out prediction over 49k–50k. Their held-out CE still falls, by 19–56% from 40k to 50k in `tt-inter`.
- **Parameters keep moving** under the balance of gradient and decay.
  - Between `ckpt-40000` and `ckpt-50000` the distance is 2.6–3.4 in `tt-inter`, 5.4–6.1 in `tu-inter` and 4.4–4.6 in `tt-final`. The parameter norms at 40k–50k are 25.4–35.8.
  - Parameter norms shrink slowly after a peak near 20k, for example 27.6 → 25.5 in `tt-inter` s51200.
  - In F1, `tt-final` moves 7.3–7.8 per 25,000 updates at 75k–100k.
- **Directed in the inter arms, not in `tt-final`.** Over 40k–50k, the distance between checkpoints is 71–84% of the path length estimated from logged update norms in `tt-inter` and `tu-inter`. In `tt-final` it is 38–42% over 40k–50k, and 34–38% over 75k–100k, which is consistent with its noisier gradient. The estimate is coarse, because it takes the logged norm, sampled every 50 updates, as typical.

### Per-step readouts converge toward each other (reviewer CX-T2)

At 50k in `tu-inter`, the step-3 readout V3 reads h1 at 0.81 / 0.97 / 0.84, so CX-T2 held narrowly. But the per-step readouts have moved a long way toward each other:
- ‖V_k − V_3‖/‖V_3‖ fell from about 1.1 to 0.65–1.05;
- V3 reads h2 at 0.95–1.00;
- V2 reads h1 at 0.94–1.00.

The catch-up therefore partly re-ties the readouts. In s51201 it does so fully in function, with every cross-readout ≥ 0.92, but not in the weights: ‖V1 − V3‖/‖V3‖ is still 0.92 (reviewer T4).

## Predictions against outcomes

| prediction (QUESTION.md, F1) | outcome |
|---|---|
| `tt-inter`: s51200 and s51202 cross before 10k, s51201 by 20k; all stay; ¬N ≥ 0.95 | held: 7,700 / 9,150 / 8,200; ¬N crosses by 11,200 |
| `tu-inter`: all > 0.80 by 50k, ≥ 2 of 3 cross, crossings after 25k | all three cross; s51200 at 24,900, just before 25k |
| `tt-final`: all censored at 50k, 0.30–0.60, ¬N < 0.25, still rising | censored and rising; 0.56 / 0.70 / 0.66, so two seeds exceed 0.60; ¬N 0.26 in s51201 |
| errors stay neighbour-compatible far above the uniform rate | held |
| √v̂ below eps ≤ 0.01 of coordinates | **failed completely**: every coordinate below eps from about 28k |
| F1: 0.70–0.95 at 100k; at most one seed crosses | 0.898 / 0.961 / 0.914, so one seed is above the range; exactly one crosses |
| F1: ¬N 0.2–0.6 at 100k; same-prefix compatibility ≥ 85% | ¬N failed (0.74–0.88); compatibility held (96–100% at 75k and 100k) |

## Reading (interpretation, bounded)

All of this holds for one task draw and split, one recipe (including eps = 1e-8 and wd 0.1), three init seeds, and at most 50,000 updates (100,000 for `tt-final`).

**The readout-sharing gap at 10k was mainly a difference in learning speed under this recipe.**
- `tu-inter` reaches the same held-out and ¬N accuracy as `tt-inter`, crossing the threshold about 3–4× later.
- This is the outcome the charter names as evidence against a strong sharing-necessity story, for this arm and budget.
- It comes with two qualifiers.
  - **The late phase is a different update rule.** The late part of the catch-up, including most crossings, happens in the eps regime. There Adam behaves as decay-balanced gradient descent, not as the early normalized Adam. "Speed" here includes those late dynamics.
  - **Untied readouts may only partly stay untied.** The readouts converge toward one another as the arm catches up. So "slower" partly means "takes longer to approach a shared readout", and the result does not show that untied readouts compose by a different route.

**For held-out length-3 accuracy, the supervision gap is a large speed difference within 100k, and its end state is unresolved.**
- `tt-final` takes more than 10× longer than `tt-inter` in the one seed that crosses (95,050 against 7,700–9,150), and is still rising in all three.
- **This holds for length-3 accuracy only** (reviewer T2, CX-T4).
  - At 100k, the shared readout of `tt-final` reads the intermediate states h1 and h2 at 0.00–0.19.
  - On all programs of length 4–6, `tt-final` scores 0.06–0.14, against chance 0.125. Its tied transition is applied with the tied readout.
  - I recomputed both from the committed `ckpt-100000` (`time_analyze.py lengths` → `figures/time_followup_lengths.csv`), and they agree with the reviewer.
  - So at 100k final-only training still does not extend past length 3. Its length-3 gain does not come with intermediate states that the shared readout can read.
- The rise is not a neighbour-copy plateau. ¬N, which no one-slot neighbour copy can answer, climbs from 0 at 10k to 0.74–0.88 at 100k.
- Most of this rise happens in the eps regime. Whether it would continue, stall or speed up under a different eps, or under an optimizer that does not slow down at tiny gradients, is untested.
- A plateau would not have shown necessity either. Here there is no plateau.

**What this lane does not show.**
- That any arm has reached an asymptote.
- That `tt-final` will close the gap, at length 3 or at any other length.
- That the late dynamics would be the same with another eps, decay, learning rate or optimizer.
- That the networks compute by composing internal states. Accuracy on ¬N rows rules out one copying rule, not every shortcut.
- Any mechanism. The mechanisms lane has the 20k–50k states.

## Limitations

- **Coverage.** One task draw and split, and three init seeds. Seed-to-seed spread in crossing times is wide for `tu-inter` (24.9–33.6k).
- **The `tu-inter` sources.** They are the reviewer's r1 arm; the reviewer said so where it reviewed these claims.
- **Parameters.** The arms are not parameter-matched: 19,208 or 21,272.
- **Diagnostics coverage.** They start at 10,000. The early Adam regime is only seen in r1's accuracy and loss records.
- **The eps regime is a property of this recipe.** It is not separated from the effects of training time. F1 extended time, not the optimizer.
- **Logging resolution.** 50 updates. The path-length estimate in `time_checkpoints.csv` is approximate.
- **Host speed.** It varied: 3.4–3.9 ms per update in the main nine and 5.8–6.1 ms in F1, which used the same code. Timing does not enter any result.
- **Recorded git status.** Each run's `config.json` records the git status at launch. Several runs show lane-file edits made while earlier jobs ran, such as QUESTION.md revisions and `time_analyze.py`. The training code and environment hashes are identical in all 15 runs.
- **Literature.** No novelty claim is made, and no targeted literature check was done in this lane. The charter's starting reference (arXiv 1802.06467, lookup-table composition) was not re-read for this note.

## Reviewer handoff

1. **Joins and recipes for the F1 extensions.** `checks.json` → `c2-time-tt-final-ext-*`. Each extension's step-50,000 row must equal its 50k run's last row, and its recipe must equal the r1 source.
2. **The F1 classification at 100k** (`figures/time_followup_threshold.csv`): s51201 crosses at 95,050; s51200 and s51202 are censored. Check the columns `preds_differ_last_10k`, `pred_changes_last_1k` and `held_ce_rel_change_last_10k`.
3. **F1 checkpoints.** `ckpt-75000` and `ckpt-100000` of the three extensions should rescore to their logged rows, as was done for 20k–50k.
4. **Eps-regime numbers.** The numbers quoted from REVIEW.md §2 T1 are the 0.88–0.89 and 1.02–1.06× ratios at 50k. The timing and gain fractions in this note were recomputed from `time_main.csv` and `time_followup.csv`. Check the two agree.
5. **Error compatibility** at 75k and 100k (`time_followup_error_compatibility.csv`).
6. **Wording.** Check that the claim wording above stays within "this task, recipe and budget", per Bobby's standing rule on finite-budget results.
