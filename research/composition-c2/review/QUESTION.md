# c2 independent review: plan, predictions and counterexamples

Role: [../coordinator/charter/06-reviewer.txt](../coordinator/charter/06-reviewer.txt) (Bobby's, verbatim). Decisions: [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md).
- **Conventions.** Run IDs `c2-rev-`, seeds 65000–65999, budget 25 min (D6). Files are `rev_*.py` (D3 import hygiene).
- **What I own.** I write only in `research/composition-c2/review/`. I never edit lane or common files.
- **Status values.** PASS, CONCERN, BLOCK or NOT CHECKED. Each names exact artifacts and what it affects.
- **Disclosure.** I wrote r1's per-step-readout arm (`r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`). It is now an input to the time and sharing lanes. I say so wherever a claim I review rests on it, and I checked those files the same way as everyone else's.

## 1. Pin check (`b3e8132`), before any new training

Evidence: [rev_pin.py](rev_pin.py), with output in [out/rev_pin.txt](out/rev_pin.txt). Each item is checked with code written here, not by re-running the pin's own assertions.

| item | how | prediction |
|---|---|---|
| test_c2.py | Run in its own process. | 10/10 pass. |
| Task rows, any length | A plain-Python oracle from the r1 tables (copied as data), lengths 1–6, every row, against `c2task.rows`. Train/held programs and the Latin square, and N/¬N, recomputed independently. | Identical. |
| Init | `c2rnn.init` against the recovered r1 `ckpt-0` arrays: tied/tied against `r1-comp-rnn-inter-s51200`, tied/per-step against my arm. The four schemes compute one function. A copy written in place leaves the other copies unchanged. | Bitwise. |
| Loss and step weights | My own forward and loss, for all four schemes, with weights including 0 and non-unit values. Its value is compared with `c2rnn.loss_grad`, and every gradient entry with central differences of *my* loss. A weight-0 step's untied readout copy gets an exact zero gradient. | Agrees to about 1e-9. |
| Any-length forward and continuation | My forward with an explicit step map at length 5, and a continuation from a mid-program state. | Bitwise. |
| No-change evaluation | All 9 recovered `ckpt-10000`, evaluated with `c2train.evaluate`, against the last published r1 row (r1 keys). | Exactly equal. |
| Recovery record | Every file in `RECOVERY.json`: its committed bytes against the recorded SHA-256 and against the r1 manifest. My three runs are also checked against the originals still on my container. The steps and the scheme stored in each file are checked too. | 27/27. |
| Resume guards | Wrong hash and wrong scheme are refused. *Risk I expect to find:* a resume does not compare `--step-weights`, `--lr`, `--wd`, `--task-id` or `--init-seed` with the source run, so a tied/final run resumed without `--step-weights 0 0 1` would silently become an intermediate-supervised run. | Hash and scheme refused; the recipe not checked. |
| Cross-host resume | One reviewer run, `c2-rev-verify-final-s51200-a1`: `c2train.py --resume` from the recovered r1 `rnn-final-s51200` `ckpt-1000` to 10,000 with `--step-weights 0 0 1`, on this host. Its metrics are compared with r1's published rows, and its `ckpt-10000` arrays with the recovered original. It doubles as my host calibration. | Identical on this host too. This is the third host for that run. |

## 2. Lane reviews

Each lane's QUESTION.md is read when its M0 lands. My counterexample for each is recorded under Revisions below before I open its result tables.

### 2.1 Time (M0 `b7d1ab6`)

The `tu-inter` source runs are my r1 arm (disclosure above). Checks on the lane's published files and committed checkpoints, with code written here:
1. **Recipe against source (C1).** For every run in `registry.jsonl`, compare `config.json` (step weights, readout, task, lr, wd, `resumed_from` hash and `meta.condition`) with its source's recipe.
2. **Joining.** The step-10,000 row of all nine continuations equals r1's published step-10,000 row.
3. **Rescoring.** Every committed checkpoint (20k–50k) is rescored with my own forward. The result must equal the logged row at that step.
4. **Thresholds.** Threshold times, censoring and the post-crossing minimum, recomputed from `metrics.jsonl` with my own code. The same for ¬N. For 0–10,000, ¬N is recomputed from r1's `held_final_pred`.
5. **Attempts.** The registry and manifest: every attempt is kept, including capped or failed ones, and every published file matches its hash.
6. **Figure.** The figure, against the CSV and against items 2–4.

### 2.2 Generalization (M0 `e5a4415`)

Checks, with code written here:
1. **Task facts.** Recomputed with my own permutation code: the group the four ops generate, and the collision and function-class counts at lengths 4–6.
2. **Part A primary metric.**
   - Recomputed for all six r1 checkpoints at L = 1–6, every row, with my own forward and oracle. Compared with the lane's CSV.
   - At L = 3, the held-out prediction strings against r1's.
   - The lane's explicit step map (D10) is valid.
3. **Part B tasks and inits.**
   - Draws and splits are recomputed from the seeds with my own acceptance code and compared with `tasks.json`, accepted and rejected seeds alike.
   - `c2task.load` of each task ID equals the logged draw.
   - Held-out programs have no product collision at length ≤ 3. Each split is Latin, and squares are distinct within a draw.
   - `ckpt-0` is identical for inter and final within a split, and across splits with the same init seed.
   - Every run's recipe matches its plan entry (D9).
   - All 16 `ckpt-10000` are rescored with my forward. Held-out and ¬N use my recount on each task.

### 2.3 Mechanisms (M0 `65f9245`)

The per-step-readout condition is my r1 arm (disclosure above). Checks, with code written here:
1. **M1 at ckpt-0 and ckpt-1000.** My forward reproduces the published r1 rows at those steps. The 10k rows were already done in §1.
2. **Natural interchange by actual patching.** The donor's h_i replaces the recipient's h_j. The recipient's ops j+1..3 are applied with the shared transition and read with V, or V3 for the per-step readout. The target is the oracle suffix of the donor's symbol. All 16 cells, all 9 runs at 10k, compared with the lane's S(i, j). Also checked: the self-swap identity, the diagonal constancy, and agreement with my `rev_gen.py` length accuracies.
3. **Centroid patches.** S_cent is recomputed with centroids fit here on the 384 training rows only.
4. **No held-out rows in any fit.** The centroid and subspace fits (including follow-up B) use no held-out row. All pair sets and subsets are the predeclared ones.

### 2.4 Sharing (M0 `f84dbc3`)

The TU cell is my r1 arm (disclosure above). Checks, with code written here:
1. **Recipe.** Each run's recipe and scheme are read from `config.json` (D9).
2. **Init identity.** The step-0 rows are identical across the four cells per seed and equal r1's.
3. **Parameter counts.** They are read from the checkpoints. All parameters are decayed.
4. **Gradient aggregation.** At init on seeds 51200–51202, with full training data, the untied copies' gradients sum to the tied gradient.
5. **Replays.** Their metrics equal r1's published rows.
6. **Endpoints.** All 12 `ckpt-10000` are rescored with my forward: held-out, N/¬N with my recount, and train.

## Revisions

(Dated, before the matching analysis or run.)

- **Mechanisms NOTE revision `0fe8624`** (recorded in the commit that adds this entry, before the code below exists). I have read its diff against `817a768`. Its new numbers are checked against my CX-M1 output (`out/rev_mech.txt`) and against its own O6 table. There is one new geometric claim in O6: "U is centered, so a pure scale residual (r ∝ c(x)) lies largely outside it".
  - *Check.* At depth 2, in each of the 9 runs, compute the share of ‖c(x)‖² outside U for each training centroid c(x). U is fit here, as in follow-up B.
  - *Prediction.* The share is above 0.5 for every symbol in every run. "Largely" then holds. If it is below 0.5 anywhere, the sentence should be bounded.

- **Generalization follow-up anatomy and drift** (recorded in the commit that adds this entry, before the code below exists). This was written after reading the merged NOTE (`6fe1435`), its follow-up section, `gen_followup.py` and the head of `summary.json`. So the checks below are recomputations of numbers I have seen. Only CX-G3 is a new measurement.
  - **Checks (11 tied/inter `ckpt-10000`; my oracle, my forward, my draws and splits from `partb`).**
    - On exhaustive length-6 rows: first wrong step and the transition (true state before, op) at steps 4–6. Also the at-risk counts. Together these give all of `anatomy.csv`.
    - The `summary.json` measures: top-1/4/8 shares, Spearman ρ (step 5 vs 6; step 4 vs 5 where step 4 has ≥ 20 errors), the other-op share and the held-out-window shares at the error and at risk. Also the F1–F3 verdicts.
    - `windows.csv`, `followup.csv`, and the NOTE's ranges in the follow-up section and the held-out-window bullet.
    - The drift columns (`norm_ratio`, `centroid_distance_ratio`) of `steps.csv` for all 22 frozen models. Also the NOTE's step-6 ranges and its after-first-error percentages.
    - The at-risk window share should be exactly 0.25. Because the split is a Latin square, every op pair at positions (t−2, t−1) completes exactly one held-out program, whatever the filtering on the prefix.
  - **CX-G3, against "this matches gradual degradation of the iterated continuous state more than a few broken transitions".** If the continuous state degrades, then within a fixed transition the rows whose incoming state has drifted further should fail more often.
    - *Test.* At steps t = 5 and 6, take the rows at risk (correct through t − 1). Measure drift as ‖h_{t−1} − c(x_{t−1})‖, where c is the training-h3 centroid of the true symbol. Split each of the 32 transitions' at-risk rows into drift quartiles within that transition, then pool the error counts by quartile.
    - *Prediction.* The top-quartile error rate is at least 2× the bottom-quartile rate, at both steps, in at least 9 of 11 models. Low confidence.
    - *If it fails.* Drift is the NOTE's descriptive measure, and it does not locate the failures within a transition. The "degradation of the continuous state" reading would then rest on the per-step error rate alone.

- **After CX-S2 ran, sharing NOTE follow-up numbers** (recorded in the commit that adds this entry, before the code below exists). CX-S2 and the three checks above ran first: recipes, pairing and endpoints PASS; CX-S2's first half FAILED, its second half held (REVIEW.md §5). I have now read the NOTE's follow-up section (`501071c`). These are recomputations of its stated numbers, with no new prediction beyond "they match".
  - **From the logged rows.** The held-out, ¬N and wd 0 − wd 0.1 table; the paired contrasts and interaction; F2–F4; held-out at 1,000 and the change over the last 2,000 updates at both recipes; N at 10k; the first step where all three training heads are at 1.0. Also the held-out change from 1,000 to 10,000, against "then barely moves".
  - **Figure CSVs.** `followup_endpoints.csv` and `followup_trajectories.csv` against the logged rows.
  - **Optimizer.** The 10k row carries no `diag`, so I read the 9,950 row: median √v̂, the fraction below eps from 500 on, the update norm and the minimum train margin. The NOTE's "at 10k" values come from `groups_wd0.csv`, built from local-only checkpoints. I check that file against the NOTE, and its arrays are NOT CHECKED.

- **04:38, sharing wd-0 follow-up** (69ed8e2). This is post hoc in one respect. It was written after the coordinator's one-line summary, which said held-out at 10k was 0.30–0.45, ¬N ≤ 0.07, and "nearly flat after 1–4k, not eps-limited". It was written before opening any wd-0 file.
  - **Checks.** The recipes (wd 0, everything else as at wd 0.1). The step-0 rows equal the wd-0.1 runs' per seed. The endpoints recomputed from the logged prediction strings with my ¬N recount. No checkpoints are committed.
  - **CX-S2, against "without decay, no cell composes within 10k".** "Nearly flat" could be an argmax property while the function still moves.
  - *Test.* Per run, from 5k to 10k: the held-out change, the held-out final CE, the train final margin (from `--diag`), and the number of held-out predictions that change between consecutive evaluations.
  - *Prediction.* Held-out changes by at most 0.03 in every run. But the train margin keeps growing, by more than 10%, and held-out CE rises: the logits keep scaling up on fixed answers. That would make "flat" a statement about answers, not about the function.

- **04:28, sharing counterexample** (f966c73). This was written after reading its M0 `f84dbc3` and its NOTE's Interpretation section (`d8735fc`). I had also seen the coordinator's relay of the 10k held-out and ¬N table. I had not seen any trajectory, CSV or figure.
  - **CX-S1, against "tying the transition does worse under per-step readouts on every seed" and "the two tyings are non-additive" (NOTE Interpretation 2).** Both claims are read at exactly 10,000 updates, with cells still rising.
  - *Test.* Is the sign stable over the last 2,000 updates? I check every logged evaluation from 8,000 to 10,000 (41 per run), per seed: UU − TU and the interaction (TT − TU) − (UT − UU).
  - *Prediction.* UU − TU > 0 at every one of those evaluations in every seed, and the interaction > 0 at every one. If either flips sign at any evaluation in any seed, the "every seed" wording is fragile to the endpoint.

- **04:21, mechanisms counterexample** (2cfd1e1). This was written after reading its M0 `65f9245`, before opening its tables, CSV or figure. I had seen only the coordinator's one-line relay: interchange equals recombined evaluation, and centroid states do better exactly where natural states fail.
  - **CX-M1, against "history-specific residue in the natural state causes the failures that centroid patches avoid".** The alternative is that a centroid patch simply injects the symbol, where the natural state no longer carries it.
  - **The cells.** Every cell with S_cent − S ≥ 0.05.
  - **(a) Symbol-present donors.** Restrict natural donors to those whose h_i is nearest, among the training centroids c_i(·), to its own true symbol. Then recompute S.
  - **(b) Norm-matched donors.** Rescale each natural donor to its centroid's norm, then recompute S.
  - *Prediction for tied/inter.* In every such cell, (a) closes less than half of the gap, because nearly all donors are symbol-present. (b) closes less than a quarter. Both together mean the residue reading survives.
  - *Other conditions.* For tied/final and the per-step readout, (a) and (b) are reported descriptively, with no prediction.

- **04:14, D10 and the post-pin recovery addition** (cb8f982). These were written after reading D10 and `c2rnn._step_map`, and before running anything.
  - *Prediction (D10).* For a tied/tied model, `forward` with no map raises AssertionError at L ≥ 4 and works at L ≤ 3. An untied model with no map raises ValueError at L ≥ 4. For tied/tied, every valid explicit map gives bitwise identical states and logits, equal to my forward. For each untied scheme, two different maps give different logits, so the map is used there.
  - *Prediction (recovery).* The 28th `RECOVERY.json` entry (`r1-comp-interp-s51200-a1` `ckpt-10000`, r1's P2) has committed bytes equal to its recorded SHA-256 and to r1's published manifest.
- **04:14, generalization counterexamples** (cb8f982). These were written after reading its M0 `e5a4415` and the coordinator's one-line relay ("length 3 reproduces r1 exactly"), and before opening its tables, CSV or figure.
  - **CX-G1, against "reusable iteration" (Part A).**
    - *If tied/inter succeeds at L = 4–6.* I split rows by whether *any* 3-op window of the program, not only the prefix, is a held-out length-3 program. *Prediction:* in every seed with L = 4 accuracy ≥ 0.5, the two groups differ by less than 0.05 at each L. The alternative is success that is partly window-specific.
    - *If it fails.* I decode h_t at t = 4–6 by the nearest training-row centroid of h1–h3 for each symbol, and compare that with the readout. *Prediction:* centroid decoding is within 0.1 of the readout, so the failure is in the state, not only in reading it.
  - **CX-G2, against "the contrast reproduces on new tasks" (Part B).** A new draw that equals the original after relabelling symbols and ops (σ P_π(k) σ⁻¹ = P'_k for some σ in S8 and π in S4) is the same task with a different init, not a new task. I test every pair among the original and the two new draws over all σ and π. I also test, for each draw, whether its two splits correspond under such a relabelling. *Prediction:* no pair is equivalent.

- **04:11, time lane counterexamples** (db69507; written after reading the time lane's M0 `b7d1ab6` and before any of its results exist). Analysis only, on the lane's committed `ckpt-40000` and `ckpt-50000`.
  - **CX-T1, against "the supervision gap persists" (`tt-final`).**
    - *Test.* On the held-out rows that are wrong at 50,000, compute the logit gap (largest wrong logit minus correct logit) at 40k and at 50k. Also compute held-out final CE on ¬N rows over 40k–50k. A median gap that shrinks by more than 10%, or ¬N CE that falls steadily, means the accuracy plateau hides a function still moving toward the boundary. Separately, the last Adam step at 50k is decomposed into its decay part and its gradient part, with the fraction of √v̂ below eps, as in r1's R4 check.
    - *Prediction.* All three seeds censored, and held-out below 0.6 at 50k. In at least two seeds the median gap shrinks by more than 10% from 40k to 50k, so the lane's "persists" rule does not apply cleanly and the honest reading is "unresolved". The gradient term still dominates the update (more than half its norm), so this is not R4's eps regime. Low confidence on the gap.
  - **CX-T2, against "the readout gap is mainly speed" (`tu-inter`, if it crosses).**
    - *Test.* Does the catch-up re-tie the readouts? At 10k and 50k, compute the relative distances ‖V_k − V_3‖/‖V_3‖ for k = 1, 2, and the cross-readout accuracy of V_v on h_t over all 512 rows, as in r1's untied table (V3 on h1 was 0.44–0.62 at 10k).
    - *Prediction.* The distances shrink from 10k to 50k, but V3 on h1 stays below 0.9 in at least two seeds. So any catch-up is reached with readouts that remain position-specific, not by recovering the tied solution.
    - *Reading, either way.* If V3 on h1 rises above 0.9 together with the held-out crossing, "speed" means time to re-tie. That would be worth saying in the NOTE.

- **04:06, c2probe added to the pin check** (b0fb22c; written after the first `rev_pin.py` pass, sections 0–7, and before any probe was fitted). `c2probe.decode_table` on the recovered `r1-comp-rnn-inter-s51200-a1` `ckpt-10000` at probe seed 64300 (D5). Checks: its labels equal the oracle's symbols at positions 0–3; each probe's weights equal r1's own `composition/analyze.py fit_probe` on the same activations, bitwise; every table entry, re-scored with my own forward and argmax, equals `table[i][j]` with i the fitted position and j the scored one (a transposed table fails this); the diagonal equals r1's held-out probe accuracy. *Prediction:* all identical.
