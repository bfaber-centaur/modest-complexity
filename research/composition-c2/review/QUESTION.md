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

## Revisions

(Dated, before the matching analysis or run.)

- **04:1y, D10 and the post-pin recovery addition.** These were written after reading D10 and `c2rnn._step_map`, and before running anything.
  - *Prediction (D10).* For a tied/tied model, `forward` with no map raises AssertionError at L ≥ 4 and works at L ≤ 3. An untied model with no map raises ValueError at L ≥ 4. For tied/tied, every valid explicit map gives bitwise identical states and logits, equal to my forward. For each untied scheme, two different maps give different logits, so the map is used there.
  - *Prediction (recovery).* The 28th `RECOVERY.json` entry (`r1-comp-interp-s51200-a1` `ckpt-10000`, r1's P2) has committed bytes equal to its recorded SHA-256 and to r1's published manifest.
- **04:1y, generalization counterexamples.** These were written after reading its M0 `e5a4415` and the coordinator's one-line relay ("length 3 reproduces r1 exactly"), and before opening its tables, CSV or figure.
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
