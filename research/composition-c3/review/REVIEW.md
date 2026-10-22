# c3 reviewer: verdicts

Questions and predictions, committed before each measurement: [QUESTION.md](QUESTION.md). Evidence: `rev_c3pin.py` and its output, [out/rev_c3pin.txt](out/rev_c3pin.txt). Disclosure: the correction lane's per-step-readout base (`r1-rev-comp-rnn-untied-*`) is my r1 arm.

| item | verdict | section |
|---|---|---|
| Pin `be41d16` (D3) | **PASS**. No defect in what a branch or a correction computes. Notes N1–N4. My finite-difference check P3 missed its own tolerance at `ckpt-10000` (see §1). | §1 |
| Correction lane (PR #63, `c3f03cd`) | **PASS**. Every number in its tables, both figures and the NOTE recomputes exactly from my own code (B and C to 0 rows). Leak checks hold. CX-C1 held: the bins of gain recovered are stable across three other map and probe draws. CX-C2 failed narrowly: a val-tuned ridge decoder reads new prefixes up to 5/32 better, but recovers at most 6% of the gain, so the conclusion stands. C6: on adapter-held-out prefixes B is not purely decoder-plus-centroid. Wording notes W1–W3. | §2 |
| Scaffolding lane (PR #65, M2 at `4512bce`) | **PASS**. Every number in its six figure CSVs, its frozen L1–6 table and its checks recomputes from my own forward and oracle. That covers 317,262 trajectory values, 114 checkpoints exhaustively at L = 1–6, and 66 branch checkpoints rescored to their logged rows. The joins, schedules, exposure, manifests and registry hold. CX-S1 failed: length-3 retention at S = 10k does carry length 6 and the h1/h2 readout, so the lane's "persistence" stands. CX-S2 held for the total norm (4% below keep), but the shrinkage is in Ws, bs and bv, not V, and A grows. Three NOTE number nits (N-S1–N-S3) and a figure note. | §3 |
| Follow-up plan entries (`84dc657`) and correction wording (`897fa97`) | **PASS** (short checks, §3 end and §2 end) | §2, §3 |
| Scaffolding follow-up (PR #65 at `5317c54`): Adam reset against carried at S = 1k | **PASS**. Every number in NOTE §F and the seven follow-up CSVs recomputes from my own forward. The logged first update after the reset equals my fresh-state Adam update exactly. The matched controls hold bitwise, and the M2 nits and caption are applied. CX-F1 held: the reset does not slow early acquisition in either arm. Two wording points, W-F1 and W-F2: scope "carrying slows acquisition" to this switch and reset time, and note that up to 3k the reset speeds keep about as much as drop. One figure note. | §5 |
| Re-pin `82cef77` (D10, `--reset-optimizer-at`) | **PASS**. Runs without the flag are bitwise unchanged; the reset is exactly a fresh Adam state on the carried weights; resumes after or at the reset continue bitwise; every wrong reset history is refused. CX-R1 held: a reset also gives 2.1–2.6× larger steps over the next 100 updates, which the follow-up's wording should name. Note N5. | §4 |
| Synthesis (`coordinator/SYNTHESIS.md`, draft PR #70 at `05515a2`) | **PASS, with nits**. Every number traces to a reviewed output and recomputes. W-F1, W-F2, CX-R1, CX-F1 and the correction and M2 points are carried. Claim scope holds, and run counts and compute equal the registries. CX-SY1 failed: no predicted over-compression. The draft at `f2c7c63` misstated the scope of the A′ − A range, and `05515a2` already fixes it. The "Not independently checked" items (`511a381`, `285b3dd`, `5b92e1b`) are now checked and pass. Two misses of mine are recorded: my re-pin review did not catch the `corr_core` call-site break, and §5 passed NOTE §F2's range wording. Nits N-Y1–N-Y5. | §6 |

## 1. Pin check of `be41d16`

**PASS.** Every check of what the code computes passed. These were written before the runs (QUESTION.md §1, `05aac5a`), with additions at `d3717f9` and `ddff562` made before each pass.

- **P0, pin identity.** `common/*.py` at the branch head equals `be41d16`. c2's `common/` equals its own pin `b3e8132`.
- **P1, test suites.** `test_c3.py` (8 tests, 14.7 s) and `test_c2.py` (10 tests, 15.3 s) pass, each in its own process.
- **P2, schedule.** `weights_at` equals my reading of D5 at every t < 50,000, for S ∈ {1,000, 10,000} and D ∈ {1, 2,000}.
  - The final weight is exactly 1 throughout.
  - In the taper, the intermediate weight at S is 1 − 1/D, and it is 0 from S + D − 1 on.
  - `exposure()` equals my count for all 12 planned branches. For S..50k with the taper, the intermediate heads get 1,999 weighted updates, or 383,808 weighted row presentations (384 × 999.5).
  - The 12 branches total 534,000 updates.
- **P3, scheduled loss and gradient.** `c2rnn.loss_grad` at w = 0 0 1, 0.5 0.5 1 and 1 1 1 equals my loss to 5.6e-16 on all six tied/inter sources.
  - At `ckpt-1000`, a central finite difference of my loss along a random direction agrees with the gradient to ≤ 1.5e-7 relative.
  - My finite-difference check missed my own tolerance of 1e-6 twice:
    - At `ckpt-10000` it reached 3.8e-2 (P3). The state there is near-stationary: the analytic directional derivative is 2.6e-7 to 1.5e-6, and the error does not shrink with the step size.
    - At `ckpt-10000` plus N(0, 0.01²), it reached 8.2e-6 (P3b, added after P3).
  - I read both as limits of the test (round-off, curvature and ReLU kinks), not as a gradient error, but I did not prove that.
  - The gradient code is c2's pinned `c2rnn`, unchanged, which `test_c2.py` checks bitwise against r1's training. `test_c3.py` checks that the gradient is linear in the weights to 1e-13. A gradient error large enough to change a branch would show at the 1e-2 to 1 level at `ckpt-1000`.
- **P4, resume recipe check** (`check_source`, called directly, no training). Every pairing went the predicted way: 27 accepted and 14 refused.
  - **Accepted.** Each tied/inter `ckpt-1000` and `ckpt-10000` with its own r1 config, under keep, drop or taper at its step, or a switch after it. A c2 time `ckpt-20000` with its config. My per-step-readout `ckpt-10000` with its config.
  - **Refused.**
    - A config from another seed, or from tied/final, because its MANIFEST.json lists a different hash.
    - A switch or taper that starts before the checkpoint's step, including one update early (9,999).
    - A change of lr, wd, task, width or Adam eps.
    - The per-step-readout checkpoint paired with a tied config, or with another seed's per-step config (refused by run ID).
- **P5, `c3correct`**, on all six correction bases (`r1-rev-comp-rnn-untied-*` and `r1-comp-rnn-final-*` `ckpt-10000`), all 512 length-3 rows, depth 2.
  - The identity correction equals my forward exactly (difference 0).
  - The training-centroid correction for the true symbol equals my own V3 continuation exactly. Steps 1–2 are untouched.
  - Held-out class-mean accuracy reproduces c2 O4 (D2): 1.000 in every per-step-readout seed, and 0.875 / 0.875 / 0.969 in tied/final.
  - `frozen` refuses both a direct write and an Adam step. `params_sha256` is unchanged by the forwards and changes with a one-ulp edit.
  - A per-step-readout model at length 5 without a step map is refused.
- **P6, oracle.** c2task's original draw and split equal r1's tables. `c2task.rows` equals my oracle at L = 1–7: every program up to L = 5, and 3,000 sampled programs at L = 6 and 7.
- **P7, the coordinator's runs.**
  - The verification run's 41 rows (1,000–3,000) equal r1's published rows on every r1 key. Its weights are all 1 1 1, and its source is manifest-listed with the right run ID.
  - In both calibration runs, the logged weights equal my w(t) on every row and the exposure equals mine.
  - The committed files match their MANIFEST hashes, and the registry has one complete record per run with no failures.
- **P8, run IDs.** No ID repeats across the 18 `plan.json` files under `research/` (394 IDs).
- **P9, my runs** (plan committed before launch; registry in this directory).
  - **Keep from 10k.** `c3-rev-verify-keep10k-s51202-a1`, a no-change resume of r1 tied/inter s51202 `ckpt-10000` to 11,000, equals `c2-time-tt-inter-s51202-a1` on all 21 rows: r1 keys, `held_subset`, and `diag` on every row it has. So D5's reuse of the c2 keep arm matches c3train's no-change mode from 10k.
  - **Drop at 10k against c2train.** `c3-rev-drop-s51200-a1` (switch to 0 0 1 at 10,000) equals `c3-rev-c2final-s51200-a1` (c2train, `--step-weights 0 0 1`, the same `ckpt-10000`) at `ckpt-10300`, in weights, both moments and step, and on every row including `diag`.
    - Every parameter moves over the 300 updates, so no head is frozen (max |Δ|: Ws 5.5e-2 … bv 1.3e-3).
    - The Adam step carries to 10,300.
    - Its exposure reads [0, 0, 300] for this run and [10000, 10000, 10300] from init.
  - **Mid-taper resume.** `c3-rev-taper-whole-s51201-a1` runs S = 1,000, D = 2,000 to 2,600. `c3-rev-taper-cont-s51201-a1` resumes it at 1,800 (intermediate weight 0.5995). The two are identical at `ckpt-2600` and on rows 1,800–2,600, including the weights and `diag`.
    - The whole run's logged weights equal D5 on every row.
    - Its row at 1,000 equals r1's published row, so the branch point is untouched.

**Notes.** None of these changes a c3 result.
- **N1 (CX-P1).** `check_source` binds a checkpoint to its config through `MANIFEST.json` when one lists it, and otherwise only through the run ID in the checkpoint's meta.
  - The per-step-readout runs keep their hashes in the shared `research/regimes/review/runs/MANIFEST.sha256`, which it does not read. So for those runs the run ID is the only binding. The recovered files do match that manifest.
  - No c3 lane resumes these checkpoints; the correction lane freezes them. So this is a note.
  - **Remedy, if one is ever resumed.** Take `--resume-sha256` from `recovered/RECOVERY.json`, as the plan entries already do.
- **N2.** A resume from an unpublished local c3 run directory (mid-taper, as in P9) is also bound by run ID alone. `run_jobs.py` always passes `--run-id`.
- **N3.** The recipe's `adam` constants are labels. c3train calls `nets.adam_step` with its defaults (0.9, 0.999, 1e-8), which equal those labels. The check does not cover the evaluation or checkpoint settings, and these do not change a trajectory.
- **N4.** For a c2 source, the weights before that run's own resume step are taken from its config's constant step weights. c2 D9 and the c2 time review (§2 there) checked those chains.

**For the correction lane (P5).** On the original split, each depth-2 state occurs exactly three times among the training rows. So D6's "each unique state once" centroid gives the same centroids as a per-row mean there. The weighting matters only on grouped-prefix splits.

## 2. Correction lane (PR #63 at `c3f03cd`)

**PASS.** Checks and counterexamples were committed before I opened any lane result (QUESTION.md §2, `44e7556`; script at `ae77579`).
- **Evidence.** `rev_corr.py`, run as `c3-rev-corr-a1` (failed in C7 on my own panel mapping; [out/rev_corr_pass1.txt](out/rev_corr_pass1.txt)) and `c3-rev-corr-a2` ([out/rev_corr.txt](out/rev_corr.txt)). Also `rev_corr_note.py` ([out/rev_corr_note.txt](out/rev_corr_note.txt)).
- **My own code.** The forward pass is c2's `rev_pin.my_forward`. The prefix sets, split, centroids and map training are mine, written from the lane's QUESTION. The probe is c2's pinned `c2probe`.
- **Disclosure.** The per-step-readout bases are my r1 arm.

**C1–C5, C7: PASS in every cell.**
- **C1.** All 14 bases (6 original, 8 F2) hash to RECOVERY.json or MANIFEST.json at step 10,000. Each loaded file's `params_sha256` equals the lane's before, after-fit and after records.
- **C2.** My grouped split equals the lane's for every task: 8/4/4 per symbol, disjoint. Each prefix's 4 rows stay in one group, and eval = 32 held + 96 training rows.
- **C3.** My accuracies equal every cell of the lane's four `accuracy.csv` files (10 main and 30 grouped cells per base). A, D and E are exact. B and C, from my own map code, are also exact: the difference is 0 rows everywhere.
- **C4.** The symbol accuracy, centroid error and decoder accuracy of the corrected states equal the lane's `states.csv`.
- **C5, leakage, from the code.** B, C and D get h_2 alone, and only E gets the true symbol. Fits use the fit prefixes only. Calibration uses fit → val only, with seed 900200, and no final answers.
- **C7.** All 240 values in `figures/correction.csv` and all 320 in `figures/followup_f2.csv` recompute from my numbers. I looked at both PNGs. They show every seed, mark chance and label E as the oracle reference. The fill code differs between panels (held/train in panels 1–2, grouped/main in 3–4), but each panel title says which.
- **NOTE numbers** (`rev_corr_note.py`). Every quoted range matches:
  - main: A, B, C, D = E;
  - grouped eval: E 0.84–1.00; B −50% to 0% of the gain; D above A only in per-step 51201 (0.625 vs 0.562); training rows 0.17–0.64; decoder 0.53–0.66 / 0.22–0.31;
  - F2: B 79–100% (main), and in grouped every learned arm below A, B −110% to −13%, D −90% to −7%, training rows 0.19–0.62, decoder 0.19–0.44, E 0.91–1.00.

**CX-C1, the map and probe draws: held.** I refit B with map seeds 73010–73012 and D with probe seeds 73020–73022. The bin of gain recovered did not change in any of the 6 original bases, in either evaluation:
- main: above 60% for both B and D;
- grouped: below 20% for both B and D.

So the lane's single draws (71200, 71301) carry its per-base statements.

**CX-C2, other decoders: failed narrowly. The conclusion stands.** On grouped eval prefixes of the 11 tied/final bases (3 original, 8 F2):
- **The prediction missed.** A ridge decoder with λ chosen on val prefixes raises held final accuracy over the lane's probe by up to +0.156 (5 of 32 rows, `ops63001-ls63102-s63201`) and +0.125 (original s51201). Decoder accuracy rises by up to +0.156. I predicted at most 0.10 and 0.15.
- **The gain stays small.** Even so, ridge recovers at most +6% of the gain (original s51201; −80% to −11% elsewhere), so "recovers essentially none of the gain" holds for this second decoder too.
- **Nearest-centroid decoding is worse than the probe everywhere.**
- **What it means.** The NOTE's decoder accuracies (0.22–0.31; 0.19–0.44 in F2) belong to its unregularized probe; ridge reaches 0.28–0.34 and up to 0.56 in F2. That fits the NOTE's own bound: "this adapter under this exposure… does not show whether the symbol is linearly available in h_2".

**C6, decoder-plus-centroid accounting: holds on seen states, not on adapter-held-out prefixes.**
- **Main evaluation.** B's corrected state never has the wrong nearest-centroid symbol, and B ≤ D + 1/32 in all 14 bases.
- **Grouped eval prefixes.**
  - B exceeds D by more than 1/32 in 3 of 14 bases: tied/final s51201 0.312 vs 0.219, which is the NOTE's reported failure of its prediction 4, plus two F2 bases, 0.344 vs 0.281 and 0.250 vs 0.188. The NOTE reports only the first, since F2 made no such prediction.
  - On held rows where B's corrected state decodes to the wrong symbol, B is still right on 13–33% in 8 of 14 bases; I predicted at most chance, 12.5%.
- **What it means.** B's output on new prefixes is not a clean centroid snap: it keeps some of the natural state. B stays below A there, so this is not evidence of a useful extra computation. It does mean that "B behaves like a decoder plus re-encoder" holds on seen states only.

**Wording notes (minimal remedies).**
- **W1.** The short answer says that on seen states the correction "recovers the full gain". D does in all six bases. B does in five; in tied/final s51202 it recovers 75% (0.805 against E 0.969). Remedy: "D recovers the full gain, and B 75–103%".
- **W2.** "Replacing a correctly handled state with a slightly wrong centroid" is imprecise. D substitutes the centroid of a wrong symbol, and B an off-centroid state. Remedy: say that.
- **W3.** "The same holds for eight bases from new task draws" should say that these are tied/final only, as the F2 section does. The NOTE's prediction table already says that F2's predictions came after the original grouped result.

**Charter priorities.**
- True symbols enter only the targets and E.
- Fits use only their declared split.
- Prefix copies stay together.
- Adapter-held-out and base-held-out rows are reported separately.
- Base weights are unchanged.
- Rescue is kept apart from identification ("What they do not show").

The specificity checks are CX-C1, CX-C2 and C6 above.

**Addendum, the lane's wording fixes at `897fa97`: PASS.** Every new number recomputes from my `rev_corr.json`:
- B recovers 100/100/100% (per-step) and 101/103/75% (tied/final) on main, and 79–100% in F2. D is exact.
- "B above D on grouped held rows" holds in 3 of 14 bases, exactly the three it names.

One nit, which the lane has since applied (`285b3dd`): the 13–33% on wrong-symbol rows holds in 8 of 14 bases, and the other six are 5–12.5%.

## 3. Scaffolding lane (PR #65, M2 at `4512bce`)

**PASS.** The checks S1–S7 and the counterexamples CX-S1 and CX-S2 were committed at `ad55463`, before any branch result existed. Implementation choices are at `d7a1613`.
- **Evidence.** `rev_scaf.py` (run `c3-rev-scaf-a1`, 78 s), output [out/rev_scaf.txt](out/rev_scaf.txt). It reads the lane's files from `git archive 4512bce`. My forward and oracle are the c2 reviewer's.
- **NOTE numbers.** `rev_scaf_note.py`, output [out/rev_scaf_note.txt](out/rev_scaf_note.txt).

**Recomputation (all pass).**
- **S0, rows and N.** My training and held-out rows equal `c2task.rows` in order. My N mask (the true label is among the one-slot training neighbours' labels) equals the pinned one, 86 / 42.
- **S1, branch points.** All 12 configs resume r1 tied/inter `ckpt-S` at its RECOVERY.json hash, which equals the file and is manifest-listed. The source run ID, the branch's run ID and every explicit recipe flag are right. Each branch's first row equals r1's published row at S on every r1 key.
- **S2, schedule.** `step_weights` equals my w(t) on all 10,692 logged branch rows, and the final weight is exactly 1. Under the taper, the intermediate weight is 0.9995 at S, 0.0005 at S + 1,998, and 0 from S + 1,999. Exposure in every config, this run and from init, equals my recount.
- **S3, records.**
  - Every committed file matches its MANIFEST.json hash.
  - All 66 branch checkpoints rescore with my forward to their logged rows: train and held-out accuracy per step exactly, CE to 1e-9, and the prediction string. Count, meta step and run ID are right.
  - The registry holds exactly one running and one complete event for each of the 14 plan entries: no failure, cap or retry.
- **S4, frozen lengths.**
  - All 114 × 9 values of `scaf_frozen.csv` equal my exhaustive evaluation (8·4^L rows per length, plus held-out, N and ¬N). The checkpoint set is exactly the declared fixed set.
  - My values for keep and final-only at 10k and 50k equal both c2 generalization `length.csv` files (36 + 36).
  - Every evaluated prediction string equals the logged one.
- **S5, arithmetic.**
  - All 462 rows of `scaf_retention.csv` recompute: Δ_branch, Δ_keep and Δ_D to 1e-12, and the labels. 84 of 84 endpoints are "retained", and the largest |Δ_keep| at 50k is 0.010, so the 0.05 rule and its tolerance are nowhere near decisive.
  - All 36 rows of `scaf_threshold.csv` recompute from the joined trajectories. Every logged `held_subset` equals my split of its prediction string.
- **S6, movement.** In every branch every parameter array changes over S → S + 2,000 and S + 2,000 → 50k (smallest max change 5.7e-3). The Adam count carries (S + 2,000; 50,000).
- **S7, figure.**
  - All 16,698 rows (317,262 values) of `scaf_main.csv` recompute from the logged rows.
  - `scaf_optimizer.csv` and `scaf_compatibility.csv` recompute too, the latter with my own neighbour sets.
  - I looked at `scaf_main.png`. It shows what the CSVs hold. One note: at 50k, A, B and C coincide at 1.0 in the length-curve panel and in the length-3 panels, and only C's green is visible. A caption line ("coincident arms overplot; C is drawn last") would stop a reader from thinking keep or drop is missing.

**Counterexamples.**
- **CX-S1 failed: length-3 retention does carry length transfer here.** At S = 10k, B and C are retained at length 3 in all three seeds.
  - My L6 at `ckpt-50000` is at most 0.0023 below keep's.
  - The shared readout reads h1 and h2 on the training rows at 1.0 at `ckpt-50000`, and at every logged row after S, in every seed.

  So the lane's "persistence at S = 10k" is not an artifact of looking only at length 3. Its own prediction 3 also failed (NOTE), for the same reason.
- **CX-S2 held for the total norm, not for where the shrinkage is.**
  - At 50k, every S = 10k drop branch has 4.1–4.3% less parameter norm than keep, in all three seeds. Taper is the same.
  - The groups that shrink are the start embedding Ws (0.68 of keep's norm), its bias bs (0.71–0.74) and the readout bias bv (0.55–0.78).
  - The readout V shrinks only to 0.94–0.95. The shared transition A grows to 1.04–1.05.

  This is a post-hoc reviewer observation, not a lane claim: without the step-1/2 loss, scale moves from the input embedding into the iterated transition. It does not change any of the lane's conclusions.

**The NOTE's numbers.** O1–O8, the crossing table and handoff item 2 all recompute. There are three small inaccuracies, and none changes a conclusion.
- **N-S1, O1.** The L4–6 range of Δ_D(50k) is +0.85 to +0.90, not +0.91 (the maximum is 0.9047).
- **N-S2, O2.** "Keep reaches 1.0 by 20k" holds for s51200 and s51202. In s51201, keep's L6 is 0.986 at 20k and 1.0 from 30k (O4's own table says 0.99).
- **N-S3, O5.**
  - D's train CE at 50k is quoted as "8.6 / 11". Those are s51200's values (8.60 / 11.15). The three seeds span 8.6–10.2 on step 1 and 11.1–11.9 on step 2.
  - The step-1 minimum after S (0.969) is at 17,200, not at 7,100; 7,100 is the step-2 minimum (0.922). Both are in drop at 1k, s51201.

**Interpretation, under the lane's pre-registered rules.**
- **Lasting benefit.** At both S and in all seeds, B and C end 0.30–0.44 above D on held-out L3.
- **Persistence at S = 10k and further improvement at S = 1k.** Both hold as stated.
- **Wording.**
  - The NOTE keeps paths separate from endpoints.
  - It says that a drop changes both the loss and the carried moments.
  - It states "50k is not an asymptote".
  - It names the scope (one draw, paired seeds).
- **The threshold reading** was changed after the results (ADAPTIVE 06:00Z), to the c2 convention that QUESTION.md cites by name. It changes nothing at S = 1k. At S = 10k, prediction 1's "s51201 reaches the threshold" was already met at S on held-out (0.961). The informative test is ¬N, which crosses after S (11,300 / 11,150), and the NOTE reports that.

**Follow-up plan entries (`84dc657`): PASS.**
- All six reset runs resume the right seed's r1 tied/inter `ckpt-1000` at the file's hash. R = 1,000 equals the resume step.
- Each drop-reset run's flags equal its carried drop branch's exactly, apart from the reset. Keep-reset differs from drop-reset only by having no switch.
- Checkpoints are at 3k, 10k and 20k–50k, with `--diag`, 50k updates and unique IDs.
- The four 1k → 3k verify runs, with no reset, are the right matched-control checks, and they give carried keep a 3k point.
- The addendum's prediction 3 is consistent with my CX-R1 (§4).

## 4. Re-pin check of `82cef77` (D10, `--reset-optimizer-at`)

**PASS.** Checks RP0–RP7 and CX-R1 were committed before any run at the new pin (QUESTION.md §4, `5ce9d97`; continuation entries `55cf081`; script `f3abedc`). Evidence: `rev_repin.py` (run `c3-rev-repin-a1`, 39 s), output [out/rev_repin.txt](out/rev_repin.txt). Every line passed.

- **RP0, scope.** `be41d16..82cef77` changes only `c3train.py`, `test_c3.py` and the README in c3's `common/`. `c3correct.py`, c2's `common/` (`b3e8132`) and `research/regimes/common/` are unchanged, and HEAD's `common/` equals `82cef77`.
- **RP1, tests.** `test_c3.py` (9 tests, 15.5 s) and `test_c2.py` (10 tests, 13.4 s) pass, each in its own process.
- **RP2, runs without the flag.** My `be41d16` drop branch (10,000 → 10,300) and whole taper run (1,000 → 2,600), re-run at `82cef77`, are bitwise equal: every checkpoint (weights, both moments, count) and every metrics row, including `--diag`.
- **RP3, old sources.** My pin check's 41 resume pairings give the same outcomes under the new `check_source` (27 accepted, 14 refused). A `be41d16` c3 checkpoint, whose config has no `reset_at`, is accepted with its own flags and refused with a reset declared before its step.
- **RP4, the reset.**
  - The reset branch's `ckpt-1000` equals the r1 source in weights, moments and count, so it is saved before the reset.
  - At 1,000 + k (k = 1, 2, 3) the saved Adam count is k and meta `step` is 1,000 + k.
  - Weights and both moments equal my own Adam from zero moments on the carried weights, bitwise, in drop (s51200) and keep (s51201).
  - In the update at 1,000, every coordinate with \|g\| > 1e-4 (69–70% of them) moves by lr to within 1e-4 relative, after removing decay. That is the fresh-state sign step.
- **RP5, continuation.** Resuming the reset branch from its `ckpt-1200` (after the reset), or from its `ckpt-1000` (at the reset, pre-reset state), with the same flags, gives its `ckpt-1400` and every row bitwise.
- **RP6, refusals.** From the reset branch's `ckpt-1200`:
  - Refused: no reset declared, a reset at 1,100, a reset at 1,300, and an Adam count forged to 1,200.
  - Accepted: a reset at 1,000.

  Elsewhere:
  - Accepted: its `ckpt-1000` without a reset (a legitimate branch from the pre-reset state), and r1 `ckpt-10000` with a reset at 10,000 or 20,000.
  - Refused: r1 `ckpt-1000` with a reset at 500, and r1 `ckpt-10000` with a reset at 1,000.
- **RP7, records.** Both reset configs record `schedule.reset_at = 1000`. The drop branch's exposure equals a drop at 1,000 without a reset: [1,000, 1,000, 1,400] updates from init.

**CX-R1 held: a reset is more than removing memory.** In-process, I ran 100 updates from each r1 tied/inter `ckpt-1000`, carried against reset, for keep and drop, with my Adam (equal bitwise to c3train per RP4).
- The reset branch's mean RMS per-coordinate move (decay removed) over 1,000–1,099 is 2.14–2.64× the carried one in all six seed × arm pairs. It is 1.1–1.3e-4 against 4.3–6.2e-5.
- The ratio is 15–19× on the first update. It is still 1.4–1.8× over updates 1,090–1,099.

The carried second moment holds about 1,000 updates of history from larger early gradients. A fresh state re-estimates it from the current, smaller gradients, and its bias correction makes the first updates sign steps of size lr. So the reset arms differ from the carried arms in two ways: Adam's memory of earlier gradients is gone, and the effective step size is larger for at least the first hundred updates (β2 = 0.999 suggests several hundred). This is a design fact, not a defect. **Suggested wording for the follow-up:** "a reset removes the optimizer's history and also enlarges the next few hundred steps." Do not write "the reset removes only the moments' memory." A difference between reset and carried arms can come from either.

**N5, a note, not a defect.** A reset step outside the run's own range, such as `--reset-optimizer-at` beyond `--updates` or before a resume's start, is not rejected by argument parsing. The second case is refused by `check_source` when the source had no reset there (RP6, r1 `ckpt-1000` with a reset at 500). The first case runs without a reset, while its config still records `reset_at`. The planned follow-up resumes at R = 1,000 with R = 1,000, which is inside the range, so this does not affect it. I will check the six plan entries when they are pushed.

## 5. Scaffolding follow-up (PR #65 at `5317c54`): Adam reset against carried at S = 1k

**PASS.** The checks F1–F4 and CX-F1 were committed at `8340a69`, before I read any follow-up result. The script was committed at `5236c16`, before its run.
- **Evidence.** `rev_scaf_fu.py` (run `c3-rev-scaf-fu-a1`, 72 s), output [out/rev_scaf_fu.txt](out/rev_scaf_fu.txt). Every line passes.
- **Post-hoc table.** [out/rev_scaf_fu_posthoc.txt](out/rev_scaf_fu_posthoc.txt), from `rev_scaf_fu_posthoc.py`, which reads logged rows only.

**Recomputation.**
- **F1, the reset runs.**
  - All six resume r1 tied/inter `ckpt-1000` at its RECOVERY.json hash and record `reset_at` 1000, with the right flags.
  - Each first row equals r1's at 1,000. Step weights and exposure equal my recount.
  - The saved Adam count is step − 1,000 at every committed checkpoint (2,000 at 3k, 49,000 at 50k).
  - The logged update and gradient norms at row 1,000 equal my own fresh-state Adam update on the source weights exactly.
- **F2, matched controls.**
  - The three carried-keep runs (1k → 3k, no reset) equal r1's rows in every r1 key.
  - The carried-drop run equals the drop branch in every key including `diag`, and its `ckpt-3000` equals the branch's bitwise.
- **F3, records.**
  - Manifests and the registry (24 entries, one running and one complete event each) hold.
  - All 37 committed follow-up checkpoints rescore to their logged rows.
  - All 99 × 9 frozen values equal my exhaustive L1–6 evaluation.
- **F4, arithmetic.** Every value of the effects, threshold, transient, optimizer and compatibility CSVs, and all 281,865 values of `scaf_followup.csv`, recompute. The keep rows for 1k–3k come from the verify runs, as the lane builds them. The rule label is "weights at S suffice" in all three seeds.
- **NOTE §F.** Each of these recomputes as quoted:
  - F1 (|reset effect| ≤ 0.003 at 50k);
  - the F2 crossing table and its effect values;
  - handoff 6 (0.766 against 0.438);
  - the F3 ratios (16.9–21.6× first update; 1.15–1.75 over 1,050–1,500; 7 of 12 above 1.5);
  - F3's train CE claims;
  - the F4 eps onsets and F5;
  - the O7 ratio (0.53–0.56).
- **The M2 nits.** N-S1–N-S3 are applied in the NOTE, and the overplot caption is in the figure code.

**CX-F1 held: the reset does not slow early acquisition, in either arm.**
- **B′ − B** on held-out L3 is +0.086 / +0.055 / +0.039 at 3k and +0.055 / +0.125 / +0.125 at 10k.
- **A′ − A** is +0.070 / +0.047 / +0.047 at 3k (A from r1's logged row) and −0.008 / +0.008 / −0.016 at 10k.

So, as committed: "weights at S suffice" (B′ ≈ B at 50k) cannot exclude a moments effect that the burst of larger steps offsets. Here the burst, if anything, helps.

**Post-hoc (after reading the NOTE): when the keep control is informative** ([out/rev_scaf_fu_posthoc.txt](out/rev_scaf_fu_posthoc.txt)).
- **Up to 3k**, the reset raises held-out L3 by about the same amount in keep (A′ − A up to +0.07) as in drop (B′ − B up to +0.09).
- **From 4k on**, keep's effect fades to −0.07 to +0.05, while drop's grows to +0.06 to +0.15.
- **Keep is not yet at its ceiling at 4k–6k**: A is at 0.48–0.86 there. So the drop-specific part of the reset effect is visible without a ceiling. In the NOTE's "difference on L6 at 20k" (+0.25 / +0.31 / +0.18), keep is at 0.99–1.0, so that difference is effectively B′ − B alone.

**Wording points (minimal remedies).**
- **W-F1, scope of "carrying the Adam state across the drop slows acquisition".**
  - **What the data support.** At one switch time that is also the reset time (1k), one recipe and three paired seeds, all three seeds show the same direction: the carried drop branch crossed 2,350–5,900 updates later than the reset one. Keep moved −400 to +750.
  - **Remedy.** Scope the sentence: "Here (S = R = 1k, this recipe), the drop branch with its Adam state carried crossed 2,350–5,900 updates later than with a reset at the switch."
  - **Clarity.** Replace "the cause is the reset, not a single mechanism" with "the effect belongs to the reset as a whole".
  - **CX-R1.** The NOTE already names both components (lost history; larger early steps) and says the run does not separate them, which is what CX-R1 asked for.
- **W-F2, the keep control.** Add one line that the post-hoc table supports:
  - up to 3k, the reset speeds keep about as much as drop, which fits the early-step burst;
  - the drop-specific part appears from about 4k, while keep is still below its ceiling;
  - it is consistent with stale second moments (O7: carried B's update norm falls to 0.53–0.56 of its value at S by S + 2,000), but it does not prove it.

  Also say that from about 8k on, keep is near its ceiling, so A′ − A cannot show a speed-up there.

**Figure note.**
- **The problem.** In `scaf_followup.png`'s 900–6,000 zoom, the frozen L4/L6 panel draws straight lines between fixed checkpoints: 1k → 3k → 10k, and 1k → 10k for keep, which has no 3k checkpoint. Inside the window they look like data at 4k–6k, and keep's line sits below A′'s there only because of the interpolation.
- **Remedy.** In the zoom, draw markers only, or say "lines interpolate between checkpoints".

## 6. Synthesis (`coordinator/SYNTHESIS.md`, draft PR #70 at `05515a2`)

**PASS, with nits.**
- **Order of work.** The checks SY1–SY6 and CX-SY1 were committed at `696edf3`, before I read the synthesis. I read the draft at `f2c7c63`, and then its revision at `05515a2`. The post-hoc parts of `rev_synth.py` were committed at `42210be`, before their run.
- **Evidence.**
  - [out/rev_synth.txt](out/rev_synth.txt): the registries, the keep control on logged rows, and the correction call site at the head.
  - My earlier outputs: `out/rev_corr*.txt`, `out/rev_scaf*.txt`, `out/rev_scaf_fu*.txt` and `out/rev_repin.txt`.

**SY1, numbers: PASS.**
- **Correction (§1).** Every value in the table and the bullets equals `out/rev_corr_note.txt`. For example, B recovers 75–103% (79–100% in F2), D is above A in 1 of 6 bases (2 of 32 rows), and the learned arms fall to 0.17–0.64 on training-program rows.
- **Scaffolding (§2).** The following all equal `out/rev_scaf_note.txt` and the threshold CSV:
  - the 50k table, and the 84 of 84 measures within 5 pp;
  - Δ_D, which uses my N-S1 rounding (+0.85 to +0.90);
  - the crossings (drop 14,600–20,450; taper 12,050 / 21,400 / 15,300; keep 7,700–9,150);
  - the lags behind keep (drop 6,900–12,250; taper 4,350–12,250);
  - −0.39 / −0.86 for drop s51202 at 10k;
  - the S = 10k lag in L6 of up to 0.34;
  - the readout minimum of 0.92 and frozen L1–2 ≥ 0.95;
  - the eps onset of about 29k.
- **"D reads h1 and h2 near chance."** This holds. D's step-1/2 training accuracy at 50k is 0.03–0.16 against a chance level of 1/8.
- **Follow-up.** The crossing table, the 2,350–5,900 gap, and the third-to-a-half of the lag (2,350 / 6,900, 5,900 / 10,850 and 4,100 / 12,250) all recompute. So do B′ − B on L6 at 20k, the transient ratios and the predictions.
- **Updates.** The update counts (534k + 294k + 8k) recompute.

**SY2, review points carried: PASS at `05515a2`.**
- **Correction.** W1 ("D recovers the full gain, and B 75–103%"), W2 and W3 are respected. C6 and the 13–33% scope at `285b3dd` (8 of 14 bases, 5–12.5% in the other 6) equal my per-base values.
- **Scaffolding.** CX-S1 and CX-S2 are reported, and N-S1–N-S3 are carried.
- **Follow-up.** W-F1 is quoted as the remedy proposed it. W-F2 is a bullet of its own. CX-R1 and CX-F1 are stated ("the effect belongs to the reset as a whole"; "weights at S suffice" cannot exclude a moments effect that the burst offsets).
- **At `f2c7c63`.** The draft said A′ − A "stays between −0.06 and +0.09 on every measure and time" and that the reset "barely moves keep". The range covers only the fixed times from 10k on, where keep is at its ceiling. On logged rows up to 3k, A′ − A on held-out L3 reaches +0.08, against +0.09 for B′ − B. From 3k to 10k, A′ − A on ¬N spans −0.19 to +0.17. The revision at `05515a2` replaced both statements with the W-F2 bullet, so this is resolved.

**SY3, claim scope: PASS.**
- No statement of necessity or impossibility appears. D, A and E are named as tested baselines, not bounds.
- Every comparison gives its seed count. The follow-up is scoped to S = R = 1k and this recipe.

**SY4, runs and compute: PASS** (`rev_synth.py`).
- **Correction.** 6 runs, 0 failed. The union of their wall times is 1.16 min; the synthesis says "about 1.1".
- **Scaffolding.** 24 runs, 0 failed, with a union of 43.99 min, against the synthesis's 27.2 + 16.8.
- **Coordinator.** 3 runs, with a union of 0.52 min.
- **Reviewer.** 16 registered runs, of which 1 failed (`c3-rev-corr-a1`), with a union of 10.09 min. My tally of 11.5 min in the README also counts the reads-only checks that ran outside `run_jobs`. This check adds about 2 s, so the total stays at 11.5 of 20 min.
- **Observations and interpretation** are kept apart: each section ends in its own "Reading".

**SY5, packet shape: PASS.** It covers the correction, the scaffolding and the limitations, and asks two next questions.

**SY6, disclosure: partial.** The synthesis links REVIEW.md, whose first lines disclose that the per-step bases are my r1 arm, but it does not say so itself (N-Y2).

**CX-SY1 failed, at both versions.** I found none of the three forms I predicted:
- The follow-up's slowing is scoped: S = 1k in the setting, one reset time in the limitations, and S = R = 1k at `05515a2`.
- It is not attributed to the moments: "whether the history or the step size is responsible is open".
- CX-S2 is not reported as a uniform shrinkage.

The one over-compression I did find, in the keep-control range at `f2c7c63`, was a form I had not predicted, and `05515a2` already fixes it.

**The "Not independently checked" items, now checked.**
- **`511a381`** (scaffolding NOTE W-F1 and W-F2, and the follow-up figure zoom).
  - The NOTE text matches my remedies, and its numbers match §5.
  - The diff from `f2c7c63` changes no CSV.
  - The zoom now draws the frozen L4/L6 at the 1k and 3k checkpoints as markers only.
- **`285b3dd`** (the correction NOTE's 13–33% scope). Its counts equal my C6 values: 8 bases at 13–33% and 6 at 5–12.5%.
- **`5b92e1b`** (the `corr_core.load_base` call site).
  - At the head, the pre-fix call fails with a TypeError, which reproduces the break.
  - The fixed call passes the meta step and the Adam count, which are equal for these bases since they have no reset. With it, `load_base` reproduces every recorded field of `checks.json` (hashes, steps, recipe, schedule, source, params hash) for all 14 bases.
  - `test_corr.py` passes.

**Two misses of mine.**
- **The re-pin review.** My §4 ran `test_c3.py` and `test_c2.py` but not `test_corr.py`, and it did not look for other callers of `check_source`. So I did not catch the break that `5b92e1b` fixes. The coordinator found it.
- **The follow-up review.** In §5 I passed NOTE §F2's "A′ − A is between −0.06 and +0.09 on every measure and time". My own output showed that the range was computed from 10k on, and I did not flag the wording. The line is still in the NOTE at `05515a2` (N-Y5).

**Nits (minimal remedies), none blocking.**
- **N-Y1.** "The reused carried runs reproduce A and B row for row" should say "A in all three seeds, and B in s51200 (including `ckpt-3000`, bitwise)". Only one carried-drop check was run.
- **N-Y2.** In §1 *Setting*, add "(the per-step bases are the reviewer's r1 arm; REVIEW.md)".
- **N-Y3.** In the CX-S2 bullet:
  - Ws is at 0.68–0.69 of keep's norm.
  - The group ratios were part of the pre-registered CX-S2 (its "V shrinks most" half failed), so "a reviewer counterexample, not a mechanism" is more accurate than "a post-hoc reviewer observation".
- **N-Y4.** Replace *Not independently checked* with a pointer to REVIEW §6. Set the reviewer row to "16 (1 failed) · 11.5 min (20)", and add that the tally includes reads-only checks run outside `run_jobs`.
- **N-Y5** (lane NOTE §F2). Scope "on every measure and time" to "at the fixed times from 10k on (keep has no 3k checkpoint)".
- **Optional.**
  - The zoom panel's title says "lines join checkpoints", but the zoom draws none.
  - CX-C2 was run on the 11 tied/final bases.
  - The correction wall time from its registry rounds to 1.2 min.
