# c3 reviewer: verdicts

Questions and predictions, committed before each measurement: [QUESTION.md](QUESTION.md). Evidence: `rev_c3pin.py` and its output, [out/rev_c3pin.txt](out/rev_c3pin.txt). Disclosure: the correction lane's per-step-readout base (`r1-rev-comp-rnn-untied-*`) is my r1 arm.

| item | verdict | section |
|---|---|---|
| Pin `be41d16` (D3) | **PASS**. No defect in what a branch or a correction computes. Notes N1–N4. My finite-difference check P3 missed its own tolerance at `ckpt-10000` (see §1). | §1 |
| Correction lane (PR #63, `c3f03cd`) | **PASS**. Every number in its tables, both figures and the NOTE recomputes exactly from my own code (B and C to 0 rows). Leak checks hold. CX-C1 held: the bins of gain recovered are stable across three other map and probe draws. CX-C2 failed narrowly: a val-tuned ridge decoder reads new prefixes up to 5/32 better, but recovers at most 6% of the gain, so the conclusion stands. C6: on adapter-held-out prefixes B is not purely decoder-plus-centroid. Wording notes W1–W3. | §2 |
| Scaffolding lane | in progress (PR #65 at `4512bce`) | |
| Re-pin `82cef77` (D10, `--reset-optimizer-at`) | **PASS**. Runs without the flag are bitwise unchanged; the reset is exactly a fresh Adam state on the carried weights; resumes after or at the reset continue bitwise; every wrong reset history is refused. CX-R1 held: a reset also gives 2.1–2.6× larger steps over the next 100 updates, which the follow-up's wording should name. Note N5. | §4 |

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
