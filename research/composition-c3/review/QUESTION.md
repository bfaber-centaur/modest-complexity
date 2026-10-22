# c3 reviewer: questions and predictions

Role: [../coordinator/charter/04-reviewer.txt](../coordinator/charter/04-reviewer.txt). Assignment, budget (20 min) and gates: [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md). Run IDs `c3-rev-`, new seeds 73000–73999 (none used so far; my runs reuse the original init seeds 51200–51202 for pairing, as D7 allows).

**Disclosure.** The c3 correction lane's per-step-readout base, `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`, is my r1 arm.

This file is written before each measurement. Later entries go under "Revisions", labelled by the commit that records them.

## 1. Pin check of `be41d16` (DECISIONS D3)

Code under test is imported read-only or run as a subprocess. The forward pass, loss and oracle are mine from c2 (`../../composition-c2/review/rev_pin.py`: `my_forward`, `my_loss`, `oracle_rows`, written from r1's tables), and the schedule is re-derived from the D5 text. Script: `rev_c3pin.py`.

| # | check | prediction |
|---|---|---|
| P0 | `common/*.py` at the branch head equals `be41d16`; c2's `common/` equals its pin `b3e8132` | identical |
| P1 | `test_c3.py` and `test_c2.py`, each in its own process | all pass |
| P2 | Schedule. My w(t) from D5 ("the update at S + k uses intermediate weight 1 − (k+1)/D, reaching 0 at S + D − 1"; final weight exactly 1) against `weights_at`, for S ∈ {1,000, 10,000}, D ∈ {1, 2,000}, every t < 50,000. Exposure counts against `exposure()` for the 12 planned branches, and their total of 534k updates | identical; total 534,000 |
| P3 | The scheduled loss and gradient. `c2rnn.loss_grad` at w = [0, 0, 1], [0.5, 0.5, 1] and [1, 1, 1] on each tied/inter `ckpt-1000` and `ckpt-10000` against `my_loss`, plus a central finite difference of `my_loss` along one random direction | losses equal to 1e-12; directional derivatives agree to 1e-6 relative |
| P4 | Resume recipe check (`check_source` called directly; no training). Accepted: each tied/inter `ckpt-1000`/`ckpt-10000` with its own r1 config, at S equal to its step or later; a c2 time `ckpt-20000` with its own config; my per-step-readout `ckpt-10000` with its own config. Refused: (a) another seed's config, (b) the tied/final config with weights 0 0 1, (c) a switch before the checkpoint's step, (d) lr, wd, task or width changed, (e) a c2 time `ckpt-20000` with S = 10,000 | as listed. My guess at a gap: the per-step-readout run directory has no `MANIFEST.json`, so that source is bound to its config only through the checkpoint's run ID |
| P5 | `c3correct`. On the six correction bases (`r1-rev-comp-rnn-untied-*` and `r1-comp-rnn-final-*` `ckpt-10000`) at depth 2, on all 512 length-3 rows: identity against `my_forward`; the class-mean correction (training centroids of the true depth-2 symbol) against `my_forward` with h2 replaced; the readout used after the correction is V3 in the per-step-readout arm; `frozen` refuses writes, including through `nets.adam_step`; `params_sha256` is unchanged by a forward pass and changes with any one element | identity and class-mean logits equal mine to 1e-12. Held-out class-mean accuracy reproduces c2 mechanisms O4 (D2): 1.00 in the per-step-readout arm, 0.88 / 0.88 / 0.97 in tied/final |
| P6 | Oracle at any length: `c2task.rows` against `oracle_rows` for L = 1–7 (every program for L ≤ 5, 3,000 sampled per length above) | identical |
| P7 | The coordinator's verification run: all 41 rows against r1's published rows on every r1 key; `step_weights` all [1, 1, 1]; `resumed_from.source` (`manifest_listed`, run ID). The two calibration runs: logged `step_weights` against my w(t); `exposure` against mine; `runtime.json` complete; registry complete | identical / complete |
| P8 | Run IDs: no ID repeated across every `plan.json` and `registry.jsonl` under `research/` | unique |
| P9 | My verification runs (plan below), through `run_jobs.py`: | |
| | `c3-rev-verify-keep10k-s51202-a1`: no-change resume of r1 tied/inter s51202 `ckpt-10000` to 11,000 with `--diag`, against `c2-time-tt-inter-s51202-a1` rows 10,000–11,000 on every key (r1 keys, `held_subset`, `diag`). This is the arm-A reuse D5 relies on | identical on all 21 rows |
| | `c3-rev-drop-s51200-a1` (c3train, S = 10,000, switch 0 0 1) and `c3-rev-c2final-s51200-a1` (c2train `--step-weights 0 0 1`), both resumed from r1 tied/inter s51200 `ckpt-10000` to 10,300 | final checkpoints (weights, both moments, step) and rows identical |
| | `c3-rev-taper-whole-s51201-a1`: S = 1,000, D = 2,000, from r1 tied/inter s51201 `ckpt-1000` to 2,600 with a mid-taper checkpoint at 1,800. Then `c3-rev-taper-cont-s51201-a1`, resumed from that `ckpt-1800` (its plan entry is added after the first run, because it needs that file's hash) | `ckpt-2600` and rows ≥ 1,800 identical. Logged weights equal mine at every row |

**Pin counterexample (CX-P1).** The resume recipe check binds a checkpoint to a config through the MANIFEST, when there is one, and through the checkpoint's run ID. My prediction: no accepted pairing in P4 has a checkpoint from one run and a config from another. If the per-step-readout source has no MANIFEST, its binding rests on the run ID alone. Then a config whose run ID matches but whose recipe differs is the remaining gap. That cannot happen with committed r1 files, so it would be a note, not a CONCERN.

Verdict rule: BLOCK if any of P1–P3, P5 or P9 fails in a way that changes what a branch or correction computes. CONCERN if a check fails without changing results, or a recipe pairing that should be refused is accepted. Otherwise PASS.

## 2. Correction lane (PR #63 at `c3f03cd`)

I have read its QUESTION.md (`3e3f73d` and the addenda through `95052f5`), ADAPTIVE.md and `corr_core.py`, and nothing else from the lane. I have not opened `accuracy.csv`, `fits.csv`, `states.csv`, `checks.json`, the figures, README or NOTE. Two results reached me before this entry: the coordinator's relay and ADAPTIVE both say that D beats A once on grouped held rows (per-step s51201, 0.625 against 0.562), and that per-step V2 reads every prefix's h_2 at 1.00.

**Checks** (script `rev_corr.py`, run as `c3-rev-corr-a1` through `run_jobs.py`). My forward pass and oracle come from the c2 review (`rev_pin.py`). The prefix sets, split, centroids, map training and measures are mine, written from the QUESTION text. The probe is c2's pinned `c2probe.fit_probe`, reviewed in c2.

| # | check | prediction |
|---|---|---|
| C1 | Bases. The six D4 files and the eight F2 files hash to RECOVERY.json or MANIFEST.json, and step 10,000. `params_sha256` of each loaded file equals the lane's before and after records | all equal |
| C2 | Sets. There are 128 prefixes, 16 per symbol. My grouped split, from the QUESTION rule (prefix index lex(op1, op2)·8 + s; within each symbol in order, one generator `default_rng(71300)`; cut 8/4/4), equals the lane's. Every prefix's 4 rows (3 training, 1 held) sit in one group, and the groups are disjoint. Eval = 32 held + 96 training rows | equal, and leak-free. The same is checked per F2 task |
| C3 | My recomputation of every arm × base × evaluation × row set in the lane's accuracy table (original and F2). A and E use my forward and my unique-prefix centroids. D uses `c2probe.fit_probe(H_fit, x_fit, 71301)` and my centroids. B and C are my own map code (128 → 64 → 128 ReLU, W1 He-normal then W2 LeCun-normal from `default_rng(71200)`, zero biases, Adam via `nets.adam_step`, lr 1e-3, 2,000 updates, MSE averaged over prefixes and units) | A, D and E exact. B and C within 1 row of each row set (identical if my draw order is theirs) |
| C4 | Also on my side: nearest-centroid symbol accuracy of the corrected state, centroid error, decoder accuracy, gain recovered (E − A ≥ 0.1 only), and train-row accuracy, against whatever of these the lane reports | equal, to rounding |
| C5 | Leakage, from the code. At evaluation B, C and D receive h_2 only; E alone uses the true symbol. Grouped centroids, map, decoder and calibration use fit (and val for calibration) prefixes only. The eval prefixes enter nothing that is fit or chosen | holds |
| C6 | Decoder-plus-centroid accounting. Per base and evaluation, is B ≤ D + 1/32 on held rows? And per row: of the rows where B's corrected state has the wrong nearest-centroid symbol, how many does B still answer correctly? | B ≤ D + 1/32 everywhere. B answers ≤ 1/8 of its wrong-symbol rows (chance), so its gain is the decoded symbol |

**Counterexamples** (committed before I open any lane result).
- **CX-C1, the map draw.** I refit B with three map draws of my own (73010, 73011, 73012), everything else fixed, and refit the decoder with probe seeds 73020–73022. The lane's per-base claim is the bin of gain recovered: below 20%, 20–60%, or above 60% (where E − A ≥ 0.1). Prediction: in both evaluations, that bin is unchanged across my three draws in at least 5 of the 6 original bases. If it changes in more, the single map and probe draws do not carry the lane's per-base statements.
- **CX-C2, the decoder.** This would weaken a negative grouped result. Two decoders that the lane did not try, both fit on fit prefixes only:
  1. nearest centroid (fit centroids);
  2. ridge regression to one-hot symbols, with λ chosen from {1e-3, 1e-2, 1e-1, 1, 10} on val prefixes only.

  Each is followed by c(x̂). Prediction: on grouped eval prefixes, neither raises decoder accuracy by more than 0.15 over the lane's probe, or held-row final accuracy by more than 0.1, in any tied/final base (original or F2). If one does, then "the adapter fails on new prefixes" is about the unregularized probe, not about what h_2 carries linearly.

## Revisions

### 05aac5a + runs (05:23–05:24), before any analysis of them

- The five runs in the plan finished. Before I compare any of their rows, P9 gains two checks:
  - In the drop branch, every parameter moves over 10,000–10,300, so no head is frozen. The Adam step count carries (10,300), and the first moments are nonzero. Prediction: true.
  - The taper run's row at 1,000 equals r1's published row, so the branch point is untouched. Prediction: true.
- P5 also checks that a per-step-readout model run at length 5 without a step map is refused.

### d3717f9, first pass of `rev_c3pin.py` (05:26–05:27)

Three things in the first pass need entries before the second pass.
- **P3 failed my own tolerance, at `ckpt-10000` only.** At `ckpt-1000` the finite difference agrees to ≤ 1.5e-7 relative in all 9 cases. At `ckpt-10000` the worst relative difference is 3.8e-2. There, the analytic directional derivative is only 2.6e-7 to 1.5e-6, because the state is near-stationary. A one-off scan of step sizes (1e-4 to 1e-7) shows no e² convergence there, which fits round-off and curvature dominating a near-zero derivative better than a gradient error. The P3 line stays in the output as it is.
  - **P3b, new.** The same check at `ckpt-10000` plus N(0, 0.01²) per element, 3 seeds × 3 weightings. Prediction: ≤ 1e-6 relative.
- **CX-P1, second line failed because my assumption about the file layout was wrong.** The per-step-readout runs have neither a `MANIFEST.json` nor their own `MANIFEST.sha256`. Their hashes are in the shared `research/regimes/review/runs/MANIFEST.sha256`, which `check_source` does not read. The check now tests that and that the recovered files match it. Prediction: true.
- **P5 crashed** on a key name of mine (`bv` does not exist in the untied-readout scheme). The changed element is now in `bs`.

### 44e7556 + script (before running `c3-rev-corr-a1`)

- **Files read since 44e7556.** The header lines of the lane's CSVs, `corr_eval.py`, `corr_calib.py` and the first data row of `figures/correction.csv`. That row is A main held s51200 = 0.633, which equals r1's published value.
- **C5, from the code.**
  - `corr_core.correction` passes h_2 alone to B, C and D, and gives E the true symbol.
  - `fit_arms` receives only the fit prefixes.
  - `corr_calib.py` uses grouped fit → val prefixes only, with init seed 900200, and computes no final answers.
  - Holds.
- **C7, added (no prediction beyond "equal").** Every value in `figures/correction.csv` and `figures/followup_f2.csv` recomputes from my numbers: A, D and E exactly, and B and C within one row.
- **Run.** `rev_corr.py` runs as `c3-rev-corr-a1` through `run_jobs.py` (plan entry in this commit).

### ae77579, first run `c3-rev-corr-a1` (failed with exit code 1 after 190 s; log kept as [out/rev_corr_pass1.txt](out/rev_corr_pass1.txt))

- **Complete before the crash, and recorded as they stand:**
  - C1–C4 pass in all 28 base × evaluation cells, with B and C exact to 0 rows.
  - C6 fails in part: "B ≤ D + 1/32" fails in 3 grouped cells, and "≤ 1/8 of wrong-symbol rows" fails in 8 grouped cells.
  - CX-C1 passes, 6/6 in every case.
  - CX-C2 fails: the largest rises are +0.156 in decoder accuracy and +0.156 in held final accuracy.
- **The crash was in C7, a bug of mine.** C7 read the `symbol_acc`/`centroid_err` panels, which have no split in their name, as the main evaluation. They are keyed by prefixes instead: `fit` means main and `eval` means grouped.
- **Rerun `c3-rev-corr-a2`.** The fix maps the split from the prefixes. Two other additions change no check:
  - the CX-C2 lines print the gain recovered;
  - all my accuracies are dumped to `rev_corr.json`.

  Prediction: C1–C6 and the counterexamples come out as in the first run (the code is deterministic), and C7 passes.

