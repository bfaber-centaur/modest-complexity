# c3 reviewer: verdicts

Questions and predictions, committed before each measurement: [QUESTION.md](QUESTION.md). Evidence: `rev_c3pin.py` and its output, [out/rev_c3pin.txt](out/rev_c3pin.txt). Disclosure: the correction lane's per-step-readout base (`r1-rev-comp-rnn-untied-*`) is my r1 arm.

| item | verdict | section |
|---|---|---|
| Pin `be41d16` (D3) | **PASS**. No defect in what a branch or a correction computes. Notes N1–N4. My finite-difference check P3 missed its own tolerance at `ckpt-10000` (see §1). | §1 |
| Correction lane | not started | |
| Scaffolding lane | not started | |

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
