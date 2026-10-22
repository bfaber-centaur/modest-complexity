# c3 learned correction: can a hidden-state-only correction recover the oracle-centroid gain?

- **Lane:** correction (thread "Hidden-state interchange").
- **Role file:** [charter/02-learned-correction.txt](../coordinator/charter/02-learned-correction.txt).
- **Assignment:** DECISIONS D4 and D6. Run IDs are `c3-corr-`; seeds come from 71000–71999.
- **Order:** written before any c3 result in this lane. Adaptive choices made later go in [ADAPTIVE.md](ADAPTIVE.md), and the single follow-up is appended below when proposed.

## Question

c2 replaced the depth-2 state with the training class mean of its *true* symbol, then applied the unchanged op 3. That fixed most held-out failures of the weaker RNNs (c2 mechanisms O4/O5; DECISIONS D2):

| arm | class mean | natural state |
|---|---|---|
| per-step readout | 1.00 | 0.52–0.63 |
| tied/final | 0.88 / 0.88 / 0.97 | 0.27–0.32 |

That intervention used an oracle symbol. Here the question is whether a small map whose only input at inference is h_2 recovers part of that gain.

**What c2 already says about this.**
- **Scale versus symbol.** Reviewer CX-M1 found that in these two arms, at (2, 2), nearest-centroid donors recover 10–59% of the class-mean gain. At (3, 3) they recover 64–86%. Part of the gain is therefore the right symbol; part is not.
- **Training time.** Both arms improve with training (c2 time), so a 10k correction is a finite-time rescue.

## The overlap, stated first (D2)

**Main evaluation: seen states, new suffixes.** Every one of the 128 depth-2 states that a length-3 program reaches is a training state. Each (start, op1, op2) prefix appears with 3 training suffixes and 1 held-out suffix, and h_2 depends only on the prefix. So the main evaluation applies the correction to states it was fit on, then runs a new op 3 (a new suffix combination). It is *not* a test of the correction generalizing to unseen hidden states. Any decoder that fits the training h_2 exactly reproduces the oracle arm there.

**Grouped evaluation: adapter-held-out prefixes.** The grouped-prefix split below holds out prefixes from the correction's own fit. The frozen base network was still trained on those prefixes, so they are adapter-held-out prefixes, not globally unseen states. The base network's split is unchanged.

## Bases (frozen; no base training)

`ckpt-10000` of six runs:
- `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`: per-step readout, intermediate supervision;
- `r1-comp-rnn-final-s5120{0,1,2}-a1`: tied/tied, final-only.

The init seeds are paired, not new. Before use:
- each file's SHA-256 must equal c2 `recovered/RECOVERY.json`;
- checkpoint meta step and Adam step must be 10,000;
- the recipe, read from r1's `config.json` with `c3train.recipe_of`, must match. Required: task `ops51000-ls51100`, transition tied, readout untied / tied, width 128, lr 1e-3, wd 0.1, Adam (0.9, 0.999, 1e-8), step weights 1 1 1 / 0 0 1.

The parameters go through `c3correct.frozen`. `c3correct.params_sha256` is recorded before and after every fit and must not change.

**Correction point.** One correction at depth 2 via `c3correct.corrected_forward(p, s, ops, depth=2, correct)`. The unchanged step 3 follows, with the unchanged readout (`V3` for per-step).

## Sets (fixed before any fitting)

The unit is the unique depth-2 prefix (s, op1, op2): 128 prefixes, 16 per depth-2 symbol x.

- **Main.**
  - *Fit:* all 128 prefixes, each weighted once.
  - *Centroids:* c(x) is the unweighted mean of h_2 over the 16 prefixes with symbol x. Each prefix appears in exactly 3 training rows, so this equals c2's row-weighted centroid; it is stated here so the weighting is explicit.
  - *Evaluation:* the 128 held-out-program rows (seen prefix, new op 3). Training performance is the 384 training-program rows.
- **Grouped.** Within each symbol x, the 16 prefixes are permuted with seed 71300 and split 8 / 4 / 4:
  - **fit** (64 prefixes) for centroids, map and decoder;
  - **val** (32) for calibration only;
  - **eval** (32), never used to choose anything.

  Evaluation covers the eval prefixes' 32 held-out-program rows (base: new suffix) and their 96 training-program rows (base: seen row, adapter: new prefix). The same split is used for all six bases.

## Arms (the same in both evaluations)

| arm | correction applied to h_2 | trained on |
|---|---|---|
| A | identity (no correction) | — |
| B | map g(h_2), trained by squared error toward c(x) of the prefix's true symbol | fit prefixes: (h_2, c(x)) |
| C | the same architecture, initial draw and update exposure as B, trained to reconstruct h_2. It controls for the module and its training, but its target is not matched to B's in complexity | fit prefixes: (h_2, h_2) |
| D | linear softmax decoder h_2 → x̂ (`c2probe.fit_probe` defaults: Adam lr 1e-2, 1,000 full-batch updates; probe seed 71301), then c(x̂) | fit prefixes: (h_2, x) |
| E | c(x) of the true symbol: the oracle reference, not a guaranteed upper bound | fit centroids |

- **At evaluation,** B, C and D see only h_2. No symbol, op, suffix or answer enters them; `corrected_forward` passes h_2 alone.
- **No final-answer loss.** Nothing is trained through the final-answer loss.

**Map.** g(h) = relu(relu(h W1 + b1) W2 + b2): 128 → k → 128, ReLU bottleneck, non-negative output.
- Init (seed 71200): W1 He-normal, W2 LeCun-normal, zero biases. The same draw is used for B and C and for all six bases.
- Training: full-batch Adam (0.9, 0.999, 1e-8), no decay, 2,000 updates, mean squared error averaged over prefixes and units.

**Calibration** (`c3-corr-calib-s900200-a1`, before any evaluation; D7 calibration seed, D8 host timing).
- Map init seed 900200 for every grid cell, so the evaluated draw (71200) is never the one tuned on. The run also reports ms per map update on this host.
- Grid: k ∈ {16, 32, 64} × lr ∈ {1e-3, 1e-2}.
- Data: the grouped fit → val prefixes, all six bases.
- Criterion: arm B's validation squared error to c(x), averaged over bases. The smallest wins; on a tie (within 1%), the smaller k, then the smaller lr.
- Nothing about final answers or eval prefixes is computed in calibration. C's validation error is logged, not used.
- The chosen (k, lr) is recorded in ADAPTIVE.md and declared in a dated addendum at the end of this file (D6). It is then fixed for every arm, base and evaluation.

## Measures

For each base, arm and evaluation:
1. **Final accuracy** on held-out-program rows and on training-program rows.
2. **Symbol of the corrected state:** nearest-centroid symbol of the corrected h_2 (centroids from the same fit set), as accuracy against the true x.
3. **Centroid error:** ||ĥ − c(x)||² over the mean squared distance between centroids.
4. **For D:** decoder accuracy.
5. **Gain recovered:** (arm − A) / (E − A) on held-out-program rows. Reported only where E − A ≥ 0.1.
6. **Map fit:** training squared error for B and C.
7. **Base hashes** before and after each fit.

## Predictions

1. **Checks.** Base hashes are unchanged. A reproduces r1's published held-out accuracy: per-step 0.633 / 0.523 / 0.516, tied/final 0.273 / 0.312 / 0.320 (c2 `c2-mech-residual-a1`, dose λ = 1). E reproduces c2's dose λ = 0: per-step 1.00 on held-out and training rows, tied/final 0.875 / 0.875 / 0.969 on both.
2. **Main evaluation (seen states).**
   - D's decoder fits training h_2 at 1.00 in both arms, so D = E exactly.
   - B recovers ≥ 80% of the gain in per-step readout and ≥ 60% in tied/final. tied/final's h_2 has a larger within-symbol spread (c2 geometry: 6.4–7.6, against 3.2–3.6 in per-step readout), and its shared readout reads h_2 near chance (0.09–0.15 on training rows), although a linear probe fits it at 1.00 (c2 probe).
   - C recovers ≤ 30%. A k-dimensional reconstruction may still remove some history variation, which is why C is reported.
3. **Grouped evaluation, held-out-program rows of eval prefixes.**
   - D's decoder generalizes to the new prefixes at ≥ 0.9 (per-step) and 0.5–0.9 (tied/final).
   - B recovers 50–100% (per-step) and 20–80% (tied/final) of the gain.
   - Training-program rows of eval prefixes stay ≥ 0.9 under B and D in per-step readout. In tied/final they may drop below A: c2 found that the class mean hurt some tied/final training rows (0.88–0.97 against 1.00).
4. **Ordering.** B ≤ D + 0.05 in both evaluations: B behaves like a decoder plus re-encoder and does no better.
5. **Not predicted:** whether B beats D on tied/final grouped prefixes, which would mean the map uses more than the symbol.

## First figure (plan)

`figures/correction.png` + `.svg` + `.csv`, 2 rows (per-step readout, tied/final) × 3 columns:
1. **Main evaluation:** accuracy on held-out-program rows by arm A–E, one marker per seed, with training-program rows as hollow markers.
2. **Grouped evaluation:** eval-prefix held-out-program rows (filled) and training-program rows (hollow) by arm. Fit-prefix rows go in the CSV.
3. **Symbol and centroid error** of the corrected state by arm, for both evaluations.

Chance (1/8) is marked, with E labelled as the oracle reference.

## Run IDs, seeds, budget

| run | what |
|---|---|
| `c3-corr-calib-s900200-a1` | calibration (grouped fit → val only) and host timing |
| `c3-corr-main-a1` | arms A–E, main evaluation |
| `c3-corr-grouped-a1` | arms A–E, grouped evaluation |

- **Seeds:** map 71200, split 71300, probe 71301; calibration map init 900200.
- **Budget:** 10 min initial and 5 min follow-up (D8), elapsed with a job running, via `run_jobs.py`. Each map fit is 2,000 updates on at most 128 rows, so the initial work should take a few minutes.
- **Gate:** no fitting before the reviewer's PASS on pin `be41d16`, relayed by the coordinator.

## Follow-up candidates (one, chosen after the first figure with the coordinator)

The application rule is declared now. The B and D modules fit at depth 2 in the main evaluation are applied unchanged, with no refit or tuning:
- **F1, another depth.** At depth 3 of length-4 to length-6 programs: the correction at h_3, then 1–3 more ops, with `V3` read in the recipient frame for per-step as in c2. Also at depth 1 of length-3 programs. Every program of each length is scored.
- **F2, a second task draw.** On c2 generalization's tied/final 10k checkpoints for a new task draw, where the c2 reviewer passed that lane. Modules are refit there with the same recipe and split rule, because a depth-2 map does not transfer across tasks. Only tied/final exists there; there is no per-step readout arm.

Failure in either case bounds portability; it does not undo the depth-2 result.

## Interpretation limits (fixed now)

- **What a gain shows.** That a supervised post-hoc correction can use the frozen representation, under this exposure. It does not show that the base network selects centroids internally, that the history residual is uniformly harmful, or that the correction learned no new computation.
- **Main evaluation.** Seen states only.
- **Grouped evaluation.** Adapter-held-out prefixes of states the base network was trained on.

## Literature

Lookup-table composition: Liška, Kruszewski and Baroni (2018), arXiv:1802.06467. This design is not a reproduction of theirs, and no novelty claim is made.

## Addenda (dated, appended after calibration or follow-up approval)

**2026-10-22 05:32Z: map dimensions and optimizer fixed (D6).** Calibration `c3-corr-calib-s900200-a1` chose **k = 64, lr = 1e-3** by the declared rule. The mean arm-B validation errors were 0.254 for that cell against 0.258 for the next best (k = 32, lr = 1e-2), which is outside the 1% tie band. Every other part of the map is as declared: 128 → 64 → 128, ReLU, full-batch Adam (0.9, 0.999, 1e-8), no decay, 2,000 updates, init seed 71200. Calibration also logged output units that stay at zero while their target is positive somewhere: 3–10 for B and 8–17 for C. This was not used for selection, and the evaluation runs report the same count in `fits.csv`. Map updates took 0.25 ms on this host.

**2026-10-22 05:40Z: follow-up F2, declared before running.** The coordinator approved F2 over F1 at 05:37Z. Their reason: c2 O4 already shows off-diagonal class means at chance, so F1 would mostly confirm a predicted failure.
- **Bases.** The eight c2 generalization tied/final `ckpt-10000` runs `c2-gen-final-{task}-s{63200,63201}-a1`, for tasks `ops63000-ls63100`, `ops63000-ls63101`, `ops63001-ls63102` and `ops63001-ls63103` (two op draws, four splits).
  - SHA-256 is checked against each run's MANIFEST.json, and the recipe through `c3train.check_source`: the task from its config, tied/tied, width 128, lr 1e-3, wd 0.1, r1's Adam constants, weights 0 0 1.
  - Hashes are recorded before and after each fit.
  - All eight already pass these checks (`test_corr.py`).
  - Only tied/final exists on these draws.
- **Recipe, fixed and not recalibrated.** k = 64, lr = 1e-3, 2,000 updates, map seed 71200, probe seed 71301. Centroids are unique-prefix weighted.
- **Sets.**
  - Main: all 128 prefixes of each task, evaluated on that task's own held-out programs.
  - Grouped: the same rule (8/4/4 per symbol, symbols in order, one generator) with the same seed 71300, applied to each task's own prefix symbols. Every task has 16 prefixes per symbol (tested).
- **Runs.** `c3-corr-f2-main-a1` and `c3-corr-f2-grouped-a1` (`corr_eval.py --bases f2`), within the 5-minute follow-up budget.
- **Predictions (made before running).**
  1. A's held accuracy equals each run's logged step-10000 value, and base hashes are unchanged.
  2. Main: D = E in all eight. B recovers ≥ 60% of E − A in at least 6 of 8 bases, where E − A ≥ 0.1.
  3. Grouped eval prefixes:
     - E − A ≥ 0.1 on held rows in at least 6 of 8;
     - B and D each recover < 20% of the gain in at least 6 of those;
     - decoder accuracy ≤ 0.5 in at least 6 of 8;
     - B and D lower training-program rows below 0.9 in all 8.
- **Bound, fixed now.** Each grouped fit has 64 prefixes (8 per symbol) in a 128-dimensional state, with an unregularized decoder and map. A failure bounds this adapter under this exposure. It does not show whether the symbol is linearly available in h_2.

