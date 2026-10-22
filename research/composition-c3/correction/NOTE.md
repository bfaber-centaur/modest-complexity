# c3 learned correction: note and reviewer handoff

**Question (charter 02).** Can a small correction that sees only the frozen model's hidden state recover part of the oracle-centroid improvement on unseen programs?

**Short answer.**
- **States the correction was fit on:** yes. With a new op 3, the decoder arm D recovers the full gain. The map B recovers 75–103% of it on the original bases (75% in tied/final 51202) and 79–100% in F2.
- **Prefixes held out from the correction:** no, under this design. It recovers essentially none of the gain and breaks answers the model already got right.
- **Replication:** the same holds for eight bases from new task draws (F2). These are tied/final only, because no per-step-readout arm exists on those draws.

Full tables are in [README.md](README.md); the design is in [QUESTION.md](QUESTION.md) and [ADAPTIVE.md](ADAPTIVE.md).

## The overlap (D2)

**Main evaluation.** Every depth-2 state reached by a held-out program is a training state, and the main correction is fit on all 128. So the main evaluation tests seen states followed by new op-3 suffixes. Any decoder that fits the training states reproduces the oracle there, and D does exactly that.

**Grouped evaluation.** Prefixes are held out from the correction's fit (8 fit / 4 val / 4 eval per symbol). The base network was trained on all of them, so these are adapter-held-out prefixes, not globally unseen states.

## Observations

**Original bases** (per-step readout and tied/final, ckpt-10000, seeds 51200–51202).
- **Main, held rows.**
  - B: 1.00 (per-step) and 0.80–0.89 (tied/final).
  - D = E: 1.00 and 0.875–0.969.
  - A: 0.52–0.63 and 0.27–0.32.
  - C: below A in all six.
- **Grouped eval prefixes, held rows.**
  - E, using the 64 fit prefixes' centroids: 0.84–1.00.
  - B: never above A (−50% to 0% of the gain).
  - D: above A in one base only (per-step 51201: 0.625 vs 0.562, 2 of 32 rows).
  - C: below or equal to A.
- **Grouped eval prefixes, train rows.** Learned arms fall from A's 1.00 to 0.17–0.64.
- **Decoder accuracy on eval prefixes.** 0.53–0.66 (per-step) and 0.22–0.31 (tied/final), against 1.00 on fit prefixes.

**F2** (eight c2 generalization tied/final bases on four new task draws; recipe fixed, refit per base). This replicates tied/final only; no per-step-readout arm exists on these draws.
- **Main.** B recovers 79–100% of the gain, D = E, and C is below A.
- **Grouped eval prefixes.**
  - Every learned arm is below A on held rows in all eight bases. B ranges from −110% to −13% of the gain, D from −90% to −7%.
  - Train rows fall to 0.19–0.62.
  - The decoder reaches 0.19–0.44. E stays at 0.91–1.00.

**Checks.**
- 18/18 original reproductions are exact, and 8/8 F2 runs match their logged accuracy.
- Every base hash is unchanged after every fit.
- SHA-256 matches RECOVERY.json for the original bases and each run's MANIFEST.json for F2. The recipe checks passed.

**Post-hoc reference.** This was requested by the coordinator and involves no fitting. The per-step bases' own V2 reads the symbol of every prefix's h_2 at 1.00 (fit, val and eval), but V2 was trained on all 128 prefixes.

## Bound and interpretation

**What the grouped failure bounds.** Each grouped fit has 64 prefixes (8 per symbol) in a 128-dimensional state, with an unregularized linear decoder and a 64-unit map trained for 2,000 updates. So the failure bounds *this adapter under this exposure*. It does not show whether the symbol is linearly available in h_2.
- The base's own V2, trained on all 128 prefixes, separates them perfectly.
- With only 16 prefixes per symbol at depth 2, no grouped split can fit on many more.

**What the results support.**
1. A supervised post-hoc correction can map the frozen h_2 of a *seen* prefix to its centroid, and that rescues the new suffix. The rescue is exact for D, and 75–103% for B.
2. Under this design, a correction fit on half the prefixes does not transfer to the other half. For D, the step that fails is reading the symbol across prefixes, not replacing the state with a centroid: E uses the same fit centroids and still works. B is not purely a decoder plus a centroid. On grouped eval prefixes, B is above D on held rows in 3 of 14 bases: original tied/final 51201, and F2 `ops63000-ls63100` s63201 and `ops63000-ls63101` s63201. The reviewer also finds that B answers 13–33% of the rows whose state it decodes to a wrong symbol correctly (c3 REVIEW §2). B still stays below A.
3. Replacing a correctly handled state with a wrong symbol's centroid (D) or an off-centroid state (B) costs more than it gains. Training-program rows on new prefixes fall from 1.00 to 0.17–0.64.

**What they do not show.** The base model selecting centroids internally; the history residual being uniformly harmful; the main correction having learned no new computation (charter limits); or any necessity or impossibility claim.

## Predictions

| set | outcome |
|---|---|
| Main | held |
| Original grouped | **failed**: I predicted decoder ≥ 0.9 / 0.5–0.9 and B recovering 50–100% / 20–80%; neither happened |
| F2 | held, all predicted failures; written after the original grouped result, so these are not independent of it |

The original "B ≤ D + 0.05" prediction failed once (tied/final 51201, eval prefixes).

## Reviewer handoff

**Commits.**

| commit | content |
|---|---|
| 3e3f73d | QUESTION.md (M0) |
| 4b70c74 | code |
| 26c5590 | calibration choice |
| 6fe4d74 | first figure |
| 95052f5 | F2 declaration |

The F2 results, post-hoc reference and this note are in the following commit.

**Commands.** In README, Regenerate. The tests are `test_corr.py` (8).

**Suggested checks.**
1. `corr_core.correction`: B, C and D see only the h_2 matrix. E alone gets the true symbol, and it is labelled as the oracle.
2. Centroids are unique-prefix weighted, and equal c2's row-weighted centroids on the original task (test).
3. The grouped split is balanced, 8/4/4 per symbol, and disjoint (test). No eval prefix enters calibration: `corr_calib.py` uses only fit → val.
4. Recompute one base's grouped eval row from `accuracy.csv`.
5. The calibration choice follows the declared rule (`runs/c3-corr-calib-s900200-a1/checks.json`).
6. The F2 bases' MANIFEST hashes, and the A reproduction against `metrics.jsonl`.

**Known quirks.**
- The declared zero-bias ReLU output leaves 3–20 output units stuck at zero across fits (`fits.csv`).
- `correction.svg` metadata changes on every re-render; the PNG is byte-stable.

**Compute.** About 0.8 of 10 minutes initially, plus 0.3 of 5 minutes for the follow-up.

**Review.** c3 REVIEW §2 (PR #64) passed this lane: every number reproduced, the leakage checks held, CX-C1 held, and CX-C2 failed narrowly (a val-tuned ridge decoder recovers at most 6%, which is inside the bound above). Wording fixes W1–W3 and C6 were applied without new runs.
