# e3 audit, third investigation: the mechanisms continuations (written 2026-10-22, about 00:40 UTC, before running any check below)

Session `e3-au-cloud-1`. Base `4b47de1` (`swarm/exploration`, with mechanisms packet 1 integrated at `d0a0657`). Budget used so far: about 0.7 of 30 min.

## Already known

- The nine start files and the continuation code passed the gate (investigation 2).
- I have listed the published files and read nothing else of packet 1. Per run, checkpoints are published at updates 0, 10, 30, 60, 100, 300, 1,000, 2,000, 5,000, 10,000, 15,000 and 20,000, and `metrics.jsonl` has a row every 10 updates. That row has train, cal and audit accuracy, which gives the run's own held-out set (cal ∪ audit, 662 pairs) every 10 updates; the 483 common pairs are a subset of it.
- The coordinator's message gives the headline: B rises from 0.43–0.58 to 0.74–0.86 on the common pairs at updates 70–100, then falls to chance by about 5,000. Meanwhile P(S over N) returns to A's level, about 0.94. The lane reads the fall as "erasure", a return toward the operand-label memorizer.
- From investigation 1: P(S over N) on these pairs is 0.94 for the trained 30% runs at 50k (A's starting point) and 0.93 for the no-decay 30% runs, against 0.71–0.73 for the closed-form memorizer. A value near 0.94 therefore marks a gradient-fitted network on this training set, not a particular network.
- My 300-update check run of arm A for init 30000 barely moved (train cross-entropy 3.651e-6 to 3.649e-6).

## Software checks

| # | Check | PASS if |
|---|---|---|
| K1 | Provenance and accounting | The registry has one running and one complete event for each of the nine IDs, with 20,000 updates. Each `config.json` records the start file's SHA-256, `init_age` 50,000 and the D5 recipe. Each step-0 checkpoint holds the start file's U, V, W with zero moments. The published files match their manifest hashes |
| K2 | Checkpoint predictions | My own forward pass on every published checkpoint of the nine runs gives the logged train, cal and audit accuracy exactly, and the logged cross-entropies within 1e-10 relative plus 1e-14 absolute (set for small, cancellation-limited values this time) |
| K3 | Dense replay around the peak | An independent replay (`independent.py`; zero moments, t = 1, 2, …) of the first 300 updates of each of the nine runs matches the published checkpoints at 10, 30, 60, 100 and 300 within 1e-9 × the largest weight. It gives the common-set accuracy at every update |
| K4 | Rounding sensitivity of the fall | For each B run, an independent replay of updates 1,000 → 2,000 from the published checkpoint at 1,000 with its moments (t = 1,001, …) is compared with the checkpoint at 2,000. Its common-set accuracy agrees within 1/483 |

## Questions and decision rules

1. **Is the rise robust?** It is "robust" if, in each of the three B runs, the replay's peak common-set accuracy in updates 1–300 exceeds its update-0 value by at least 0.10 and the published values reproduce (K2, K3). I report the peak's height and update, and how many logged rows and checkpoints fall within ±20 updates of it.
2. **Is the fall robust?** It is "robust" if, in all three B runs, common-set accuracy at 5,000 and 20,000 is at most 0.065 (twice the 1/31 chance level) by my own forward pass, and K4 passes. A's level (0.000–0.004 at update 0) is reported alongside, because "chance" and "A's level" differ.
3. **Edit or training?** Per source, for B and for C against A at the same update: the gain supplied by the edit is acc(B, 0) − acc(A, 0). The gain produced by training is the replay's peak minus acc(B, 0). The gain retained is acc(B, 20k) − acc(A, 20k). "Nothing retained" holds if the retained gain is at most 0.02 in all three sources.
4. **What does erasure return to?** I measure, at every published checkpoint:
   - argmax agreement of B and of C with A on the common pairs;
   - the same between A runs of different sources, the baseline for "a different network of the same kind";
   - weight distance ‖B − A‖ / ‖A‖;
   - P(S over N), P(a+b over N) and P(a+b over S) with `pairwise.py` and the e2 sets.

   Verdicts at 20,000 updates:
   - **"Returns to A"** if B agrees with A on at least 0.80 of the common pairs in all three sources and the weight distance is below half its update-0 value.
   - **"A different network of the same kind"** if agreement is at most the cross-source A–A agreement plus 0.10 and the distance stays above half its update-0 value.
   - **"Partial"** otherwise.

   P(S over N) back near 0.94 says only that the network is again a gradient-fitted one on this training set (investigation 1). That reading is SUPPORTED LOCALLY if B's and C's values at 20,000 are within the range of A's and the no-decay runs' (0.93–0.95).

## Expectations, stated before computing

- I expect K1–K3 to pass.
- I expect K4 to agree to far better than 1/483, since 1,000 updates is well before the 15,000–17,000 updates after which e1 saw rounding differences grow.
- I expect the rise to be robust, because it spans several checkpoints.
- For question 4, I expect "a different network of the same kind": a weight distance well above zero, and agreement with A no higher than between A runs. If so, "a return toward the memorizer" should be worded as a return to a fitted, operand-label-preferring network, not to the network that existed before the edit.

## Failure modes

- Three sources from one split and width. The evaluation pairs are shared, so pairs are not replicates.
- The replay uses the published start files. A bug common to `experiment.py` and `independent.py` would not show; investigation 2 limits that.
- Argmax agreement on networks near 0 accuracy measures agreement on wrong answers. That is what is wanted here, but it is sensitive to near-ties between the top wrong labels.
