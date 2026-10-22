# Composition follow-ups c3: synthesis

**Status.** DRAFT (correction section only; scaffolding pending).

**Scope of every claim.**
- **Task.** The original task and split (`ops51000-ls51100`), with three paired init seeds (51200–51202). The correction follow-up adds eight tied/final bases from four new task draws.
- **Recipe.** r1's: width 128, full-batch Adam at lr 1e-3 and eps 1e-8, decoupled decay 0.1, 384 training rows.
- **Budgets.** As stated per lane.
- **Coverage.** Three seeds are exploratory coverage, not statistical power. Branches from one checkpoint are not independent samples. No claim here is one of necessity, impossibility, a hidden algorithm or novelty.

Lane notes: [correction](../correction/NOTE.md), [scaffolding](../scaffolding/NOTE.md). Review: [review/REVIEW.md](../review/REVIEW.md). Decisions and the c2 evidence behind both questions: [DECISIONS.md](DECISIONS.md) (D2).

## 1. What an oracle-free correction achieved

**Setting.** The bases are the frozen 10k checkpoints of the two weaker c2 arms: per-step readout with intermediate labels, and tied/final.
- A correction is applied to h_2 only, followed by the unchanged step 3.
- Arms: identity (A); a small map trained toward the true symbol's training centroid (B); the same map trained to reconstruct h_2 (C); a linear decoder followed by the decoded symbol's centroid (D); and the true symbol's centroid, an oracle reference (E).
- At evaluation, B, C and D see only h_2.

**Two evaluations, and why both.** On the original split, every depth-2 state that a held-out program reaches is also a training state (D2).
- The *main* evaluation therefore tests seen states followed by a new op 3. Any decoder that fits the training states reproduces the oracle there.
- The *grouped* evaluation holds half the prefixes out of the correction's own fit (8 fit / 4 val / 4 eval per symbol). These are adapter-held-out prefixes. They are not globally unseen states, since the base network was trained on all of them.

| held-out-program rows | A (identity) | B (map → centroid) | D (decoder → centroid) | E (oracle reference) |
|---|---|---|---|---|
| main, per-step readout | 0.52–0.63 | 1.00 | 1.00 | 1.00 |
| main, tied/final | 0.27–0.32 | 0.80–0.89 | 0.875–0.969 (= E) | 0.875–0.969 |
| grouped eval prefixes, original bases | | never above A (−50% to 0% of the gain) | above A in 1 of 6 (2 of 32 rows) | 0.84–1.00 |
| grouped eval prefixes, F2 (8 new-draw tied/final bases) | | below A in 8 of 8 (−110% to −13%) | below A in 8 of 8 (−90% to −7%) | 0.91–1.00 |

- **On seen states the correction works.** D recovers the full gain, exactly, and B recovers 75–103% (79–100% in F2). C (reconstruction) stays below A.
- **On adapter-held-out prefixes it does not.** No learned arm recovers the gain. Every learned arm also breaks the training-program rows of those prefixes, which the base answers at 1.00: they fall to 0.17–0.64.
  - The step that fails is reading the symbol across prefixes. The decoder reads eval-prefix symbols at 0.53–0.66 (per-step) and 0.22–0.31 (tied/final), against 1.00 on fit prefixes.
  - The centroids are not the problem: E uses the same fit centroids and stays at 0.84–1.00.
  - B is not purely a decoder plus a centroid (reviewer C6), but it stays below A.
- **Robustness (reviewer).** The gain bins hold across three other map draws and three other probe draws (CX-C1). A validation-tuned ridge decoder recovers at most 6% of the grouped gain (CX-C2).
- **Bound.** Each grouped fit has 64 prefixes (8 per symbol) in a 128-dimensional state. So the failure bounds *this adapter under this exposure*, not whether the symbol is linearly available. The per-step bases' own V2, trained on all 128 prefixes, reads every prefix's symbol at 1.00 (a post-hoc reference, not an arm). With 16 prefixes per symbol at depth 2, no grouped split can fit on many more.

**Reading.**
- A supervised post-hoc correction can use the frozen representation to rescue new suffixes after seen states.
- On this task it did not generalize to prefixes outside its own fit, and replacing a state the base already handled cost more than it gained.
- This does not show that the base network selects centroids internally, or that the history residual is uniformly harmful. It does not show that the correction learned no new computation, or that no other adapter would transfer.

## 2. What temporary supervision left behind

*Pending the scaffolding lane.*

## 3. Limitations

*Pending.*

## 4. Next questions (at most two)

*Pending.*

## Runs and compute

*Pending.*
