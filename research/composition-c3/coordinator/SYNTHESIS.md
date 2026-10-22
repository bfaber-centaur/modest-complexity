# Composition follow-ups c3: synthesis

**Status.** DRAFT. Sections 1 and 2 are written; the scaffolding follow-up, its review, limitations, next questions and the runs table are pending.

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

**Setting.** The tied/tied RNN on the original task, with three init seeds (51200–51202).
- Intermediate labels (loss weights [1, 1, 1]) are used until a switch S, then final labels only ([0, 0, 1]). The switch is either immediate (drop, B) or a linear taper over 2,000 updates (taper, C).
- References, both run to 50k: keep [1, 1, 1] throughout (A), and final-only from init (D). Both are the existing r1 runs plus c2's continuations.
- S = 1,000: every seed is partial (held-out length 3 at 0.34–0.36, ¬N at most 0.02).
- S = 10,000: length 3 is at or near 1.0, but length transfer is still forming (L6 at 0.56–0.86).
- Measures are logged every 50 updates. Frozen lengths 1–6 are evaluated at fixed checkpoints only: S, S + 2,000, 10k, and 20k–50k.

| at 50,000 updates | held-out length 3 | ¬N | L4–L6 |
|---|---|---|---|
| A keep | 1.00 | 1.00 | 1.00 |
| B drop and C taper, both S, all seeds (12 branches) | 1.00 | 1.00 | 0.99–1.00 |
| D final-only from init | 0.56–0.70 | 0.12–0.26 | 0.10–0.15 |

- **At 50k, removing the labels lost nothing measured.** All 84 endpoint measures (12 branches × held-out L3, ¬N, L1, L2, L4, L5, L6) are within 5 pp of keep. Every branch ends 0.30–0.44 above D on held-out length 3 and 0.85–0.90 above it on L4–L6.
- **Even 1,000 updates of intermediate labels made a lasting difference.** After S = 1k, drop and taper (1,000 full updates, then a 2,000-update taper) reach keep's 50k endpoint in all three seeds, including lengths 4–6. D does not within 50k. In c2, D reached 0.90–0.96 on held-out length 3 by 100k with lengths 4–6 at 0.06–0.14 (D2).
- **The cost of removing them was time.** At S = 1k, held-out length 3 first holds ≥ 0.95 at 14,600–20,450 updates (drop) and 12,050–21,400 (taper), against keep's 7,700–9,150. The largest gaps fall at 10k–20k: drop s51202 at 10k is −0.39 on held-out and −0.86 on ¬N. At S = 10k, L6 lags keep by up to 0.34 at 20k and closes by 40k–50k.
- **Previously supervised states stayed readable without their labels.** The shared readout on h1 and h2 stays at or above 0.92 on training rows after the switch (1.0 throughout after S = 10k). Frozen L1–L2 is ≥ 0.95 at every fixed time and 1.0 at 50k. D's readout, never trained on h1 or h2, reads them near chance.
- **Reviewer counterexamples.** CX-S1 tested whether length-3 retention at S = 10k could hide a loss of L6 or of the h1/h2 readout; it failed (L6 within 0.0023 of keep, readout at 1.0), so persistence stands. CX-S2 held for the total parameter norm: after S = 10k it ends 4.1–4.3% below keep. But the shrinkage is in the start embedding (Ws at 0.68 of keep's norm, bs at 0.71–0.74) and the readout bias (0.55–0.78), not the readout V (0.94–0.95), and the shared transition A grows 4–5%. This is a post-hoc reviewer observation, not a mechanism.
- **Taper and drop end the same.** At S = 1k the taper crosses earlier in two seeds and later in one.
- **Optimizer.** The update norm falls after the switch. The eps regime begins at about 29k in every branch, as in keep and D.
- **Three of the lane's six predictions failed.** It expected forgetting of the step-1/2 readout and of length transfer after S = 10k, and a stall short of keep after S = 1k. None happened by 50k.
- **Checks.** Every branch joins r1 tied/inter exactly at S. The logged schedules and exposures match the declared ones. The frozen evaluation reproduces c2's published lengths table for keep and D. The reviewer recomputed every value from its own forward pass and passed the lane (REVIEW §3).

**Attribution (pending the follow-up).** A drop changes the loss, but the Adam moments carried from three-head training remain for about a thousand updates (b2 = 0.999). The follow-up resets the moments at S = 1k in keep and in drop, with matched carried controls (6 runs). It asks whether the post-switch acquisition comes from the state at S or partly from the carried moments.

**Reading.**
- On this task and recipe, 1,000 updates of intermediate supervision were enough to put tied/tied training on keep's 50k endpoint. Removing the labels at 1k or 10k slowed acquisition but did not undo it, and nothing was forgotten by 50k.
- This does not show that the labels are necessary: D is one tested baseline at a finite horizon, not a bound. It does not say what the trained network computes, or that the result holds past 50k, on other tasks, or under other recipes.
- The equal endpoints hide different paths. At S = 1k the lag behind keep lasts 5,000–12,000 updates.

## 3. Limitations

*Pending.*

## 4. Next questions (at most two)

*Pending.*

## Runs and compute

*Pending.*
