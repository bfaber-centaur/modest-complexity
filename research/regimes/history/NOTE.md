# R3 training history and recovery: note (M4)

Design and pre-run predictions are in [QUESTION.md](QUESTION.md); commands, environment and the budget are in [README.md](README.md). Every number below is a range over the three paired init seeds (53200–53202). The numbers come from [figures/survival_summary.csv](figures/survival_summary.csv) and [figures/followup_head_reset.csv](figures/followup_head_reset.csv), which list run IDs.

## Question

After a small ReLU network learns A and then B:
- What survives of A?
- What does a return to A cost?
- Does it matter whether B shares structure with A?

## Design in one paragraph

**Setup.** The domain is 10-bit inputs, with 256 training and 768 held-out inputs. The tasks are:
- A = x0⊕x1⊕x2;
- related B = x0⊕x1⊕x3;
- unrelated B = x4⊕x5⊕x6;
- the conflicting mapping, notA through the same head, as a separate experiment.

**Network and training.** A 10→128 ReLU trunk with separate A and B heads, trained by full-batch Adam at lr 1e-3 with wd 0. The optimizer is reset at every phase boundary in every arm.

**Schedule.** A for 2000 updates. B for d ∈ {100, 1000, 10000}, branched from one hashed ckA per seed. Then A for 4000.

**Controls.**
- continued-A;
- d = 0;
- fresh-A, which is the same seed's phase A;
- scratch-B, a trunk trained on B alone, which is the history-free reference for a frozen-feature linear readout.

**Follow-up.** One adaptive follow-up, M3: the return phase with head A reset to its initial draw.

## Observations

**Primary figure:** [figures/recovery_curves.png](figures/recovery_curves.png). Supporting figures: [survival_summary.png](figures/survival_summary.png) and [followup_head_reset.png](figures/followup_head_reset.png).

1. **Forgetting tracks B acquisition, not elapsed updates.**
   - At d = 100, B is not yet learned (held-out 0.47–0.52), and held-out A stays at 0.988–0.999.
   - Once B is learned, held-out A through the bitwise-unchanged A head falls:
     - to 0.76–0.86 at d = 1000 and 0.69–0.80 at d = 10000 for related B;
     - to 0.76–0.83 and 0.71–0.77 for unrelated B.
   - In the curves the drop begins as B rises. Training accuracy on A drops less than held-out accuracy: 0.75–0.95 against 0.69–0.86 at the end of B.
   - The optimizer reset alone (the d = 0 arm) produced no visible transient: held-out A stayed ≥ 0.995.
   - Continued-A for the same updates keeps ≥ 0.995, although its trunk also moves substantially. Its relative drift is 0.20–0.31 at d = 1000 and 0.41–0.60 at d = 10000, against 0.50–0.58 and 0.76–0.94 under B. Drift size alone does not predict forgetting.

2. **A stays linearly decodable from the trunk after B.** A fresh linear readout on the frozen end-of-B trunk reaches 0.91–1.00 held-out at d ≥ 1000. Its comparison points:

   | readout trunk | held-out A |
   |---|---|
   | end-of-B trunk | 0.91–1.00 |
   | same B for the same d, no A history | 0.49–0.79 |
   | untrained trunk | 0.58–0.63 |
   | end of A | 0.99–1.00 |
   | shuffled-label null | 0.46–0.58 held-out (it fits 0.76–0.86 of the permuted training labels) |

   This is QUESTION.md outcome 2: A-relevant features remain, and the old head reads them out wrongly. A linear readout cannot say whether the network would use these features.

3. **Relearning is fast, and the follow-up splits the savings between trunk and head.** Return-phase updates to held-out A ≥ 0.95, against 540–700 for fresh-A from the same init. No arm was censored. The ranges cover d ∈ {1000, 10000}.

   | starting point for the return | related B | unrelated B |
   |---|---|---|
   | trained A head kept (M2) | 62–330 | 37–560 |
   | head A reset to its init draw, end-of-B trunk (M3) | 230–410 | 190–310 |
   | head A at init, trunk trained on B only (M3) | 460–640 | 740–1100 |
   | head A reset, end-of-A trunk (M3) | 110–180 | 110–180 |

   - The end-of-B trunk alone, with a head identical to fresh-A's, relearns 0.27–0.76× as fast as fresh-A. That is ≤ 0.5× in 8 of 12 seed–condition pairs.
   - It is also faster than the B-only trunk in every pair, by 50–910 updates. The smallest gap is related B, d = 10000, seed 53202 (410 against 460). So the savings are not carried only by the preserved head.
   - The kept head was faster than the reset head in 10 of 12 pairs. The two exceptions are both unrelated B in seed 53200 (300 vs 270 and 560 vs 310), where the old head slowed relearning.
   - These match the M3 predictions, except that "≤ 0.5× in most cases" held in 8 of 12 pairs.
   - **A side observation.** A trunk trained on unrelated B alone made A slower to learn than an untrained trunk (740–1100 against 540–700).

4. **Related versus unrelated B shows no consistent ordering.** Across seeds there is no ordering in end-of-B A accuracy or relearning cost. Seed 53200 forgot more under unrelated B; seeds 53201 and 53202 forgot more under related B. My 55/45 prediction that related B interferes at least as much is neither supported nor refuted at three seeds.

   - **Forward transfer.** Learning unrelated B after A was slower than learning it from scratch in all three seeds: 590–1400 updates against 370–480. Related B took 510–640 against 460–600.
   - **B after the return to A.** B was largely kept: held-out B ended at 0.76–0.94 after d = 1000 and 0.88–1.00 after d = 10000.

5. **Conflict (separate experiment).** notA through the shared head reached held-out notA ≥ 0.95 in 340–410 updates from ckA, and drove held-out A to 0.00–0.02.
   - The readout recovers A at 0.98–1.00. As stated in advance, this is uninformative here, because notA needs the same features.
   - The return to A took 190–440 updates after d ≥ 1000, and 29–54 after d = 100, where notA was only partly learned.

## Limitations

- **Coverage.** One data split, three init seeds, one width, one lr and one optimizer recipe. Full batch, wd 0, and all tasks are 3-bit parities. Branches from one ckA are not independent replicates.
- **Readout and criterion.** "Decodable by a linear readout" was measured with one readout recipe: standardized features, 2000 Adam updates at lr 1e-2. Relearning times come from a logging grid with resolution 1, 10 or 100 updates (1 below 100 steps, 10 below 1000).
- **No mechanism identified.** "Related" is defined by task structure, and nothing here shows that the network computes x0⊕x1 as a shared unit. No unit-level intervention was done, so which directions carry A through B is not identified.
- **The follow-up is adaptive.** It was chosen after seeing M2. The reset head is the init draw; a different fresh draw could shift the times somewhat.
- **What the measurements cannot say.** Neither failure to recover nor fast relearning establishes what was erased or kept. These measurements bound it from two sides only.

## Next question

Which trunk directions carry A through B? One option is to project the A readout's weights onto units and check, per unit, whether B rewired input weights (for related B, x2 to x3) or only re-scaled them. Another is to ablate or patch the units the readout relies on, and to test whether relearning with a reset head depends on them. A Fisher-weighted drift measure (as in EWC) would test whether forgetting is predicted by movement along A-important directions rather than by total drift.
