# r1 independent review

Reviewer's plan and pre-stated counterexamples: [QUESTION.md](QUESTION.md). Status values are PASS, CONCERN, BLOCK and NOT YET CHECKED. A finding blocks only what it names.

| section | status | last updated |
|---|---|---|
| §0 shared code | PASS, with three usage concerns (C0.1–C0.3) | 2026-10-22, common at `0c7bd19` |
| §1 R1 composition | M1 oracle/split PASS; design notes; M2 not yet | 2026-10-22, lane `ff9b721` |
| §2 R2 shared subcomputation | **M4: no BLOCK; mergeable.** Counterexamples: z not privileged in y1 features (C2.4); the gate-bit explanation of fine-tuned transfer is refuted by a gate-only arm (C2.5). Two NOTE edits suggested (C2.5, C2.6) | 2026-10-22, lane `f493c1a` |
| §3 R3 history and recovery | M1 oracle/split/metrics/branch hashes PASS; C0.1/C0.2 avoided | 2026-10-22, lane `f71c84f` |
| §4 R4 competing routes | M1 PASS; M2 numbers recomputed PASS; counterexample: route choice at ρ = 15/16 is mostly base-specific, not context-level (C4.1, wording) | 2026-10-22, lane `f0f68f7` |

## §0 Shared code (`research/regimes/common/`)

Reviewed at swarm/regimes `e71ad8c`, where `common/` is unchanged since `0c7bd19`. SHA-256: `nets.py` `1a1e3ef2…`, `run_jobs.py` `dfe8540c…`. The evidence is [check_common.py](check_common.py) and its output [out/check_common.txt](out/check_common.txt). These are code checks, not research runs, and take 3.5 s wall. The shared tests were not used as evidence.

| item | status | evidence |
|---|---|---|
| Loss value | PASS | Independent rewrite (explicit layers, row masks, log-softmax) equals `loss_grad`'s total to 1e-12 on 3 hidden layers × 3 heads, with head weights 1 and 0.37. |
| Gradients, multi-head, multi-layer, unlabeled rows | PASS | All 197 parameter entries match central differences of the rewrite, not of `nets.py`. The worst relative error is 2.1e-6. Preactivations were held ≥ 2e-3 from the ReLU kink. A fully unlabeled head gets exact zero gradient and no per-head loss. A row unlabeled for every head has no effect on any gradient. A no-hidden-layer net also works. |
| Adam, decoupled decay, frozen names | PASS | A schedule of all-trainable ×3, then trunk + A head frozen ×4, then trunk unfrozen ×3, with wd 0.2, matches a separately written loop to 1.1e-16. Frozen names get no update, no decay and untouched moments, bitwise. |
| `adam_state` is a full reset | PASS | t = 0, all moments zero, no aliasing of the weights or the old state. Training from a reset continues bitwise like a state built on copied weights. |
| Checkpoint round-trip | PASS | Weights, both moments, t, dtypes and nested metadata round-trip. A loaded state continues bitwise for 5 steps with decay. Identical contents saved 2 s apart give the same file SHA-256, because Python 3.12's `ZipFile.open(name, 'w')` writes a fixed 1980 date. On this environment a checkpoint file hash therefore identifies its contents. |
| Runner capped/failed semantics | PASS | On a throwaway plan: exit 0 gives `complete`; exit 3 gives `failed` with returncode 3; a timeout gives `capped` with returncode null and keeps its log. Each run has exactly one `running` and one terminal event. Reused IDs (by registry or by existing directory) and unknown IDs are refused before anything is appended. |

**Usage concerns.** None is a defect in the code as documented. Each affects only lanes that use the pattern. I check them per lane in §1–§4.

- **C0.1 A head left out of `targets` is not frozen.** If it stays in `trainable` (the default), it keeps moving on carried Adam momentum, and it decays whenever wd > 0.
  - Measured: after 5 joint steps, 5 steps on B alone with the optimizer carried and wd 0 moved `W_A` by up to 3.4e-2. After an `adam_state` reset with wd 0 it stayed bitwise fixed.
  - *Affects:* any claim that B training "never touches the A head" (R3). That holds only with a reset and wd = 0, or with the A head excluded from `trainable`. It also affects any optimizer-carried follow-up (R3), where the A head moves through momentum, not through B's loss.
- **C0.2 Freeze, then unfreeze, without a reset.** The step count t is shared, so bias correction is about 1 while a newly unfrozen name's moments start at zero.
  - Measured: with constant gradient, unfreezing at t = 1000 gives a first step of 2.5 × lr and up to 5.2 × lr within 20 steps. A fresh state gives 1.0 × lr.
  - *Affects:* any frozen-readout-then-fine-tune or frozen-trunk-then-unfreeze arm that keeps the same state (R2 frozen-feature comparisons, R3 readout recovery). The remedy is `adam_state` at the unfreeze, stated in the config.
- **C0.3 Relative `--root` from outside the repository root.** The runner checks reuse and writes logs relative to the caller's cwd. The job runs with cwd = repository root, so `{out}` resolves there instead.
  - Shown: invoked from a scratch directory with `--root rr`, the log went to `./rr/x.log` and the run directory to `<repo>/rr/x`. The reuse check then cannot see the run directory.
  - *Affects:* only invocations outside the repository root; PLAN §3's command runs from the root.
  - Proposed one-line fix for the coordinator: `args.root = args.root.resolve()` after parsing. Low priority.

**Notes for interpretation, not findings.**
- Per-head loss is the mean over that head's labeled rows. A head labeled on fewer rows therefore carries more weight per labeled example. Exposure accounting for intermediate-state heads (R1) and partially labeled heads (R2, R3) should state the effective per-example weight.
- If the runner process itself dies (container reclaimed), a run keeps a `running` event with no terminal event. Lane manifests should list such runs as interrupted, not drop them.

## §1 R1 composing small programs

Reviewed at lane `ff9b721`: QUESTION.md and task.py, no training code or runs yet. The evidence is [oracle_composition.py](oracle_composition.py), with output in [out/oracle_composition.txt](out/oracle_composition.txt). The tables and the Latin square were copied from QUESTION.md as data, and every property was recomputed before the lane's arrays were compared.

| item | status | evidence and what it affects |
|---|---|---|
| Oracle and split | PASS | `draw_ops()` equals the published tables, and the first seed was accepted. The square is Latin. The 48/16 programs match. The lane enumerates exactly 576 Latin squares. All 384 + 128 rows match a rewrite for step1, step2, final and the 20-unit slot one-hot. |
| Equivalence and leakage | PASS | Cycle types, fixed points and orders are as the table states. No op is the identity or an involution, and no pair commutes. The 16 length-2 and 64 length-3 products are distinct. No held-out program equals any program of length 1–3 seen in training. |
| Position coverage | PASS | Every op appears 12× at each position, and every ordered pair 3× at slot pairs (1,2), (2,3) and (1,3). All 32 transitions occur at each position. All 128 (s,a,b) prefixes and all 128 (step1,b,c) contexts occur. The held-out repetition pattern is 6 distinct, 3 each of a=b, b=c and a=c, and 1 all-same. |
| Baselines | PASS | All ten reproduced to 3 decimals by separate code. Identity 0.102. Single-op 0.0625/0.102/0.0625. Drop-one 0.125 each. Best proper-subset lookup 0.125. 9-neighbour vote 0.077. Nearest-program oracle 0.453. |
| Loss and metrics | NOT YET CHECKED | No training script yet. |
| Exposure, interventions, checkpoints, figures | NOT YET CHECKED | |

**Answer to QUESTION.md's open item.** The 48 training programs' *final* tables alone determine the four permutations uniquely.
- Method: the first slot fixes P_k = P_0 ∘ D_k for known D_k. Then every one of the 8! choices of P_0 was tested against all 48 tables. Exactly one tuple survives, and it is P.
- So `final` lacks no information about the primitives in principle. Any `inter` advantage is about the route or representation the extra labels induce, not about missing information.

**Design notes (not findings).**
- Held-out step1/step2 accuracy in `inter` reads prefix labels on prefixes that were seen. Step1 depends only on (s,a), and step2 only on (s,a,b), and all of these occur in training with the same label. A high held-out step-k accuracy therefore measures invariance to the unseen later slots, not composition. QUESTION.md's outcome 5 already reads it this way.
- C0.1/C0.2 do not arise if every head present is trained in every step, which is what QUESTION.md describes. I recheck this against `train.py`.

## §2 R2 shared subcomputations

Reviewed at lane `c4939c2`: QUESTION.md, task.py and train.py, with smoke/calibration records only. The evidence is [oracle_shared.py](oracle_shared.py), with output in [out/oracle_shared.txt](out/oracle_shared.txt).

| item | status | evidence and what it affects |
|---|---|---|
| Oracle | PASS | X row i is the binary expansion of i. y1, y2, part and unrel equal integer-bit rewrites, with 512/256/512/512 positives. On the x4 = 1 half, y2 = z exactly and is balanced (256/512). An x4-only predictor scores 0.75 accuracy and 0.833 balanced accuracy, as stated. |
| Split | PASS | For each data seed 52100–52102, pool and test are the two halves of `default_rng(seed).permutation(1024)`. They are disjoint and cover the domain. D_n are prefixes. The lane's (x1,x2,x4) coverage per n equals a recount. Phase 1 trains on the pool only (`train.py`). |
| Metric arithmetic | PASS | `head_metrics` accuracy, balanced accuracy and cross-entropy equal a per-row loop on random logits. |
| Optimizer and freezing (C0.1, C0.2) | PASS | `train.py` excludes `pre` from `trainable` in phase 2, so it is truly frozen. Adam is reset at each phase. The readout trains a copied head with its own fresh state on fixed features. |
| Exposure, interventions, figures | NOT YET CHECKED | |

**C2.1 (CONCERN, small n): coverage and positives differ across replicates.**
- At n = 8 the three data seeds cover 4, 6 and 8 of the 8 (x1,x2,x4) patterns. At n = 16 they cover 7, 8 and 8.
- D_8 holds only 3–4 rows with x4 = 1.
- Replicate differences at n ≤ 16 are therefore partly coverage differences. The "visible" rule (three same-sign paired differences) is applied across replicates that saw different patterns. A pattern absent from D_n bounds every arm alike.
- *Affects:* any claim at n ≤ 16. It does not affect n ≥ 32, where all seeds cover all patterns.

**C2.2 (CONCERN, interpretation): `rel` and `part` differ in more than "which pair is shared".**
- `rel` = x1⊕x2⊕x3 shares the z pair {x1, x2} with y2.
- `part` = x2⊕x3⊕x4 shares {x2, x4}, and x4 is y2's gate bit, not a z input.
- So `rel` > `part` would be consistent with z reuse, but also with pretraining on the gate bit being unhelpful or harmful for a gated target. `part` ≥ `rel` would not rule out z reuse.
- *Affects:* the reading of outcome 3 versus outcome 4. The mechanism measurements (z-direction intervention, split by x4) are what separate these, not transfer curves.

**Baselines on the actual test sets.**
- The primary metric's chance line is 0.5. On the test x4 = 1 rows (250–258 per seed) the majority class scores 0.504–0.520.
- The best z-blind lookup on any one bit or any pair other than (x1, x2), fitted on the pool, scores 0.480–0.496.
- So 0.5 is right to within about 0.02.

### M2 (`55d6787`, first figure)

Evidence: [recompute_shared.py](recompute_shared.py), output in [out/recompute_shared.txt](out/recompute_shared.txt), and the lane's `figures/final.csv`.

| item | status | evidence and what it affects |
|---|---|---|
| Key numbers recomputed | PASS | The frozen-feature readout was rebuilt here from each `rel` replicate's published `pretrained.npz`: init `new` head copy, Adam lr 3e-3, 2,000 updates on unstandardized last-layer features. It equals the lane-logged test x4 = 1 accuracy at n = 32, 64, 128 and 512 exactly (max difference 0). The `new` head in `pretrained.npz` equals its initialization, so phase 1 did not touch it. |
| Exposure across controls | PASS (design) | rel, part, unrel and same all get 2,000 phase-1 updates on the 512-row pool, followed by an Adam reset and 2,000 phase-2 updates on D_n. The readout uses the same updates and lr on fixed features. |
| Figure supports text | PASS, see C2.3 | Inspected `transfer.png`. Every replicate is shown, along with train accuracy, the majority and chance lines, and the missing/capped status ("none"). |

**C2.3 (CONCERN): fine-tuning at small n degrades pretrained features, including for the same-task positive control.** This answers the coordinator's question.
- In `final.csv` at n = 32, fine-tuned `same` ends at 0.79, 0.89 and 0.99 on the test x4 = 1 half, while its frozen readout is 1.00 on all three replicates.
  - The n = 32 trajectory panel shows `same` reaching about 0.97 and then declining over phase 2.
  - At n = 16 the figures are 0.70, 0.60 and 0.83 fine-tuned against 0.74, 1.00 and 1.00 frozen.
- `rel` fine-tuned at n = 32 (0.70, 0.71, 0.94) is below its readout (0.77, 0.81, 0.95) on all three replicates.
- So at n ≤ 32 the fine-tuned comparison mixes "what transferred" with "what phase 2 (Adam reset, lr 3e-3) destroys on a few examples".
- *Affects:* reading "part ≈ rel under fine-tuning at n = 32–64" as no z-specific transfer. The frozen readout is the cleaner transfer measure at small n. Fine-tuned results at n ≥ 64 are not affected in the same way: rel fine-tuned is at least as high as its readout there.

**C2.4 (CONCERN, interpretation) — counterexample to "y1 pretraining builds our z" holds.** The prediction was stated in QUESTION.md (Revisions) before computing.
- *Test.* On the same frozen y1 features, the same readout was fitted for y2' = (x1⊕x3)∧x4 and y2'' = (x2⊕x3)∧x4.
- *Result.* Test x4 = 1 accuracy by readout target (three replicates):

  | n | y2 | y2' | y2'' |
  |---|---|---|---|
  | 128 | 0.98–0.99 | 0.96–1.00 | 0.98–0.99 |
  | 512 | 0.99–1.00 | 1.00 | 0.98–1.00 |

  - At n = 32–64 the order changes from replicate to replicate. At n = 32 the best target is y2'' (0.80) for 52200, y2' (0.90) for 52201, and y2 (0.95) for 52202.
  - Untrained features give 0.52–0.90 for all three targets.
- *Reading.* y1 pretraining makes all pairwise XORs of x1, x2 and x3 linearly available, as y1's symmetry predicts. The `rel` > `part` readout advantage therefore says that y1's features contain *a* family that includes z. It does not say that z is privileged or that y2 reuses a designated z.
- *Affects:* any NOTE claim that frozen-feature transfer identifies z.
- The lane's follow-up (`1d9d240`, a {x1,x2} pair task without z) is the right next step. The mechanism analysis should report z next to x1⊕x3 and x2⊕x3, as QUESTION.md plans.

### M4 (`f493c1a`, NOTE.md and follow-up `split`)

Evidence: the `final.csv` paired differences recomputed by script, the reviewer runs below, and [compare_gate.py](compare_gate.py), with output in [out/compare_gate.txt](out/compare_gate.txt).

| checklist item | status | evidence and what it affects |
|---|---|---|
| Oracle and split | PASS | §2 M1. The `split` label (2·(x1⊕x3⊕x5) + (x2⊕x6⊕x7)) is a 4-class code with no x1⊕x2 term, as stated. |
| Loss and metrics | PASS | `head_metrics`' balanced accuracy now averages over the classes present, which is correct for the 4-class `pre` head. The readout was reproduced exactly (M2). |
| Data and compute exposure | PASS, with a stated caveat | Every pretraining arm gets 2,000 updates on the same 512-row pool, followed by an Adam reset. `split` carries 2 bits of label per row against 1 for the others, and 4 y2-irrelevant input bits. NOTE.md lists both under Limitations. |
| Intervention semantics | PASS | NOTE obs. 6 states correctly that a uniform class-mean shift along a z direction cannot flip a linear readout of z⊕x3. It also states that a y2 network's own z direction flipping y2 only where x4 = 1 holds for every y2 solver. The steering result is reported as uninformative about reuse, which is the right reading. |
| Checkpoints and optimizer | PASS | `pre` is frozen in phase 2, Adam is reset at each phase, and the readout has its own state (M1). `pretrained.npz` leaves the `new` head at its initialization (M2). |
| Figures support claims | PASS, one numeric slip (C2.6) | All paired ranges in NOTE obs. 2–4 were recomputed from `final.csv`. Each matches to rounding except one (C2.6). |
| Counterexamples | done | C2.4 (symmetry, M2) and C2.5 (gate-only arm, below). |

**C2.5 (CONCERN, wording) — counterexample run: "fine-tuned transfer points at sharing the gate bit x4" is refuted.** The prediction was stated in QUESTION.md (Revisions) before running.
- *Runs.* `r1-rev-share-gate-s5220{0,1,2}-a1` pretrain on g = x3⊕x4⊕x5, which shares only x4 with y2. They use R2's own `train.py` at `f493c1a`, unedited, through [arm_shared.py](arm_shared.py), with the lane's replicate seeds reused on purpose for pairing.
- *Wrapper check.* A replay of the lane's `part`/52200 run (`r1-rev-share-replay-part-s52200-a1`) reproduces `final.csv` to its 6 printed digits, fine-tuned and readout, at every n. Phase 1 reached 1.00 test accuracy on g, with flips only on x3, x4 and x5.
- *Result,* fine-tuned test accuracy on x4 = 1, replicates 52200/52201/52202:

  | n | gate | gate − fresh | gate − part | gate − unrel |
  |---|---|---|---|---|
  | 32 | 0.53/0.55/0.50 | −0.04/+0.01/−0.11 | −0.06/−0.16/−0.24 | −0.02/+0.03/−0.06 |
  | 64 | 0.62/0.67/0.57 | −0.15/−0.09/−0.17 | −0.16/−0.27/−0.37 | −0.05/+0.00/−0.07 |

  The frozen readout is also at or below `fresh`: −0.07 to −0.12 at n = 64–128.
- *Reading.* Sharing the gate bit alone gives no transfer; `gate` behaves like `unrel`. The other branch, "fewer y2-irrelevant pretraining bits", is not monotone either. With irrelevant-bit counts in brackets, fine-tuned gain over `fresh` at n = 64 is about +0.13 for `part` (1), −0.14 for `gate` (2), −0.10 for `unrel` (3), and about +0.03 for `split` (4).
- What the data support is narrower: `part`'s fine-tuned benefit needs x2 *together with* x4 in one parity. Neither single factor named in the NOTE accounts for it, and at n ≤ 32 fine-tuning also degrades pretrained features (C2.3).
- *Suggested NOTE edit.* In "Reading → Fine-tuning", replace the two named explanations with: "not explained by the gate bit alone (reviewer gate-only arm) or by the irrelevant-bit count alone".

**C2.6 (minor numeric).** NOTE obs. 2 gives `split` − `rel` as "−0.19 to −0.38" at n = 32, 64 and 128. Recomputed from `final.csv`, the range is −0.16 to −0.38. Replicate 52200 at n = 128 is −0.159, and at n = 32 it is −0.186. Every difference is still negative, so the claim's direction is unaffected.

**Reading of NOTE obs. 5 ("frozen-readout success tracks linear z availability").** For a linear head, reading y2 = z∧x4 on the x4 = 1 half *is* reading z linearly, so this is close to true by construction. The informative part is which pretraining makes z linearly available. By C2.4, y1 makes all three of its pairwise XORs available equally. The NOTE's "(with its sibling XORs)" and "not singled out" already say this.

**Verdict.** No BLOCK. The NOTE's claims are supported by its figures and tables, and its limitations name the right confounds. Two edits are suggested (C2.5, C2.6). Neither changes a run, a figure or the frozen-feature conclusion. Mergeable as is, with these edits noted, or after them.

## §3 R3 training history and recovery

Reviewed at lane `f71c84f`: QUESTION.md, history.py and plan.json, with the M1 registry only. The evidence is [oracle_history.py](oracle_history.py), with output in [out/oracle_history.txt](out/oracle_history.txt).

| item | status | evidence and what it affects |
|---|---|---|
| Oracle | PASS | The domain has x_k = (i >> k) & 1, checked against a string rewrite. A, Brel, Bunrel and notA labels match, each 512/1024. Every pair of A, Brel and Bunrel agrees on exactly 1/2 of the domain. |
| Split | PASS | Equals `default_rng(53100).choice(1024, 256, replace=False)`. 256/768, disjoint, covering the domain. All 8 patterns of each task's bits are present in train; per-pattern counts are 24–38. Train positives: A 132, Brel 132, Bunrel 121. |
| Metric arithmetic | PASS | `ce_acc` equals a per-row loop. |
| Identical A checkpoints across branches | PASS (plan level) | All 39 checkpoint-started plan entries pass `--from-sha256`. Each of the 3 source files is named with one hash throughout. `history.py` refuses a file whose SHA-256 differs, and checks the loaded weights against the hash recorded at save time. §0 showed that a file hash identifies contents on this environment. Run-level confirmation from the published configs is still owed. |
| Optimizer and freezing (C0.1, C0.2) | PASS | `trainable` is the trunk plus the active head only. `adam_state` is called at every phase start. Each run asserts that frozen parameters are bitwise unchanged. The readout uses its own fresh state. |
| Exposure, interventions, figures | NOT YET CHECKED | |

## §4 R4 competing predictive routes

Reviewed at lane `4872093`: QUESTION.md, task.py and train.py; the runs since are calibration only. The evidence is [oracle_routes.py](oracle_routes.py), with output in [out/oracle_routes.txt](out/oracle_routes.txt). It was written from QUESTION.md before the lane's code was read.

| item | status | evidence and what it affects |
|---|---|---|
| Oracle | PASS | The domain is 1024 distinct ±1 rows. Labels equal plain-Python parity of positions 0–2, and s is position 3. `a` and `c` are bijections onto the rule patterns and contexts. `index` addresses the matching row. |
| Split by base | PASS | The 256 training bases are exactly the 32 training contexts × 8 rule patterns. Held-out is every input with an unseen context (512). No base crosses the split under any condition or s value. Held-out is closed under s flips and rule-bit flips. |
| Exceptions and pairing | PASS | For every ρ: realized ρ = 1 − k/32 exactly, labels and s 128/128, and k exceptions per rule pattern. Exceptions nest across ρ, and conditions differ only in s on exception bases. Shortcut-only uses the ρ = 1/2 inputs with target s, which is independent of y and of each rule bit. Each irrelevant bit is exactly independent of y in training. Exceptions per training context at ρ = 1/2 range over 2–6 of 8, so exception status is not a context feature. |
| Loss and metric calculations | PASS | All 14 logged metrics (train, conforming and exception accuracy; reliable, reversed and independent; flip, signed and absolute deltas for s, rule and irrelevant bits) equal a separate pair-by-pair calculation on random margins, for ρ = 7/8 and shortcut-only. `train.py` logs before the update at each logged step, so step 0 is the initialization. |
| Exposure | PASS | 256 rows × 20,000 updates in every condition, with the same init, the same 1,666 parameters, wd 0 and one head. Confirmed in the 21 published configs. |

### M2 (`f0f68f7`, first figure)

Evidence: [recompute_routes.py](recompute_routes.py), output in [out/recompute_routes.txt](out/recompute_routes.txt).

| item | status | evidence and what it affects |
|---|---|---|
| Published artifacts | PASS | All 96 entries of `runs/MANIFEST.sha256` match the published bytes. The registry has 24 `running` and 24 `complete` events, with none failed, capped or missing. The plan entries for the main runs were committed (33706fa, 02:42:04) 9 s before the first launch (02:42:13). That is commit time, so push order is not verifiable from git. |
| Key numbers recomputed | PASS | For all 24 runs, `final.npz` was read with plain `np.load`, run through a forward pass written here and scored with the separate metric code from §4 M1. The results equal the last metrics row and `routes_at_T.csv` exactly (worst difference 0). |
| README trajectory claims | PASS, with one wording note | Re-derived from `metrics.jsonl`. Shortcut flip ≥ 0.95 first at 22–41 updates for every ρ > 1/2. Training set fit at 366–603. Reversed accuracy at the fit: 0.00, 0.035–0.062, 0.105–0.117 and 0.31–0.33. ρ = 3/4 crosses 0.5 at 546–737 with train loss 1.3–2.3e-2. *Note:* README's "ρ = 7/8 crosses 0.5 at 2,989–3,651, loss 1–3e-4" uses "reaches 0.5"; seed 54200 sits at exactly 0.500 at 3,651. Strictly above 0.5, the range is 2,989–4,036 at loss 0.9–3.2e-4. Not material. |
| Flip-rate identity | PASS | Whenever reliable accuracy is 1, flip_s = 1 − reversed and flip_rule = reversed. This holds in all 1,089 such logged rows, and it follows exactly from the pair construction: each s pair and each rule pair has one reliable and one reversed member. The lane's conclusion is right: the signed logit deltas are the informative dependence measure. |
| Intervention semantics | PASS | Input-level flips on held-out pairs only. s pairs are oriented (s = 0, s = 1) and rule pairs (y = 0, y = 1), so a positive signed delta means the margin moves the way that route predicts. Irrelevant pairs keep only held-out endpoints. As the lane says, these describe what the function depends on, not how it is represented. At ρ = 1 every reversed input is off the training distribution (QUESTION.md says so). |
| Checkpoints and optimizer | PASS | One phase, Adam from a fresh state, and `final.npz` holds weights plus Adam state. The follow-up's bitwise check of `step20000.npz` against the published `final.npz` is the right test. I check it at M4. |
| Figure supports text | PASS, minor | Inspected `routes_over_training.png` and `routes_at_T.png`. Every claim in the README's M2 list is visible in them. *Minor:* the purple exception-base train-accuracy traces in the first column have no legend entry. They are labeled only in `routes_at_T.png`. |

**C4.1 (CONCERN, wording) — counterexample to "which route wins depends on the held-out context" (ρ = 15/16).** The prediction was stated in QUESTION.md (Revisions) before computing.
- *Method.* For each seed I took the 32 held-out contexts × 8 rule-patterns matrix of reversed-environment correctness and split its variance into main effects and interaction. Shuffled-null expectations are 31/255 = 0.12 for the context share and 7/255 = 0.03 for the pattern share.
- *Result.*

  | seed | context share | pattern share | interaction | contexts all-correct / all-wrong |
  |---|---|---|---|---|
  | 54200 | 0.25 | 0.07 | 0.68 | 1 / 2 of 32 |
  | 54201 | 0.21 | 0.07 | 0.72 | 0 / 6 of 32 |
  | 54202 | 0.23 | 0.10 | 0.68 | 0 / 2 of 32 |

- *Reading.* Context matters about 2× more than the null and more than rule pattern (about 2–3× its null). But most of the variance is base-specific (context × pattern), and almost no context goes entirely one way.
- So neither my counterexample (pattern-level choice) nor the strong form of the lane's claim (context-level choice) holds. "Which route wins depends on the individual held-out base, more on its context than on its rule pattern" is what the data support.
- *Affects:* that one sentence in the README, and the NOTE if it repeats it. Nothing else.

**Observations to pass on (not findings).**
- **ρ = 1/2 rule-only.** s is exactly independent of y in training, yet the shortcut flip rate rises transiently to about 0.6–0.7 around 100–300 updates before collapsing to 0 (figure, row 6, column 3). s is used transiently, plausibly as a memorization feature, before the rule wins. The README does not mention it.
- **Late-phase dynamics are optimizer-specific.** The rule's growth after the fit (500 → 20,000 updates, at train loss 1e-2 down to 1e-9) happens in the regime where Adam's normalized steps keep moving the weights while gradients shrink. The lane's own follow-up diagnostics (√v̂ near eps, float64 saturation near) show this. Statements about which route "training selects" late should name the recipe (full-batch Adam, lr 1e-3, wd 0), as QUESTION.md's confounder list already does.
