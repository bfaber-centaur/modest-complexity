# r1 independent review

Reviewer's plan and pre-stated counterexamples: [QUESTION.md](QUESTION.md). Status values are PASS, CONCERN, BLOCK and NOT YET CHECKED. A finding blocks only what it names.

| section | status | last updated |
|---|---|---|
| §0 shared code | PASS, with three usage concerns (C0.1–C0.3) | 2026-10-22, common at `0c7bd19` |
| §1 R1 composition | M1 oracle/split PASS; design notes; M2 not yet | 2026-10-22, lane `ff9b721` |
| §2 R2 shared subcomputation | M1 oracle/split/metrics PASS; two CONCERNs on reading (C2.1, C2.2) | 2026-10-22, lane `c4939c2` |
| §3 R3 history and recovery | M1 oracle/split/metrics/branch hashes PASS; C0.1/C0.2 avoided | 2026-10-22, lane `f71c84f` |
| §4 R4 competing routes | M1 oracle/split PASS; all 14 logged metrics PASS | 2026-10-22, lane `4872093` |

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
| Exposure | PASS (design) | 256 rows × T updates in every condition. Same init, same parameter count, wd 0, one head. |
| Interventions, figures | NOT YET CHECKED | |
