# r1 independent review

Reviewer's plan and pre-stated counterexamples: [QUESTION.md](QUESTION.md). Status values are PASS, CONCERN, BLOCK and NOT YET CHECKED. A finding blocks only what it names.

| section | status | last updated |
|---|---|---|
| §0 shared code | PASS, with three usage concerns (C0.1–C0.3) | 2026-10-22, common at `0c7bd19` |
| §1 R1 composition | **M4: no BLOCK; mergeable, one wording CONCERN (C1.1).** Every NOTE number re-derived. Counterexample 1 failed in the lane's favour: `rnn-inter` composes on rows no neighbour copy answers, and the MLP's above-chance accuracy is all on rows a neighbour copy answers. Counterexample 2: untying the RNN readout drops held-out accuracy to 0.52–0.63 from 0.96–1.00 (paired), so the shared readout carries much of the effect | 2026-10-22, lane `de0ef92` |
| §2 R2 shared subcomputation | **M4: no BLOCK; mergeable.** Counterexamples: z not privileged in y1 features (C2.4); the gate-bit explanation of fine-tuned transfer is refuted by a gate-only arm (C2.5). Two NOTE edits suggested (C2.5, C2.6) | 2026-10-22, lane `f493c1a` |
| §3 R3 history and recovery | **M4: no BLOCK; mergeable.** 12 B-phase replays match the lane's snapshots exactly; specificity counterexample failed (retention is A-specific); every NOTE number re-derived | 2026-10-22, lane `a23bdd8` |
| §4 R4 competing routes | **M4: no BLOCK; mergeable.** The interrupted a1 runs are recorded honestly and equal their a2 reruns; every branch point is bitwise; every follow-up number re-derived. Counterexample failed: at 200k Adam is exactly fixed-step gradient descent (no coordinate has √v̂ ≥ eps), so the stall is the optimizer's. Float64 saturation is immaterial (C4.3), and C4.1 (context wording) is still open | 2026-10-22, lane `d2f154f` |

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

### M2–M4 (`de0ef92`: figures, NOTE.md, RNN follow-up)

Reviewed at lane `de0ef92` (PR #44). The evidence is [recompute_composition.py](recompute_composition.py), with output in [out/recompute_composition.txt](out/recompute_composition.txt). It takes labels, held-out rows and neighbours from my oracle, not the lane's code. The reviewer runs use [rnn_untied.py](rnn_untied.py). Both predictions and the four plan entries were pushed at `00ed987` (03:12:11), before the launch at 03:12.

| checklist item | status | evidence and what it affects |
|---|---|---|
| Oracle and split | PASS | §1 M1. Also, all 9 one-op-slot neighbours of every held-out row are training rows (Latin-square property); the neighbour analyses rely on this. |
| Loss and metrics | PASS | Published `ckpt-10000` of `final`, `inter` and `shuf` (seed 51200) was rescored with a forward pass written here. Every logged train and held-out accuracy per head matches exactly, and so does the 128-character `held_final_pred`. In all 17 runs, the logged held-out predictions scored against my oracle's labels give the logged held-out accuracy exactly. RNN gradient: my untied-readout gradient is the lane's backpropagation with V_t at step t. It equals the lane's gradient bitwise at init and passes central differences (288 entries, worst abs error 7e-10). That independently checks the lane's RNN backpropagation. |
| Data and compute exposure | PASS | All MLP arms use 384 rows and 10,000 full-batch updates. They start from the same trunk: W1, b1, W2, b2 are identical for `final` and `inter`/`shuf` at each seed, checked. `inter` and `shuf` carry 3 labels per row and 2 more heads (22,296 vs 20,232 parameters). `shuf` matches `inter` in heads and label count. RNN arms have 19,208 parameters, with the same rows and updates. NOTE.md states that RNN and MLP seeds are not paired. |
| Intervention semantics | PASS | The NOTE table and the figure use only training-program pairs. The unpatched model is right on every base row (252 result rows), and the source rows are fitted training rows. The composed counterfactual is P[c_base](step2_source), and pairs where it equals either row's answer are excluded, as stated. The held-out pairs in `interventions.csv` (base accuracy 0.10–0.25) are not used. "Random ≤ 1%" is the mean of 5 random subspaces per run (at most 0.99%); single random subspaces reach 1.8%. |
| Checkpoint and optimizer handling | PASS | These are single-phase runs with one Adam state from init and no freezing, so C0.1/C0.2 do not arise. Committed files match their MANIFEST hashes. RNN checkpoints are local-only, but my tied-mode replay of `r1-comp-rnn-inter-s51200-a1` reproduces its `metrics.jsonl` line for line. |
| Follow-up pre-registration | PASS | Predictions `b842eb2` (02:55:06) precede the RNN code and calibration plan `3a58b61` (02:56:06). That commit precedes the calibration launch (02:56:08). The six-run plan `4497ce1` (02:56:23) precedes the first follow-up launch (02:56:25). These are commit times, not push times. |
| Figures support claims | PASS, with C1.1 and two slips | I inspected `heldout.png`, `decoding.png` and `followup_rnn.png`. All seeds are shown, train curves are dashed, the reference lines are drawn and P2 is marked. Every NOTE number was re-derived; the details follow the table. |
| Counterexamples | done | Two, below. |

**Numbers re-derived.** All match except the two marked slips.
- **Held-out final at 10,000:**
  - `final` 0.164/0.172/0.141;
  - `inter` 0.242/0.203/0.250;
  - `shuf` 0.125/0.109/0.102;
  - `rnn-inter` 1.000/0.961/1.000;
  - `rnn-final` 0.273/0.312/0.320.
  P1 reaches 1.0 on all 512 rows and P2 1.0 held-out.
- **Training fit.** Training accuracy first reaches 1.0 at 200 (`final`), 250–300 (`inter`) and 250 (`shuf`). *Slip:* "by about 250" holds except `inter` seed 51200, which fits at 300.
- **Intermediate heads.** `inter` held-out step1 is 1.0 and step2 0.977–1.0.
- **Wrong answers that are neighbour labels.**
  - Any slot: 0.97–1.00 against a uniform expectation of 0.732.
  - Slot c: `inter` 0.80–0.86, `final` 0.65–0.70, against 0.339 uniform. *Slip:* NOTE says "66–70%", but seed 51202 is 0.6545.
  - `heldout_error_types.csv` equals my recomputation in all 36 entries.
- **Probes.** Step1 from h1 at init: 0.96–0.99. From the input: 0.41.
  - Step2 from h1: 0.40–0.42 at init, then 0.52–0.57 (`final`), 0.75–0.83 (`inter`), 0.42–0.47 (`shuf`).
  - Step2 from h2: 0.55–0.66 (`final`), 0.98–0.99 (`inter`).
- **Interchange.** At h1, the output follows the counterfactual in 1–3% (`final`) and 4–7% (`inter`), and keeps the base answer in 77–93%.
  - At h2 in `inter`: 12–16% follow, 42–48% keep, 35–38% give the source answer (random 14–15%).
- **RNN.** `rnn-inter` fits at 400–700 updates. Held-out accuracy is 0.34–0.36 at 1k, 0.48–0.66 at 4k and 0.83–0.93 at 7k. In `rnn-final`, the shared readout on h1 and h2 gives step1 0.06–0.13 and step2 0.09–0.15.

**C1.1 (CONCERN, wording).** This affects the heading "Intermediate heads generalize; the final answer does not" and the Reading sentence "the MLP represents the intermediate state of held-out programs (decodable, and correct at the intermediate heads)". No number changes.
- Step1 depends only on (s, a) and step2 only on (s, a, b). All 128 held-out prefixes occur in training rows with the same labels (checked). So held-out step-k accuracy measures invariance to a new pairing of a seen prefix with op 3. It does not show the intermediate computation generalizing to anything unseen.
- In the RNN this holds exactly. h1 and h2 depend only on the prefix, so held-out step1/step2 accuracies equal the training-row accuracies by construction, in all 6 RNN runs at 10,000. The right panel of `followup_rnn.png` ("held-out programs") therefore plots training states.
- The held-out step2 probes test the same prefixes with a new op 3 attached. The drop from h2 (0.98–0.99, `inter`) to h1 (0.75–0.83) shows that op 3 perturbs the code there.
- Suggested wording: "the intermediate heads are unaffected by an unseen op 3; the final answer does not compose".

**C1.2 (minor, interchange).** `shuf` never sees true intermediate labels. At h2, its step2 probe subspace still makes the output follow the counterfactual in 8–10% of pairs (random ≤ 1.1%), and give the source answer in 33–36% (`inter` 35–38%).
- So the source-answer effect does not need true intermediate labels, and neither does about 60% of `inter`'s counterfactual-following.
- NOTE.md does not attribute the effect to intermediate supervision. But its table lists only `inter` at hidden 2, and a `shuf` row would prevent that reading.

**Counterexample 1 (failed for `rnn-inter`, which supports the lane): is held-out success concentrated where a neighbour copy is right?** The prediction was stated in QUESTION.md (Revisions) before computing.
- *Method.* Each held-out row is labelled N if its true answer equals the final label of some training row that differs in one op slot. 86 rows are N and 42 are not (¬N). A pure neighbour-copier scores 0 on ¬N.
- *Result,* accuracy at 10,000 updates:

  | run | all | N | ¬N |
  |---|---|---|---|
  | `final` | 0.14–0.17 | 0.21–0.24 | 0.00–0.02 |
  | `inter` | 0.20–0.25 | 0.29–0.36 | 0.00–0.02 |
  | `shuf` | 0.10–0.13 | 0.14–0.19 | 0.00–0.02 |
  | `rnn-final` | 0.27–0.32 | 0.41–0.48 | 0.000 |
  | `rnn-inter` | 0.96–1.00 | 1.000 | 1.000 / 0.881 / 1.000 |

- *Reading for the RNN.* `rnn-inter` composes on rows no neighbour copy answers, so the counterexample fails. My strict prediction (¬N ≥ 0.9, and within 0.05 of N) held on 2 of 3 seeds. On seed 51201 all 5 remaining errors are ¬N rows, each answered with a neighbour's label.
- *Reading for the MLP.* Every MLP run, and `rnn-final`, gets at most 1 of the 42 ¬N rows right. All of their above-chance accuracy is on rows a neighbour label answers. This supports NOTE reading (i), and it shows that `inter`'s +0.03 to +0.11 over `final` is not partial composition.
- *New observation: the RNN's rise has two phases.* While `rnn-inter` rises, ¬N lags N:
  - at 1,000–2,000 updates, ¬N is 0.00–0.05 while N is 0.50–0.65;
  - at 4,000, ¬N is 0.05–0.36;
  - at 7,000, 0.55–0.81;
  - at 10,000, 0.88–1.00.

  The early above-chance accuracy is neighbour-like, and composition on ¬N rows comes later. The NOTE's "emerged slowly after the training set was fit" could say so.

**Counterexample 2 (partial): is the shared readout needed?** This was the coordinator's question, and the NOTE leaves it open. The prediction was stated before running.
- *Arm.* `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1` runs the lane's `train.py` at `de0ef92` unedited, except that h1, h2 and h3 get their own readouts V1, V2, V3, each initialized as a copy of the lane's V.
  - It starts from the identical init as the lane's `rnn-inter` of the same seed. The seeds are reused on purpose for pairing.
  - The wrapper checks before training pass, and so does the tied-mode replay.
- *Result:*

  | seed | untied, held-out final at 1k / 4k / 7k / 10k | lane `rnn-inter` (tied) at 10k | untied ¬N | V3 applied to h2, step2 accuracy |
  |---|---|---|---|---|
  | 51200 | 0.26 / 0.31 / 0.46 / 0.63 | 1.00 | 0.24 | 0.52 |
  | 51201 | 0.22 / 0.30 / 0.39 / 0.52 | 0.96 | 0.07 | 0.58 |
  | 51202 | 0.23 / 0.35 / 0.41 / 0.52 | 1.00 | 0.05 | 0.56 |

  - The untied runs fit training at 500–600 updates. Their held-out step1/step2 are 1.0, which is uninformative (C1.1). Every wrong held-out answer is a neighbour label.
  - V2 reads step2 off h2 at 1.00, but V3 reads it at only 0.52–0.58. So no state code shared across positions formed.
- *Against the pre-stated thresholds* (≤ 0.5 means the shared readout is needed, ≥ 0.8 means it is not), all three seeds land in between. That is partial: −0.37 to −0.48 against the paired tied runs, above `rnn-final` (0.27–0.32, unpaired), and still rising at 10,000.
- *Answer to the coordinator.* Within this recipe, sharing the transition and supervising intermediate states through per-step readouts does not reproduce `rnn-inter`. Most of the untied arm's gain over `rnn-final` is on N rows.
- *Limits.*
  - The untied arm has 2,064 more readout parameters, and decay acts on three readouts.
  - Everything is a 10,000-update number, so "slower vs different" stays open, as it does for `rnn-final`.
  - The NOTE's disclaimer ("these runs do not separate the two") remains accurate for its own runs. It may cite this arm as reviewer evidence that the shared readout carries much of the effect.

**Verdict.** No BLOCK on R1's M4. One wording CONCERN (C1.1), plus two numeric slips and C1.2, none of which changes a number or a conclusion. Every NOTE number was re-derived from published artifacts. The neighbour counterexample failed for `rnn-inter` and confirms the MLP reading. The untied-readout arm partly answers the NOTE's open question. Mergeable; I recommend the C1.1 wording fix first.

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

### M2–M4 (`a23bdd8`: first figure, follow-up, NOTE.md)

Evidence:
- [check_history_m4.py](check_history_m4.py), output in [out/check_history_m4.txt](out/check_history_m4.txt);
- [specificity_history.py](specificity_history.py), output in [out/specificity_history.txt](out/specificity_history.txt);
- the reviewer replays `r1-rev-hist-replay-{rel,unrel}-d{1000,10000}-s5320{0,1,2}-a1`, which ran the lane's `history.py` unedited via [replay_history.py](replay_history.py).

| checklist item | status | evidence and what it affects |
|---|---|---|
| Oracle and split | PASS | §3 M1. |
| Loss and metrics | PASS | `ce_acc` (M1). Every relearning time in NOTE obs. 3 was re-derived from `metrics.jsonl.gz` as the first logged step with held-out A ≥ 0.95: fresh-A 540/690/700; kept head 62–330 (related) and 37–560 (unrelated); reset head 230–410 and 190–310; B-only trunk 460–640 and 740–1100; reset head on ckA 110–180. All match. So do the derived claims: kept faster than reset in 10 of 12 pairs (exceptions: unrelated, seed 53200), reset/fresh = 0.27–0.76 with ≤ 0.5 in 8 of 12, and end-of-B over B-only by 50–910 updates. |
| Same criterion and resolution across the relearning comparison | PASS | Fresh-A is phase 1 of the same seed's A run, and the return phases are phases of branch runs. Both are logged by the same `on_grid` (every update to 100, every 10 to 1,000, every 100 beyond) and scored with the same criterion. The coordinator's question is answered: they are comparable, at the stated grid resolution. |
| Data and compute exposure | PASS | All phases are full-batch on the same 256 rows with an Adam reset. Continued-A is matched in updates, not in A exposure, and NOTE/QUESTION.md say so. The readout comparison is matched: one `readout()` function for every snapshot (ckA, end-of-B, scratch-B at t = d, init), with the same 256 training rows, the same 2,000 updates at lr 1e-2 and the same readout init seed 53300. |
| Intervention semantics | PASS | The head-A reset puts `W_A`/`b_A` back to the same seed's init draw (`nets.init(..., seed)`), leaving the trunk and head B as loaded. That is exactly fresh-A's starting head. B-only trunks keep their never-trained init head A. |
| Checkpoint and optimizer handling | PASS | All 66 checkpoint-started runs record the MANIFEST SHA-256 of their source file and use their own seed. 42 start from ckA. The other 24 are follow-ups from end-of-B or scratch-B checkpoints. `history.py` refuses hash mismatches and asserts that frozen parameters are bitwise unchanged. **Replay:** for each of the 12 (seed, B task, d ∈ {1000, 10000}) cases, the reviewer's B phase from the published ckA reproduces the lane's end-of-B snapshot exactly (metrics, readout and null; exact float equality). |
| Follow-up pre-registration | PASS | The prediction commit `58cb3e3` (02:54:40) precedes the first follow-up launch (02:54:47), and all 27 follow-up IDs were already in its plan. That is commit time, not push time. The registry has 75 runs complete, none failed or capped. |
| Figures support claims | PASS | Inspected `recovery_curves.png`. All seeds are shown, along with the fresh-A reference, the criterion and chance lines, relearned markers and no censored runs. The drop in held-out A coincides with B's rise at d ≥ 1000, and is absent at d = 100 and under continued-A. Each numbered NOTE claim maps to `survival_summary.csv` or `followup_head_reset.csv`. |

**Counterexample (failed, which supports the lane): is A's post-B decodability A-specific?** The prediction was stated in QUESTION.md (Revisions) before running.
- *Method.* On the replayed end-of-B trunks, the lane's own readout recipe was refit with only the target changed: C1 = x7⊕x8⊕x9, which no task uses, and C2 = x0⊕x1⊕x7.
- *Result,* held-out readout accuracy:

  | trunk | A | C1 | C2 |
  |---|---|---|---|
  | untrained | 0.58–0.63 | 0.58–0.60 | 0.57–0.61 |
  | ckA | 0.99–1.00 | 0.53–0.57 | 0.66–0.69 |
  | end of Brel | 0.98–1.00 | 0.52–0.63 | 0.75–0.90 |
  | end of Bunrel | 0.91–1.00 | 0.50–0.55 | 0.58–0.67 |

  A − C1 on end-of-B trunks is +0.37 to +0.49 in every one of the 12 cases.
- *Reading.* The end-of-B trunk is not generically parity-rich. What a fresh readout recovers is specific to A. As expected, C2 is high only on related-B trunks, because x0⊕x1 is shared with Brel. That indicates what Brel shares with A; it does not show more of A surviving.
- NOTE obs. 2 stands. Its own caveat ("a linear readout cannot say whether the network would use these features") remains the right limit.

**Note on obs. 1 wording (no change required).** "Forgetting tracks B acquisition, not elapsed updates" rests on two things: d = 100 (B not yet learned) shows no forgetting, and continued-A for as many updates shows none. It is descriptive. B acquisition and B-phase duration are not separated by an equal-length B phase that fails to learn B, and "tracks" claims no more than that.

**Verdict.** No BLOCK and no CONCERN on R3's M4. Every NOTE number checked was re-derived from published artifacts. Branch-identity and optimizer handling were verified at run level and by exact replay. The one counterexample attempted failed in the lane's favour. Mergeable.

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

### M4 (`d2f154f`: NOTE.md, 10× longer follow-up)

The evidence is [check_routes_m4.py](check_routes_m4.py), with output in [out/check_routes_m4.txt](out/check_routes_m4.txt). It uses analysis only, from published files. The prediction was pushed at `aecb9b7` before computing.

| checklist item | status | evidence and what it affects |
|---|---|---|
| Oracle, split, loss and metrics | PASS | §4 M1–M2. All 9 follow-up `final.npz` (200k) were rescored with independent code, and they equal the last metrics row exactly. |
| Interrupted runs and registry honesty | PASS | The registry holds 11 follow-up IDs, each with exactly one `running` and one terminal event. No ID is reused: the reruns are `-a2`. The two killed a1 runs have hand-written `failed` events, marked `logged_by: manual`, with returncode null and a plain reason. Their 92 s wall time equals launch to last metrics write. The a1 partials publish config, metrics and `step20000.npz`, but no `final.npz` or `runtime.json`, and say so. Each a1 `metrics.jsonl` (128 lines, to step 144,119) equals the first 128 lines of its a2 rerun, line for line. *Note:* `run_jobs.py` never writes `failed` with a null returncode itself, so these two events are recognizable only by `logged_by`. That is honest, but a reader filtering on status alone would miss it. |
| Branch points and checkpoints | PASS | In all 11 follow-up runs, `step20000.npz` equals the initial run's `final.npz` in every weight and moment array, with t = 20,000. The lane's `followup_bitwise_check.json` claims the same. Every one of the 147 manifest entries matches its bytes, and every published run file is listed. |
| Pre-registration | PASS | Predictions and plan `5710c35` (02:51:41) precede the first follow-up launch (02:52:08). The a2 plan `f79f170` (03:01:04) precedes the a2 launches (03:01:18). These are commit times. |
| Diagnostics are read-only | PASS | `diagnostics()` reads p and the Adam state and writes nothing. Training arithmetic is unchanged, which the bitwise branch points and the a1 = a2 identity confirm. `mean_update_norm` is the net displacement over a logging interval divided by its length, and the NOTE correctly calls it "displacement per update". |
| Follow-up table and text | PASS, two slips | Re-derived: reversed accuracy, rule share, displacement and saturated fraction at 20k and 200k. The ≈10× and ≈150× falls. The 15/16 gains before and after 39k (rule share +0.04–0.05, then +0.01–0.02; reversed +0.07–0.12, then +0.03–0.09). Seed 54200 crossing 0.5 at 28,001. *Slips:* the median √v̂ at 200k is about 1e-14 only at ρ = 1, and 1–2e-13 at ρ = 31/32 and 15/16. The 15/16 saturated fraction is 0.14–0.23, not 0.15–0.23. |
| Flip-rate redundancy | PASS | `followup_long.png` uses no flip rates; it plots accuracies, signed Δs, rule share and numerics. Column 3 of `routes_over_training.png` does plot flip_s and flip_rule. Once reliable accuracy is 1 (after about 30 updates), they duplicate column 2's reversed curve. The text does not use them as independent evidence. Obs. 1 states the shortcut signature as one fact. Obs. 4 cites the irrelevant-bit flip rate, which is not redundant. The Limitations bullet says so. A one-line note on that column would help readers. |
| Figures support claims | PASS, C4.1 open | Inspected `followup_long.png`. All seeds are shown, a2 is marked as shown in place of the failed a1, and the eps line and the T marker are drawn. **C4.1 is still open.** Obs. 4 still says "which route wins depends on the held-out context". At ρ = 15/16 the variance is mostly base-specific: interaction 0.68–0.72, context 0.21–0.25 (§4 M2). The M2 minor notes (purple legend, the 7/8 crossing at exactly 0.5) are also unchanged; neither is material. |

**Counterexample (failed, which supports the lane): is the late stall the optimizer's, or the loss's?** The prediction is in QUESTION.md (Revisions).
- *Method.* Each run's last Adam step was decomposed from the saved state (m, v, t) at 20k and at 200k. The share of saturated rows in Σ(1 − p_target) was computed in arithmetic that does not round 1 − p to 0.
- *Result:*

  | | at 20k | at 200k |
  |---|---|---|
  | share of the step's squared norm on coordinates with √v̂ ≥ eps | 0.71–0.97 | 0 in all 9 runs |
  | step / (lr/eps)·m̂ | 0.06–0.10 | 1.000 |
  | step with eps = 0, relative to the actual step | 1.4–2.5× | 1.5–6.0 × 10⁴ |
  | saturated rows' share of Σ(1 − p) | ≤ 5e-12 | 5e-7 to 9e-7 |

- *Reading.*
  - At 20k Adam is still mostly in its normalized regime. By 200k every coordinate is eps-dominated, and the step is exactly (lr/eps)·m̂: heavy-ball gradient descent with a fixed step of 10⁵. It shrinks with the gradient, which falls with the smallest training margin (22.8–26.3 at 200k).
  - An eps-free Adam step would be 10⁴ times larger. So the NOTE's "optimizer-limited" holds against a scale-invariant optimizer, and its proposed next test (normalized GD, or tiny eps) is the right one.
  - Plain gradient descent would slow in the same way, because this is the usual vanishing-gradient tail of cross-entropy on a fitted set. Suggested wording: "Adam has become fixed-step gradient descent, so its steps shrink with the loss gradient". That is more exact than "the optimizer stalling".
  - The answer to the coordinator's question is "together with the shrinking gradient": √v̂ ≪ eps alone does not explain the collapse.

**C4.3 (CONCERN, wording): float64 saturation is immaterial.**
- Saturated rows (margin > 36.7) carry under 1e-6 of the gradient weight Σ(1 − p) at 200k. That share is computed without rounding: their 1 − p ≤ 1e-16 is already 10⁶ below the smallest-margin rows' 1e-10, so rounding it to zero changes nothing material.
- Also, only the target-logit component rounds to zero. The other logit still gets e^(−m).
- *Affects* follow-up item 4's third bullet and the Limitations sentence on saturation, which list it beside eps as a cause of the stall. The stall is fully accounted for by the eps regime and the gradient's decay, so the saturation clause should be dropped or marked as immaterial. No number changes.

**Verdict.** No BLOCK on R4's M4. The interruption is recorded honestly and the reruns are exact. Every follow-up number checks, apart from two slips. The optimizer-stall reading survives a direct test, and the counterexample failed. Two wording CONCERNs are open: C4.1 (from M2, unchanged) and C4.3 (saturation). Neither changes a number or the follow-up's conclusion. Mergeable; I recommend the two wording fixes first.
