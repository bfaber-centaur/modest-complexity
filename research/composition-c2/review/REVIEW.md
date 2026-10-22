# c2 independent review

Predictions and the order of work are in [QUESTION.md](QUESTION.md). Disclosure: I wrote r1's per-step-readout arm (`r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`), which is the tied-transition/per-step-readout input to the time and sharing lanes.

| item | verdict | evidence |
|---|---|---|
| Pin `b3e8132` (common/, recovered/, verify_resume) | **CONCERN** (C1). Training may start. | §1 |
| Time | M2 (`543ed95`) and F1 + NOTE (`e44349b`, PR #55): **PASS** on every check. **CONCERN T1** (eps regime; carried into the NOTE). **CONCERN T3**: the NOTE's 75k held-out column is wrong (0.781 / 0.797 / 0.844). **CONCERN T2** (CX-T4 held): at 100k tt-final still does not extend past length 3, so the supervision "speed" reading applies to held-out length-3 accuracy only. CX-T3 held. Notes T4, T5. **All four addressed** at `4e24ebd`/`1fd4ff6` (merged `9092ed1`), with the lane's `lengths` CSV equal to my CX-T4. | §2 |
| Sharing | M2 (`83b3303`, PR #54): **PASS** on every number checked. CX-S1 failed to dent Interpretation 2, whose sign holds at all 41 evaluations from 8k to 10k. wd 0 follow-up (`501071c`): **PASS** on every number checked; note S1 (F4 lead is 0.05–0.13). CX-S2: "nearly flat" is a slow rise in every run while logits keep growing, not a plateau. wd 0 decode and group tables NOT CHECKED. | §5 |
| Generalization | Parts A and B, A5 and the follow-up anatomy (NOTE `6fe1435`): **PASS** on every number checked. Draws are genuinely new (CX-G2). The NOTE adopts G1, G2, G5 and G6. CX-G3: drift predicts failures within a transition in every model, but only 1.2–3.3×, so it is a partial account (note G7). Note G8. | §3 |
| Mechanisms | `817a768` (PR #52): M1–M2 and follow-up B **PASS** on every number checked. CONCERN M1 on the centroid/residual reading, from CX-M1, is **addressed** in NOTE `0fe8624`, as are notes M2 and M3; that revision PASSES, including O6's scale-outside-U claim (0.67–0.98). Note M4 on one README word. | §4 |
| Synthesis drafts `eac869e`, `ef69e01`, `84b5175` | Draft 3 takes SY1–SY7 (**PASS**). Draft 1 had **CONCERN SY1**: the eps qualifier overreaches for the per-step readout, 61–87% of whose gain came before the switch. Notes SY2–SY6 (one false sentence about tied transitions, missing committed checkpoints, disclosure, unchecked inputs, minor). Every other quoted number matches. Draft 2 had **CONCERN SY7** (from T2). | §6 |

## 1. Pin check (`b3e8132`)

Evidence: [rev_pin.py](rev_pin.py), with output in [out/rev_pin.txt](out/rev_pin.txt). The script checks the code under test against a task oracle, a forward pass and a loss written there from the r1 tables. It only imports the pin's code; it never edits it. Every prediction in QUESTION.md §1 held, including the gap I expected in C1.

**PASS.**
- `test_c2.py` passes 10/10 in its own process.
- **Task rows.** `c2task.load` equals the r1 tables. `rows` and `programs` equal a plain-loop oracle for every row at lengths 1–6 and for the train/held programs. `neighbour_answerable` equals an independent recount: ¬N is 42 of 128 held-out rows, and every neighbour is a training row.
- **Init.** `c2rnn.init` equals all 9 recovered r1 `ckpt-0`, every array bitwise. That covers tied/tied (rnn-inter and rnn-final) and tied/per-step (my arm). All four schemes compute the same function at init. Writing one untied copy in place leaves the other copies unchanged.
- **Loss and step weights.** The loss value agrees with the one written here to 8.9e-16 for 4 schemes × 4 weight vectors, including 0 and non-unit weights. All 3,856 gradient entries agree with central differences of that loss; the worst absolute error is 1.6e-9. A weight-0 step gives its untied readout copy an exactly zero gradient.
- **Any-length forward.** A length-5 forward with step map [1,2,3,3,3], and a continuation from h2, equal the forward written here bitwise in all four schemes.
- **No-change evaluation.** `c2train.evaluate` on all 9 recovered `ckpt-10000` gives exactly the last published r1 row (train, held, held_final_pred).
- **Recovery record.** `RECOVERY.json` lists 27 files, and nothing unlisted sits in `recovered/`. For each file, the committed bytes, the recorded SHA-256 and the r1 published manifest agree. The step and scheme stored in each file match its name. My 9 files are also byte-equal to the originals still on this container. The other 18 rest on the r1 manifests; their originals are on the R1 worker's container, which I cannot reach.
- **Cross-host resume.** Run `c2-rev-verify-final-s51200-a1` resumed the recovered rnn-final s51200 `ckpt-1000` to 10,000 with `--step-weights 0 0 1`, on this container. Its 181 logged rows (1,000–10,000) equal r1's published rows. Its `ckpt-10000` weights, both Adam moments and t equal the recovered original. Small files and manifest are in [runs/](runs/MANIFEST.sha256); the checkpoint is local-only. This adds one more container to the coordinator's `verify_resume` (three runs, identical on its host). It shows one host and one scheme (tied/tied, final weights), not every host.
- **c2probe** (added at 04:06, before fitting). On the recovered inter s51200 `ckpt-10000` at probe seed 64300, `decode_table` passes four checks:
  - its labels equal the oracle's symbols;
  - each probe's weights equal r1's own `analyze.fit_probe` bitwise;
  - every entry, re-scored with my own forward, equals `table[fitted][scored]`, so the table is not transposed;
  - the diagonal equals r1's held-out probe accuracy.

**C1, CONCERN: a resume checks the file and the scheme, not the recipe.**
- **What is refused.** `c2train.py --resume` refuses a wrong SHA-256 and a scheme that differs from the checkpoint.
- **What is accepted silently.** In [rev_pin.py](rev_pin.py) §6 (one update each, in a temporary directory), a resume of r1 rnn-final `ckpt-10000` was accepted in two cases:
  - With the default step weights 1 1 1. The source checkpoint's own meta says `condition: final`.
  - With another task id, lr 3e-3, wd 0 and init seed 9.

  Each was recorded in `config.json` as if it were the run's recipe.
- **Where the recipe lives.**
  - r1 checkpoints carry `condition` in their meta.
  - c2's own checkpoints carry only step, run ID and scheme. So a resume from a c2 50k state can be matched to its source only through the source run's `config.json`.
  - `config.json` stores the source meta under `resumed_from`.
  - `init_seed` in a resumed run's config does not describe its weights.
- **Risk.** A tied/final continuation started without `--step-weights 0 0 1` would quietly become an intermediate-supervised run. The time lane's verify entry for final (plan at `b7d1ab6`) has the flag. D7's command text (`c2train.py --resume … --diag …`) does not spell it out for the three tied/final continuations. The per-step-readout runs are safe on this point, because the scheme check refuses a missing `--readout untied`.
- **Remedy (no pin change needed).**
  - Every resume plan entry states `--step-weights`, `--lr`, `--wd` and `--task-id` explicitly.
  - Lanes publish `config.json` for every run whose checkpoint is reused.
  - In each lane review, I compare every resumed run's recipe with its source's (`resumed_from.meta.condition`, or the source `config.json`).

**D10, pin defect found by the generalization lane: confirmed, and the reading is correct** (`rev_pin.py` §9).
- With no step map, `c2rnn.forward` raises past length 3. Tied/tied raises AssertionError; the untied schemes raise ValueError. Nothing raises at length 3.
- For tied/tied at length 6, five different valid maps give bitwise identical states and logits, equal to my forward.
- In each untied scheme, two different maps give different logits.
- So an explicit map is harmless for tied groups and required for untied ones. Nothing could have been silently wrong.

**Post-pin recovery addition.** The 28th `RECOVERY.json` entry is r1's P2, `r1-comp-interp-s51200-a1/ckpt-10000.npz`. Its committed bytes equal its recorded SHA-256 and r1's published manifest.

**Host calibration** (not a check): 5.53 ms per update for tied/tied with weights 0 0 1, one job, peak RSS 41 MiB. The r1 R1 host gave 3.9 and the coordinator's 7.5–8.5. Hosts differ by about 2×, so budgets should be calibrated on each host.

## 2. Time

### M0 (`b7d1ab6`), pre-registration only

**Checked and correct.**
- The threshold is ≥ 122 of 128 rows, and ≥ 40 of 42 for ¬N.
- Censoring: the last window that can complete starts at 49,800.
- The per-step readouts add 2,064 parameters.
- The source table: `tt-final` runs with `0 0 1`, and `tu-inter` with `--readout untied`. So C1 is handled in the text. The plan still holds only the verify entries.
- The joining check and the on-host continuation check with `--diag` are both planned before the main runs.

**T0.1 (note): the "gap persists" rule has no tolerance.**
- It needs churn "≈ 0" but gives no tolerance.
- It also leaves out held-out CE, although measure 5 defines a stable function as no churn *and* flat held-out CE.
- I suggest fixing both before results. For example: no held-out prediction changes over the last 1,000 updates, and ¬N CE changes by less than a stated amount over the last 10,000.

**T0.2 (note): the "speed" rule's ¬N condition has no time.** It needs ¬N ≥ 0.90 but does not say when: at 50,000, or sustained like the main threshold.

My counterexamples (CX-T1, CX-T2) and checks are in QUESTION.md §2.1 and Revisions, written before results.

The lane fixed both notes before reading any continuation record: R1 at f9fb2f9 and R2 at 5f48e4d.

### M2 (`543ed95`, PR #55)

Evidence: [rev_time.py](rev_time.py) with `543ed95`; output in [out/rev_time.txt](out/rev_time.txt). The `tu-inter` sources are my r1 arm.

**PASS.**
- **Recipe against source (C1/D9).** All 9 continuations carry their source's step weights, readout, task, lr, wd and width. The resume hash equals `RECOVERY.json`. `resumed_from` names the source run at step 10,000, with the matching `condition`.
- **Joining.** Each continuation's step-10,000 row equals r1's published row. The joined record runs every 50 updates from 0 to 50,000.
- **Files.** Every file listed in the 9 `MANIFEST.json` matches its SHA-256.
- **Checkpoints.** All 36 committed checkpoints (20k–50k), rescored with my forward, equal the logged row at that step: step count, held-out string, train final and ¬N.
- **Thresholds.** Recomputed from the joined records with my own rule and recount, they equal `figures/time_threshold.csv`: crossing, censoring, minimum after the crossing, end value and last-10k change.

  | arm | crossings | ¬N crossings | at 50k |
  |---|---|---|---|
  | tt-inter | 7,700 / 9,150 / 8,200 | 8,650 / 11,200 / 9,000 | 1.000 |
  | tu-inter | 24,900 / 32,300 / 33,550 | 30,250 / 34,100 / 39,350 | 1.000 |
  | tt-final | censored in all three | censored | 0.56 / 0.70 / 0.66; ¬N 0.12 / 0.26 / 0.14 |

**Lane predictions, for the record.**
- tt-final at 50k exceeds the predicted 0.30–0.60 in two seeds, and ¬N exceeds 0.25 in one.
- tu-inter s51200 crossed at 24,900, just before the predicted "after 25,000".
- The diagnostics prediction (√v̂ below eps ≤ 0.01) failed completely; see T1.

**CX-T1 (tt-final), committed before results.**
- *The gap: held.* On held-out rows still wrong at 50k, the median gap (largest wrong logit minus correct) shrank 15–28% from 40k to 50k. Held-out CE fell steadily, from 3.6–4.0 to 2.7–3.3. So tt-final is still moving toward the boundary, and the lane's "unresolved" (censored, still rising) is the right reading, not "persists".
- *The optimizer: the second half failed.* At 50k the Adam part of the last update is only 1.02–1.06× the decay part. **Every coordinate has √v̂ < eps**, and the Adam step is 0.88–0.89 of the gradient-descent-like (lr/eps)·m̂. I had predicted "not R4's eps regime"; it is that regime.

**T1, CONCERN: the late phase runs in a different optimizer regime, in every arm.** This is a descriptive post-hoc check, prompted by the coordinator's reading of `time_main.png` and computed from the logged `diag`.
- **When.** Median √v̂ falls below eps at 24,100–25,100 updates, and every coordinate follows by 27,800–29,650. The timing is the same in all three arms, tt-inter included.
- **tu-inter.** 61–87% of its 10k→50k held-out gain came before the median crossed eps (0.95 / 0.88 / 0.81 at that point). But two of its three threshold crossings (32,300 and 33,550) and all ¬N crossings fall in the full eps regime.
- **tt-final.** 56–70% of its gain came after the switch.
- **What follows.** The recipe is unchanged, so "the readout gap closes within 50k under this recipe" stands. But past ~25k the optimizer behaves like momentum gradient descent with an effective rate of lr/eps, balanced against decay. Two qualifiers belong in the NOTE:
  - "Speed" here includes the dynamics of that late regime. It is not a continuation of the early Adam dynamics.
  - Any F1 extension of tt-final to 100k stays entirely in that regime.

**CX-T2 (tu-inter), committed before results: held, narrowly.**
- At 50k, V3 read on h1 is 0.81 / 0.97 / 0.84, so below 0.9 in two of three seeds.
- The readouts did move a long way toward each other. ‖V_k − V_3‖/‖V_3‖ fell from about 1.1 to 0.65–1.05. V3 on h2 reaches 0.95–1.00, and V2 on h1 0.94–1.00.
- So the catch-up partly re-ties the readouts, fully in s51201. "Speed" here partly means time to converge toward a shared readout. That is worth one sentence in the NOTE.

### F1 and the NOTE (`e44349b`, PR #55)

Evidence: `rev_time.py 543ed95 e44349b`, functions `f1` and `note_m4`. Output in [out/rev_time.txt](out/rev_time.txt). The checks, CX-T3 and CX-T4 were committed at ff3e589, before I opened any F1 record or the NOTE. The NOTE-number list was committed at b7ddbfe, after F1's checks ran.

**PASS on every F1 check.**
- **Recipe (D9).** All three extensions are tied/tied with step weights `0 0 1`, the task, the seed, lr, wd, width and 100,000 updates. Each `resume_sha256` equals the 50k run's MANIFEST entry and the committed `ckpt-50000` bytes. Each `resumed_from` is that run at step 50,000.
- **Joins.** Each extension's step-50,000 row equals its 50k run's last row, `held_subset` included. The joined steps run 0–100,000 every 50.
- **Manifests and checkpoints.** Every MANIFEST hash matches. All six `ckpt-75000` and `ckpt-100000` rescore to their logged rows: step count, held-out string, train and ¬N.
- **Classification at 100k (handoff 2).** Recomputed from the joined trajectories, every column of `time_followup_threshold.csv` matches, including `preds_differ_last_10k` (8 / 7 / 3), `pred_changes_last_1k` (1 / 1 / 0) and `held_ce_rel_change_last_10k` (−35% / −52% / −28%).
  - s51201 crosses at 95,050 and stays above. s51200 and s51202 are censored. ¬N crosses in no seed.
  - By the lane's rules this is "unresolved", and the NOTE says so.
- **Error compatibility (handoff 5).** All 762 rows of both CSVs equal mine: wrong counts, any-neighbour and same-prefix shares, and both uniform baselines.
  - The NOTE's ranges match at the checkpoint steps from 10k on: any-neighbour 100% (uniform 64–77%), same-prefix 86–100% (uniform 29–37%), and a lowest value of 75% at 1,000-step resolution.
  - At 75k and 100k, same-prefix compatibility is 96–100%.
- **Eps regime (handoff 4).** In `tt-final` every coordinate stays below eps from 29,650 to 99,950. Median √v̂ at 99,950 is 5.3–6.0e-10. 75–86% of the 10k→100k gain comes after the median crosses eps. These agree with T1.
- **Function against parameters.** `time_followup_checkpoints.csv` norms and distances equal mine.
  - The NOTE's 40k→50k distances (2.6–3.4, 5.4–6.1, 4.4–4.6) and the 20k peak (27.6 → 25.5) match.
  - Its N/¬N values at 30k and 50k match, as do its 14–20 changed predictions and its CE falls of 19–56%.
  - Its margins (19.8, 19.0, 18.2–18.3) and its gradient norms (3.5–19e-8) also match.

**T3, CONCERN (a number in the NOTE's F1 table is wrong).** The held-out values at 75k read 0.766 / 0.750 / 0.797. The logged rows, `time_followup.csv` and my rescoring of `ckpt-75000` all give **0.781 / 0.797 / 0.844** (28, 26 and 20 rows wrong). The ¬N column at 75k (0.48 / 0.40 / 0.55) is right.

**T2, CONCERN: the 100k rise is length-3-specific (CX-T4 held).** My prediction held in every seed.

| tt-final at 100k | value | chance |
|---|---|---|
| shared readout on h1 and h2 | 0.00–0.19 | 0.125 |
| accuracy at L = 4–6 | 0.06–0.14 | 0.125 |
| accuracy at L = 3, all 512 rows | 0.975–0.990 | |

- **What it means.** The final-only arm reaches 0.90–0.96 held-out, and 0.74–0.88 on ¬N, without the iterated step function that `tt-inter` has by 10k–20k (exact through L = 6).
- **Length transfer shows no sign of closing.** It is 0.10–0.15 at 50k and 0.06–0.14 at 100k.
- **For the NOTE.** "The supervision gap is a large speed difference within 100k" holds for held-out length-3 accuracy. The Reading should say so, and add that at 100k final-only still does not extend past length 3.
- **For the synthesis.** The same applies to its "Extra labels" paragraph (§6 below).

**CX-T3: held.** On the rows still wrong at 100k (13 / 5 / 11), the median logit gap shrank 59% / 63% / 31% from 75k to 100k. Held-out CE fell from 1.4 / 1.1 / 1.5 to 0.50 / 0.22 / 0.70. This agrees with "unresolved, still rising".

**Notes.**
- **T4.** "Fully so in s51201" (CX-T2 paragraph) is true functionally, since cross-readouts there are ≥ 0.92. It is not true in the weights: ‖V1 − V3‖/‖V3‖ = 0.92.
- **T5.** The parameter norms at 40k–50k are 25.4–35.8, not "25–35".
- **NOT CHECKED.** The path-length estimates (`path_est_prev`, "34–38%"). The NOTE calls them coarse.
- **Scope (handoff 6).** Apart from T2, the wording stays within "this task, recipe and budget". It says "no plateau", "not an asymptote" and "untested" where it should. It discloses that the `tu-inter` sources are my r1 arm.

### NOTE fixes (`4e24ebd`, `1fd4ff6`; merged in PR #55 at `9092ed1`)

I read the diff of `time/` from `e44349b` to `9092ed1` and compared the new CSV with my committed `out/rev_time.txt`. This was a read only, with no new measurement.

- **T2, addressed.** The Reading now scopes the claim to "held-out length-3 accuracy". It adds a bullet saying that at 100k, final-only still does not extend past length 3, and the "does not show" list now has "at length 3 or at any other length".
  - The lane recomputed the numbers with `time_analyze.py lengths` from the committed `ckpt-100000`.
  - Every value in `figures/time_followup_lengths.csv` equals my CX-T4 output to 3 decimals: h1 0.125 / 0.000 / 0.1875, h2 0.109 / 0.039 / 0.148, and L = 4–6 between 0.063 and 0.140.
  - The lane reads h1 and h2 on the 128 held-out rows, and I read them on all 32,768 length-6 prefixes. They agree here.
- **T3, addressed.** The 75k held-out column now reads 0.781 / 0.797 / 0.844.
- **T4, addressed.** s51201's readouts are re-tied "in function", with every cross-readout ≥ 0.92. They are not re-tied in the weights: ‖V1 − V3‖/‖V3‖ is 0.92.
- **T5, addressed.** The norms at 40k–50k are now given as 25.4–35.8.
- No other text, number or checkpoint changed.

## 3. Generalization

### Part A: frozen lengths 1–6 (`30c495f`, draft PR #51)

Evidence: [rev_gen.py](rev_gen.py) with `30c495f`; output in [out/rev_gen.txt](out/rev_gen.txt). Every number below was recomputed with the oracle and forward pass of `rev_pin.py`, not the lane's code.

**PASS.**
- **Task facts.** The four ops generate all of S8 (order 40,320). The 84 products of lengths 1–3 are distinct. The length 4–6 function-class table equals the lane's: 256/256, 1,024/1,015 and 4,096/3,907 distinct, with 8 + 1 + 1 non-novel at L = 6.
- **`length.csv`.** All 168 (run, length, subset) accuracies and row counts equal mine exactly, on every row at L = 1–6. That covers L = 3 train/held/N/¬N, prefix classes and function classes.
- **`steps.csv`.** Per-step accuracy, first-error counts, and the split of the step after a first error (recovered / follows own error / other) equal mine.
- **`references.csv`.** The oracle is 1.0, and each checkpoint's own one-step map re-encoded equals mine. The P2 line is NOT CHECKED: it needs r1's MLP input encoding, and it is a reference line only.
- **The figure.** `figures/length.csv` rows equal their sources. I inspected the PNG.
- **Step map (D10).** `gen_eval.py` passes `[min(t, 3)]`, which is valid. For tied groups it is irrelevant (§1 D10).
- **Run record.** The failed attempt `a1` is kept in the registry and the runs folder.

**Lane predictions, for the record.** All hold.

| prediction | result |
|---|---|
| A1: tied/inter at L = 4 is ≥ 0.9 in at least 2 of 3 seeds | 0.994, 0.926, 0.983 |
| A2: no rise from L = 3 to L = 6 | holds |
| A3: tied/inter at L = 6 is ≥ 0.6 in at least 2 of 3 seeds | 0.856, 0.560, 0.718 |
| A4: tied/final at L = 4–6 is between 0.05 and 0.25 | 0.115–0.136 |

**G1, CONCERN for the NOTE: success past depth 3 is partly window-specific.** My counterexample CX-G1 was written before I opened any table (cb8f982). It predicted no difference, and the prediction failed.
- **The split.** Rows are split by whether any 3-op window of the program, not only the prefix, is one of the 16 held-out length-3 programs.
- **The gap.** tied/inter is less accurate on rows with such a window, in all three seeds:

  | L | gap |
  |---|---|
  | 4 | 0.01–0.10 |
  | 5 | 0.05–0.14 |
  | 6 | 0.09–0.14 |

  For example, s51201 at L = 5 scores 0.686 with such a window and 0.824 without.
- **Why the effect is at the level of the triple.**
  - The split is a Latin square. So every op pair, at positions (1,2), (2,3) or (1,3), completes a held-out program exactly once.
  - A bigram-level effect therefore cannot produce this difference. The effect is at the level of the whole 3-op window.
  - The held-out windows were themselves solved from the start state at L = 3 (held-out 1.000, 0.961, 1.000). The deficit appears when the same windows occur from a later state.
- **Wording I would accept.** "The frozen network extends past the trained depth with declining accuracy, and the decline is concentrated in programs containing held-out length-3 windows."
- **Wording that overstates.** "The network learned a reusable step function." It holds for one draw, three seeds and 10k updates.
- This is an observation about the frozen networks, not a mechanism.

**G2 (note on the figure).** At L = 3 the solid points are "all programs": 384 of the 512 rows are training rows. The like-for-like comparison with L = 4–6 is the open held-out markers. This matters most in the tied/final panel, where the solid L = 3 point (0.82) is mostly training fit. The open markers are there; the caption or NOTE should say which to read.

### Part B: replication on new tasks (`99c83c8`)

Evidence: `rev_gen.py 30c495f 99c83c8`, `partb()`. Output in [out/rev_gen.txt](out/rev_gen.txt).

**PASS.**
- **Draws and splits.** I recomputed both from their seeds. Each draw is r1's permutation draw with r1's acceptance, plus all 84 products distinct. Each split is r1's `split`, checked Latin, with distinct squares per draw.
  - The decision log, with seeds 63000 and 63001 and 63100–63103 all accepted and none rejected, equals `tasks.json`.
  - The accepted tasks, permutations, squares and held-out programs equal `tasks.json` and `c2task.load`.
- **Recipes.** All 16 runs are tied/tied, with inter `1 1 1` and final `0 0 1`. Task and init seed match the ID; lr 1e-3, wd 0.1, 10,000 updates, no resume.
- **Init seeds are not task replication.** Within each (task, init seed), inter and final have identical step-0 rows, so they start from one function. The same seeds 63200 and 63201 meet every task. `ckpt-0` is local-only, so this is checked through the logged rows.
- **Rescoring.** All 16 `ckpt-10000`, rescored on their own task with my forward and my ¬N recount, equal the last logged row. That covers the held-out string, held-out, train and ¬N.
- **CX-G2: the new draws are real new tasks.** My prediction held.
  - No pair among the original and the two new draws is a relabelling of another, under any symbol permutation (S8) with any op permutation (S4).
  - Each draw's only automorphism is the identity. So the two splits of one draw, having distinct squares, are not relabellings of each other either.

**Notes.**
- **G3 (note): draw 63001 has a sink.** All four ops send symbol 2 to symbol 1, which the lane found and reports. Independently confirmed. Draw 63000 has no such symbol. It plausibly explains part of tied/final's higher held-out on that draw (0.49–0.56), and it is a property of the draw, not a defect.
- **G4 (note): ¬N denominators differ by task.**

  | task | ¬N rows |
  |---|---|
  | `ops63000-ls63100` | 33 |
  | `ops63000-ls63101` | 32 |
  | `ops63001-*` | 24 |
  | original | 42 |

  So ¬N accuracies are per-task fractions of different sizes. With 24 rows, one row is 0.042.

### NOTE (`01d1296`) and A5

Evidence: `rev_gen.py 30c495f 99c83c8 543ed95`, `a5()`. It is post hoc and descriptive.
- **G1 and G2 adopted.** The NOTE carries both, worded as suggested. It adds that rows without a held-out window also fall, to 0.64–0.93 at L = 6. That is correct and a fair bound on G1.
- **A5, PASS.** The time lane's tied/inter `ckpt-50000` answers every row at every length 1–6 in all three seeds (my forward, every row). tied/final beyond L = 3 is 0.095–0.15. The NOTE says 0.10–0.15, which is right to rounding.
- **G6 (note, which the NOTE could add).** The length decline was already gone at **20k**: L = 6 accuracy is 0.9996, 0.986 and 1.000. That is before every arm entered Adam's eps regime at about 25k (T1). So unlike the time lane's late catch-up, A5's removal of the decline happened in the ordinary Adam phase.
- **G5 (note).** The NOTE says 0.62–0.71 of first wrong answers are another op applied to the right state, against "uniform 3/7 = 0.43". On the original draw, some ops send a state to the same symbol, so the per-(state, op) baseline averages 0.339 (minimum 0.286). The new draws differ, and draw 63001's sink makes it lower still. So 3/7 is an upper bound, and the stated lean is, if anything, understated.
- **NOT CHECKED.** The task-structure table (neighbour votes, repetition patterns). The anatomy is checked below.

### Follow-up anatomy and drift (NOTE `6fe1435`, merged at `fd4e21d`)

Evidence: `rev_gen.py 30c495f 99c83c8 543ed95 fd4e21d`, `anatomy()`. Output in [out/rev_gen.txt](out/rev_gen.txt). The checks and CX-G3 were committed at 651de53, after I read the NOTE and `summary.json` and before the code existed. I used my own oracle, forward pass, and draws and splits recomputed from seeds. No claim here rests on my r1 arm.

**PASS on every number checked.**
- **`anatomy.csv`.** All 352 (model, transition) rows of first errors and at-risk counts at steps 4–6 equal mine.
- **`summary.json`.** For all 11 models, these equal mine: counts by step, top-1/4/8 shares, both Spearman ρ, the other-op share, the window shares at the error and at risk, and the F1–F3 flags. So do `windows.csv` and `figures/followup.csv`.
- **NOTE ranges.**

  | measure | range |
  |---|---|
  | top-4 share | 0.198–0.290 |
  | top-8 share | 0.362–0.504 |
  | ρ(5, 6) | 0.623–0.956 |
  | ρ(4, 5) | 0.084–0.802 |
  | held-out window at the error | 0.313–0.411 |
  | other op on the true state | 0.617–0.712 |

  F1 holds in 0 of 11, F2 in 11 of 11 and F3 in 3 of 11.
- **F3's baseline.** The at-risk window share is exactly 0.25 in every model. The Latin square forces this whatever the prefix filtering does, so F3 compares against the right baseline.
- **Held-out windows (original three).** Accuracy with a window is lower by 0.011–0.101 at L = 4 and by 0.053–0.138 at L = 5–6. The held-out-prefix split at L = 6 matches the NOTE (0.810 vs 0.871, 0.498 vs 0.580, 0.654 vs 0.739). Programs without a held-out window reach 0.640–0.927 at L = 6 over the 11 models.
- **Drift.** All 154 drift entries of `steps.csv` (22 frozen models, steps 0–6) equal mine. The NOTE's step-6 ranges are right for the original three:

  | | norm | centroid distance |
  |---|---|---|
  | tied/inter | 1.016–1.173 | 1.026–1.188 |
  | tied/final | 1.42–1.60 | 1.49–1.70 |

  The after-first-error shares match per seed and step: recovered 0.266–0.385, follows its own error 0.379–0.414, neither 0.212–0.343.
- **G8 (note).** The drift ranges cover the original task only, where the paragraph sits. Over all 11 models the step-6 ratios are wider:

  | | norm | centroid distance |
  |---|---|---|
  | tied/inter | 1.02–1.27 | 1.03–1.37 |
  | tied/final | 1.42–2.13 | 1.49–2.61 |

**CX-G3, against "matches gradual degradation of the iterated continuous state". The prediction FAILED: 3 of 11 models met it, against the 9 required.** The test, within each transition, compares the error rate of rows whose incoming state h_{t−1} is furthest from its symbol's training centroid (top quartile) with the closest (bottom quartile).
- **Direction: holds everywhere.** The more-drifted quartile fails more often in all 22 (model, step) cases. The top/bottom ratio is 1.3–3.3× at t = 5 and 1.2–2.9× at t = 6.
- **Size: smaller than predicted.** The ratio is ≥ 2× in 7 of 11 models at t = 5 but only 3 of 11 at t = 6. In the three weakest models (`r1` s51201 and the two ops63001 s63201 runs) it is 1.2–1.4× at t = 6, and even the least-drifted quartile fails at 0.27–0.28.
- **Reading.** Within a transition, drift tracks where failures fall in every model, but only partly. Many failures come from states no further from their centroid than successful ones. Drift as measured is therefore at most a partial account of the decline.
- **G7 (note for the NOTE).** The NOTE already says the drift measures are descriptive and that the cause is not identified. Suggested addition: "Within a transition, rows whose incoming state has drifted further fail more often in every model (top vs bottom drift quartile 1.2–3.3×). Low-drift rows fail too, so this drift measure is at most a partial account."

## 4. Mechanisms

### M1–M4 (`817a768`, PR #52; M1–M2 files unchanged since `0949fa0`)

Evidence: `rev_mech.py 817a768 817a768`; output in [out/rev_mech.txt](out/rev_mech.txt). The per-step-readout condition is my r1 arm.

**PASS.**
- **M1.** At ckpt-0 and ckpt-1000, my forward pass gives r1's published 128-character held-out prediction string for all 9 runs. The 10k rows were checked in §1.
- **Natural interchange.** I did actual patching: the donor h_i goes into recipient depth j, the recipient's remaining ops run through the shared transition, and the recipient's readout reads the result (V3 for per-step). The target is recomputed from the oracle. S(i, j) equals `cells.csv` exactly in all 16 cells × 9 runs. The per-step `S_V1` and `S_V2` columns match too.
- **The lane's derivation holds.** S is constant along every diagonal, and the self-swap success equals unpatched accuracy. Interchange at a boundary is evaluation on recombined programs, as the lane's M0 said in advance. For tied runs, it repeats the generalization lane's lengths.
- **Centroid patches.** Centroids fit here on the 384 training rows only give `S_cent` equal to `cells.csv` in all 144 cells.
- **No held-out rows in any fit.** In the centroid and follow-up B code, centroids and U_i are fit on `~held` rows (`mech_interchange.py` lines 51–54, `mech_residual.py` line 67).

- **Follow-up B (`c2-mech-residual-a1`).** I recomputed the residual dose (λ = 0, 0.25, 0.5, 0.75, 1) and the inside/outside-U split. Centroids and U were fit here on training rows. All 756 entries equal `residual.csv` exactly, at depth 2 → op 3 and at depth 3 → L = 4–6, for every subset.
- **The post-hoc energy check.** `c2-mech-energy-a1` is labelled post hoc and descriptive in ADAPTIVE.md (04:23), and it chose nothing. NOTE O6 uses it only to argue "unresolved", which weakens the claim rather than strengthening it. NOTE O6 should still label it post hoc where it cites it.

**NOT CHECKED.** Consistency C, the random controls and random subspaces, the held-out subspace swap, `energy.csv`, geometry, and the D5 decode values. `c2probe` itself was checked in §1.

**Lane prediction for the record.** It predicted tied/inter at L = 0 (V reading h_0) would be ≤ 0.5. It is 1.000 in all seeds, so V reads the start embedding exactly.

**M1, CONCERN: what the centroid advantage shows.** The README says centroid states do better than natural states where those fail, "so the history-specific part of the state goes with the failures".
- **The test.** CX-M1 was committed at 2cfd1e1 before I opened results. It asks, in every cell where S_cent − S ≥ 0.05, how much of the gap two simple changes to the natural donors recover:
  - (a) keep only donors whose h_i is nearest to their own symbol's training centroid;
  - (b) rescale each donor to its centroid's norm.
- **tied/inter at L = 4–6, 11 cells.**
  - (a) recovers only 1–22% of the gap, so symbol injection does not explain the gain. All these donors are read correctly by V at their own depth.
  - (b) recovers **25–63%**, against my prediction of under 25%. So a large part of the harmful "history" variation is the state's overall scale.
- **Diagonal cells, per-step readout and tied/final.** At (3, 3), (a) recovers 64–86%. There the centroid mostly injects the right symbol into held-out states the network got wrong. The lane's own Limits note says as much. At (2, 2) it recovers 10–59%.
- **Wording I would accept.** "Replacing a state by its training class mean fixes most cross-length failures in tied/inter. Matching only the norm recovers a quarter to two thirds of that. On held-out depth-3 states the gain is mostly the injected symbol."
- **Wording that goes beyond the evidence.** "The history-specific part goes with the failures", stated without the scale component.

**M1 applies to NOTE Interpretation 1.** It says the residual "goes with the errors at L = 5–6; shrinking it removes them". The dose numbers are right. But in tied/inter, about a quarter to two thirds of what the class mean fixes is fixed by matching the donor's norm alone. So "history residual" should not be read as purely a history-specific direction.

**M2 (note): NOTE Interpretation 2 is close to tautological as worded.** It says these networks "contain, in their class-mean geometry, a state that the shared transition maps correctly".
- At depth 2, every (symbol, op 3) pair occurs in training, as handoff item 3 says. The class mean is the average of training states that the network maps correctly.
- At (2, 2), restricting natural donors to those nearest their own symbol's centroid recovers 10–59% of the gap. So part of the class-mean gain there is injecting the symbol.
- The NOTE's own Limit, "an intervention the network cannot perform itself", should sit next to this sentence.

**M3 (note): NOTE Interpretation 3 says "share of the symbol's causal effect".** The U-swap's 0.26–0.59, against ≤ 0.13 for random subspaces, is not energy-matched. O6 itself says so. "Swapping the U coordinates moves 0.26–0.59 of held-out answers to the donor's target, against ≤ 0.13 for dimension-matched random subspaces, which carry less energy" is the bounded form.

**Handoff answers.**
- Items 1 and 6: O2 is stated correctly and not overstated. My actual patching equals `cells.csv`, which in turn equals the generalization lane's lengths (§3).
- Item 3: centroids are fit on training rows, and λ = 1 equals natural (my dose recomputation).
- Item 5: the V3 recipient frame is declared and applied as declared. I wrote that arm; my checks used the same code path as for the other arms.

**Design note for follow-up B (`c2-mech-residual-a1`), written before I opened its files (I had seen only the coordinator's summary).** U_i is the span of the *centered* centroids. A pure scale residual, r ∝ c_i(x), therefore lies largely outside U, in the mean-centroid direction. "The harmful residual is outside U" would then partly restate the norm effect above. Reporting the component of r along c_i(x), or along the mean centroid, separately would keep the two apart. I will check B's numbers when they land.

### NOTE revision `0fe8624`

Evidence: `rev_mech.py 817a768 817a768 0fe8624`. The new function is `o6_scale`, and its output is in [out/rev_mech.txt](out/rev_mech.txt). The check was recorded at 35ec5f5 before the code existed.

**PASS. M1 is addressed, and M2, M3 and the post-hoc label are taken up as suggested.**
- **O4 and Interpretation 1.** Every CX-M1 range the NOTE now quotes equals my output.
  - For tied/inter at L = 4–6 (11 cells), (a) recovers 0.01–0.22 and (b) 0.25–0.63.
  - For per-step and tied/final at (3, 3), (a) recovers 0.64–0.86. At (2, 2) it recovers 0.10–0.59.
  - "A quarter to two thirds" is the right reading of (b).
- **Interpretation 2.** It now says the class-mean claim is close to true by construction, gives the (2, 2) and (3, 3) recoveries, and places the limit beside the claim. This is M2.
- **Interpretation 3.** It now gives the U-swap rates per condition (0.50–0.59, 0.35–0.38, 0.26–0.29), which equal O6's table. It drops "share of the causal effect". This is M3.
- **O6.** It now labels `c2-mech-energy-a1` post hoc.
- **O6's new sentence.** "A pure scale residual lies largely outside U" holds, and my prediction held. At depth 2, the share of ‖c(x)‖² outside U is 0.67–0.90 in tied/inter, 0.76–0.97 in per-step readout and 0.82–0.98 in tied/final, so above 0.5 for every symbol in all 9 runs.

**M4 (note, README).** The README's centroid paragraph says the deviation from the class mean that goes with long-length errors "is largely overall scale". Norm matching recovers 25–63% of the gap, median 0.42. The NOTE's "a quarter to two thirds" and "a large part" are the bounded forms. "Largely" in the README is a little stronger than the numbers.

## 5. Sharing

### M2 (`83b3303`, PR #54)

Evidence: [rev_share.py](rev_share.py) with `83b3303`; output in [out/rev_share.txt](out/rev_share.txt). The TU cell is my r1 arm, so every contrast involving TU rests on my runs. The replays below reproduce them on the lane's host.

**PASS.** These cover the charter's sharing checks.
- **Recipe (D9).** All 12 lane runs (UT, UU, and the TT/TU replays) have step weights `1 1 1` and their cell's scheme, plus lr 1e-3, wd 0.1, the original task, the ID's init seed, 10,000 updates, width 128 and no resume.
- **Identical initial functions.** Per seed, the step-0 rows of TT, TU, UT and UU are identical, including `held_final_pred`. `c2rnn.init` was also checked in §1: bitwise r1 draw, one function across schemes, independent copies.
- **Gradient aggregation.** At init on seeds 51200–51202, with all 384 training rows and weights `1 1 1`, the untied copies' gradients sum to the tied gradient (relative difference 1.7e-16).
- **Parameter and decay counts.** Read from the checkpoints: 19,208 / 21,272 / 53,256 / 55,320. Decay applies to every parameter. The lane makes no factorial parameter-control claim (NOTE Interpretation 4, Limitations).
- **Replays.** All 201 logged rows of each of the 6 TT/TU replays equal r1's published rows. The replay checkpoints are local-only, so their arrays are NOT CHECKED.
- **Endpoints.** All 12 `ckpt-10000`, rescored with my forward and recount, equal the last logged rows. NOTE's contrasts recompute from them: TT − UT +0.16 to +0.38, TU − UU −0.09 to −0.19, TT − TU +0.37 to +0.48, UT − UU −0.08 to +0.07.

**CX-S1, the robustness of NOTE Interpretation 2: the claim survives.** My counterexample was committed at f966c73 before results.
- UU − TU is positive at every one of the 41 logged evaluations from 8,000 to 10,000, in every seed (minimum +0.055).
- The interaction (TT − TU) − (UT − UU) is positive at every one too (minimum +0.289).
- So the "every seed" wording does not depend on the exact endpoint.
- All four cells, TT included, were still rising from 8k to 10k (+0.03 to +0.14), as the NOTE says.

**NOT CHECKED.** The decode and readout tables and the per-group optimizer tables.

### Follow-up at wd 0 (`501071c`, NOTE M4)

Evidence: [rev_share.py](rev_share.py) `83b3303 501071c`, functions `wd0` and `wd0_note`; output in [out/rev_share.txt](out/rev_share.txt). The checks and CX-S2 were committed at 69ed8e2 before I opened any wd-0 file. The list of NOTE numbers to recompute was committed at 3be6632, after CX-S2 ran. The wd 0 runs are the lane's. The "wd 0 − wd 0.1" column and every TU comparison at wd 0.1 rest on my r1 arm. No wd 0 checkpoint is committed, so everything here is recomputed from the logged rows.

**PASS on every number checked.**
- **Recipes.** All 12 runs have wd 0 and otherwise their wd 0.1 recipe: step weights `1 1 1`, the cell's scheme, lr 1e-3, the original task, the ID's seed, 10,000 updates, width 128, no resume.
- **Same initial function.** Each run's step-0 row equals its wd 0.1 counterpart's, including `held_final_pred` (r1's rows for TT and TU).
- **Endpoints.** Held-out and ¬N from each last-row prediction string, with my oracle and ¬N recount, equal the logged values.
- **NOTE tables.** The held-out / ¬N / wd 0 − wd 0.1 table, the five paired contrasts per seed, and the wd 0.1 interaction (+0.297 to +0.562) all match to the printed precision.
- **Predictions.** F1 matches (fit by 350–600 updates). So do F2 (−0.55 to −0.60) and F3 (spread 0.086–0.148 against 0.367–0.484).
- **Trajectories.** Held-out at 1,000 is 0.227–0.344. The last-2,000 change is −0.008 to +0.031 at wd 0, against +0.031 to +0.141 at wd 0.1. At 10k, N is 0.430–0.663 and ¬N at most 3/42.
- **Figure CSVs.** All 2,412 rows of `followup_trajectories.csv` and all 24 rows of `followup_endpoints.csv` equal the logged rows and my recount. That includes the parameter counts, the fit step and the five-in-a-row 0.95 step.
- **Optimizer.** The 10k row has no `diag`, so I used the 9,950 rows. They are consistent with the NOTE's 10k values: median √v̂ 5.9e-6 to 2.0e-5, a fraction below eps of 0.020–0.088 from 500 on, update norm 8.4e-4 to 4.9e-3, and minimum train margin 8.7–10.5. The TT s51200 transition norm at wd 0.1, computed from the recovered checkpoint (A, B, bh), is 19.07, matching the NOTE's 19.1. Its wd 0 value (29.0) is read from `groups_wd0.csv` only.

**Note S1 (F4 wording).** TT is highest on 3 of 3 seeds. But its lead over the next-best cell is 0.047–0.133, not "0.05–0.15". The 0.15 is TT − UT at seed 51202, where TU (0.320) is the runner-up. Suggested fix: "by 0.05–0.13 over the next-best cell".

**CX-S2, against "without decay, no cell composes within 10k".** Its first half FAILED and its second half held. Neither half touches the 10k claim itself, which stands. Together they show that "nearly flat" is not a plateau.
- *First half, FAILED.* I predicted that held-out would change by at most 0.03 from 5k to 10k in every run. It rose in 11 runs and held level in one. Three runs exceeded 0.03: TT s51202 +0.055, UT s51201 +0.039, UT s51202 +0.031.
- *1k to 10k.* Over this stretch, held-out rose in all 12 runs, by +0.016 to +0.109 (median +0.059). Answers change slowly, 0.06–0.27 held-out predictions per evaluation on average and at most 3, but in one direction.
- *Second half, held in 12 of 12 runs.* The median train margin grew 31–38% from 5k to 9,950 (I used 9,950 because the 10k row has no `diag`). Held-out CE rose 13–25% from 5k to 10k. So the logits keep scaling up, and the remaining wrong held-out answers become more confident.
- *Suggested NOTE wording.* "Then barely moves" and "nearly flat" should become something like "then rises slowly (+0.02 to +0.11 from 1,000 to 10,000, in every run)". The Reading's closing caveat already says that a flat stretch is not a demonstrated plateau. The data say more: the stretch is not flat, and the 10k values are not where these runs stop.

**NOT CHECKED.** The wd 0 decode table (`decode_table_wd0.csv`, `followup_decode_wd0.png`), the readout table and every `groups_wd0.csv` array. All need the local-only wd 0 checkpoints.

## 6. Synthesis draft (`eac869e`)

Every quoted number was checked against the lane NOTEs, REVIEW.md and my script outputs in `out/`, and the claims were read for scope. No new measurement.

**CONCERN (wording), no BLOCK.** The time rows are pending F1.

**Numbers that match.**
- **Scope.** 384/128/42 rows; 3.84M and 19.2M row presentations.
- **(a) table.** All crossings, the censored arm and the 50k values (`out/rev_time.txt`). "3–4× longer". The gap shrank 15–28% and CE fell (CX-T1). Eps timing 24.1–25.1k and 27.8–29.7k. Two of three per-step crossings and all ¬N crossings fall in the regime, as does 56–70% of tied/final's gain. Distances 0.65–1.05 at 50k, and V3 on h1 0.81 / 0.97 / 0.84.
- **(b).** The table (`out/rev_share.txt`), fit by 350–700 updates, the four contrasts, the interaction (+0.30 to +0.56), and every wd 0 number including TT's 0.05–0.13 lead and CX-S2.
- **(c).** 4,086 of 4,096. The 10k, 20k and 50k length table. G1's 0.01–0.14 and the top-4 share of 0.20–0.29. New tasks: 0.98–1.00, ¬N 0.88–1.00, 8 of 8 pairs, tied/final 0.42–0.56 with at most 2 ¬N rows. New-task L = 6 is 0.58–0.88, from `windows.csv`, which I checked.
- **(d).** S ≥ 0.93, S_cent ≥ 0.939, 25–63% against 1–22%, and 88–97%. The 50k drift is 0.99–1.00×.
- **Elsewhere.** The MLPs' 14–25% (r1 synthesis). Crossings at 7.7–9.2k. Peak RSS 45 MiB. 28 recovered files and 36 time checkpoints.

**SY1, CONCERN: the eps qualifier in (a) overreaches for the per-step readout.** (a) says that "closes with time" includes a change of optimizer regime and "is not a continuation of the early Adam dynamics".
- For tied/final that holds: 56–70% of its gain came after the switch.
- For the per-step readout, 61–87% of its 10k→50k held-out gain came *before* the median √v̂ crossed eps (§2 T1). Only its last approach to criterion (two of three crossings) and its ¬N crossings fall in the regime.
- Suggested wording: "For the per-step readout most of the gain came before the switch, but two of its three crossings and all ¬N crossings fall after it. For tied/final 56–70% of the gain came after it." Next question 1 is worded correctly ("crossings").

**Notes.**
- **SY2 (factual).** "Every RNN here shares its transition across steps" is false for UT and UU. Suggested: "Every RNN here applies one transition per op".
- **SY3 (factual).** The list of committed checkpoints omits the sharing lane's six UT/UU `ckpt-10000` and the generalization lane's 16 new-task `ckpt-10000`. I rescored both sets from Git.
- **SY4 (disclosure).** The per-step-readout arm, in (a) row 2 and every TU number in (b) at wd 0.1, is my r1 arm, reviewed by me. Scope should say so once.
- **SY5 (unchecked inputs).** Two quoted numbers rest on tables I marked NOT CHECKED: UT's 0.75–0.84 off-diagonal decoding (the sharing decode table) and the U-swap rates 0.26–0.59 against ≤ 0.13 (mechanisms). Both equal their NOTEs.
- **SY6 (minor).**
  - "All of it on neighbour-answerable rows" (c): one new-task tied/final run answers 2 of its 32 ¬N rows, so "almost all".
  - UU has 2.9× TT's parameters, not 2.8×.
  - (d)'s "this fits the drift" could carry CX-G3's bound: within a transition, drift accounts for failures only partly (1.2–3.3×).
  - The reviewer row should read 7.4 / 25 min, PRs #50, #53 and #56–#59.
- **Scope.** Apart from SY1, no claim reaches past the tested tasks, recipe and budgets. The short answers carry "under this recipe and these budgets", and (c) says "within what was tested".

### Draft 2 (`ef69e01`): time rows

- **SY1–SY6 are taken.** The eps qualifier now separates the per-step readout (61–87% of its gain before the switch) from tied/final.
- **F1 numbers match.** The short answer and the (a) table and bullets quote F1 correctly: 95,050 in s51201 only; 0.898 / 0.961 / 0.914 and ¬N 0.74 / 0.88 / 0.74 at 100k; +0.02 to +0.06 and CE −28 to −52% over the last 10k. The committed-checkpoint count (42 in the time lane) is also right.
- **SY7, CONCERN (from T2).** Two places need changing.
  - "Extra labels" says that final-only "transfers nothing past length 3 at 50k (100k was not evaluated at longer lengths)". It has now been evaluated: at 100k, L = 4–6 is 0.06–0.14 and the readout reads h1 and h2 at chance (CX-T4).
  - "The extra labels changed how fast composition came on the original table" should read "how fast held-out length-3 accuracy came". By 100k, final-only reaches that accuracy without length transfer.

### Draft 3 (`84b5175`): SY7

- **PASS.** SY7 is taken in all four places: the short answer (a), the final-only and length bullets in (a), the generalization bullet, and "Extra labels".
- **Numbers.** Each new number matches §2 T2 and `out/rev_time.txt` (CX-T4): h1 and h2 read at 0.00–0.19, L = 4–6 at 0.06–0.14 at 100k. The pooled "0.06–0.15 beyond length 3" over 10k, 50k and 100k matches A4 (0.115–0.136 at 10k), A5 (0.095–0.15 at 50k) and CX-T4.
- **Scope.** "Whether the model extended past length 3 at all" is bounded by the sentence's "within this budget", so it does not claim impossibility.
- **Time NOTE.** The time NOTE corrections for T2–T5 are now checked (§2). The synthesis's open item for them can be closed.

