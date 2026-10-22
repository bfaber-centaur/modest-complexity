# c2 independent review

Predictions and the order of work are in [QUESTION.md](QUESTION.md). Disclosure: I wrote r1's per-step-readout arm (`r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`), which is the tied-transition/per-step-readout input to the time and sharing lanes.

| item | verdict | evidence |
|---|---|---|
| Pin `b3e8132` (common/, recovered/, verify_resume) | **CONCERN** (C1). Training may start. | §1 |
| Time | M0 read: two wording notes (T0.1, T0.2). Results NOT CHECKED. | §2 |
| Sharing | NOT CHECKED | |
| Generalization | Part A (`30c495f`, PR #51): **PASS** on every number. CX-G1 failed as predicted-against, which bounds the iteration claim (G1). Part B NOT CHECKED. | §3 |
| Mechanisms | `817a768` (PR #52): M1–M2 and follow-up B **PASS** on every number checked. **CONCERN M1** on the centroid/residual reading, from CX-M1. NOTE wording notes M2, M3. | §4 |

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
