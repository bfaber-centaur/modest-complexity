# c2 independent review

Predictions and the order of work are in [QUESTION.md](QUESTION.md). Disclosure: I wrote r1's per-step-readout arm (`r1-rev-comp-rnn-untied-s5120{0,1,2}-a1`), which is the tied-transition/per-step-readout input to the time and sharing lanes.

| item | verdict | evidence |
|---|---|---|
| Pin `b3e8132` (common/, recovered/, verify_resume) | **CONCERN** (C1). Training may start. | §1 |
| Time | M0 read: two wording notes (T0.1, T0.2). Results NOT CHECKED. | §2 |
| Sharing | NOT CHECKED | |
| Generalization | NOT CHECKED | |
| Mechanisms | NOT CHECKED | |

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
