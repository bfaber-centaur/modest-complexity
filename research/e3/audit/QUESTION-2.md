# e3 audit, second investigation: checks of the two e3 experiments (written 2026-10-22, about 00:25 UTC, before running any check below)

Session `e3-au-cloud-1`. Base `bd4a97c` (`swarm/exploration`), with shared code at the e3 pin `9041a48` (DECISIONS D3). Budget 30 min (D7); about 0.5 min used so far.

## Already known

- I have read the `experiment.py` and `launch_runs.py` diff `4783a47..9041a48`, the two `test_continuation_*` tests, DECISIONS D3–D7 and the plan. I have run nothing at the pin.
- The coordinator checked the dynamics splits with `experiment.data` (`ladders.py`, `ladders.json`) and ran an arm-A smoke check (step 0 reproduces the e1 step-50,000 row).
- The new tests replay the continuation with `experiment.py`'s own `adam` and `lossgrad`. They test the plumbing (weights loaded, moments zero, update count restarted), not the update rule.

## Mechanisms gate, part 1: continuation code (blocking; checked now)

| # | Check | PASS if |
|---|---|---|
| C1 | Weights, moments and update count | At the pin, with a plan reserved entry's settings and an arm-A start file built from the real source checkpoint, `experiment.py` writes a step-0 checkpoint whose U, V, W equal the source exactly and whose moments are zero. An independent replay (`research/audit/independent.py`: own forward, gradient and AdamW) from the same weights with zero moments and t = 1, 2, … agrees with its checkpoints at 10, 30, 100 and 300 updates to max abs difference ≤ 1e-9 × max abs weight |
| C2 | Not a silent resume | The same independent replay with the source's own moments and t = 50,001, … differs from the run at 10 updates by far more than C1's tolerance |
| C3 | Post-edit row | The step-0 metrics row equals the source run's step-50,000 row (train, cal and audit accuracy and cross-entropy, weight norm), and my own forward pass, to 1e-12 relative |
| C4 | Data, objective, schedule | The continuation's `config.json` and the source's agree on data seed, split and target hashes, task, width, lr and wd; there is no learning-rate schedule in the code |
| C5 | Hash guard and launcher | The shared tests pass at the pin. For each of the nine reserved entries, `launch_runs.command` passes `--init-weights`, `--expect-init-sha256`, `--init-age 50000`, `--checkpoint-steps 10,30,100,300`, `--steps 20000`, `--log-every 10` and the source split hash. A reserved ID is refused |
| C6 | Sources | Each plan `source_checkpoint_sha256` equals the SHA-256 of the file it names, and that file holds step 50,000 |

This check run is a reproducibility check of at most 300 updates in a scratch directory. It is not a research run, gets no run ID and is not published; its timing counts toward my budget.

## Mechanisms gate, part 2: start files (blocking; checked when the lane pushes them)

| # | Check | PASS if |
|---|---|---|
| M1 | Format | Each of the nine files has keys U, V, W only, float64, with shapes (31, 128), (31, 128), (128, 31); its SHA-256 equals the hash the lane sends |
| M2 | Arm A | U, V, W equal the source checkpoint's exactly |
| M3 | Arm B | Equal to my own call of e2 `filter_units(w, dominant(w)[0])` (`research/e2/mechanisms/unitfilter.py`, unchanged since e2) on the source, to 1e-12 |
| M4 | Arm C | The lane's sham code reads no data split, label or evaluation set (code reading); rerunning it from the source weights with seeds 37000–37002 reproduces the files; total kept input energy equals B's within the bisection tolerance |
| M5 | Update-0 report | It exists for all nine and lists displacement and kept energy per matrix, gate flips, logit change, train loss and accuracy, and accuracy on the 483 common pairs; my own forward reproduces the accuracies |
| M6 | Matching limits visible | The lane's text calls C "input-energy-matched" and reports the W displacement, per-unit displacement, gate-flip and immediate-damage mismatches at update 0 |

## Dynamics (not blocking launch)

| # | Check | PASS if |
|---|---|---|
| Y1 | Splits | My rewrite of the split rule (`independent.regenerate_split`) at data seeds 41001–41003 and shares .3/.4/.45/.5 gives the plan's `split_sha256` and `train_pairs`; the training sets nest; every split is transpose-closed; train, cal and audit partition the 961 pairs; each `eval-d<seed>.npz` is exactly the complement of that ladder's 50% training set |
| Y2 | Design and init pairing | The 24 plan entries are exactly 3 data seeds × 2 init seeds × 4 shares, once each. Landed step-0 checkpoints equal `independent.initial_weights(31, 128, init_seed)`, the same in every ladder and share |
| Y3 | Provenance and accounting | Each launched ID has one running and one terminal registry event; complete runs have 50,000 updates and a `config.json` split hash equal to the plan's; no ID is missing or duplicated |
| Y4 | Replay | One landed run: a from-scratch replay of its first 2,000 updates, and a replay of 1,000 late updates from a published checkpoint with its moments, reproduce its checkpoints (exactly early; late within the rounding sensitivity seen in e2, reported as found) |
| Y5 | Checkpoint predictions | My forward pass on selected checkpoints reproduces the metrics rows (accuracy exactly, cross-entropy to 1e-12 relative) |
| Y6 | Separation of variation | Reviewed when the lane's figure lands: ladders shown separately (different evaluation pairs), init seeds as replicates within a ladder, pairs never counted as replicates |

## Decision rule

Each item is PASS, FAIL or NOT CHECKED. A FAIL blocks only the runs or claims it affects: a C or M failure blocks the affected continuation IDs, and a Y failure blocks claims built on the affected runs. PASS on C1–C6 and M1–M6 clears the nine continuation IDs from my side.
