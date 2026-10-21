# Coordinator notes: expedition e1

Started 2026-10-21 from `swarm/exploration` at `e0a6e0f17baa47a96ebdf9aa73fbf1ce3b7f145a`.

Upstream `bubskee/modest-complexity` main was not recorded: this session's GitHub access is scoped to `bfaber-centaur/modest-complexity`. Record it before preparing any upstream PR.

## Stage 0 decisions

- Seeds are separate: `--data-seed` (split), `--init-seed`, `--control-seed` (shuffled-label table), `--diagnostic-seed`. Omitting them keeps the historical derivation from `--seed`, and `data(31, 1000)` is byte-identical to e0a6e0f (pinned in `test_exploration.py`).
- `split_sha256` hashes inputs, addition labels and split membership, so it is the same for both tasks; `target_sha256` identifies the labels trained on. Every launched run must match the plan's split hash before training starts.
- Exploratory runs use `--diagnostics none`: loss, accuracy, margins, norm and the actual last-step AdamW displacement every 100 steps; full checkpoints (weights and Adam moments) every 1,000 steps, at the last step and at the first logged step with train accuracy >= .99. The historical inline compression, top-four intervention and readout probe are left for offline analysis. The readout probe is now logged as `readout_accessibility` when it does run.
- Run IDs: `e1-<session>-<condition>-w<width>-i<init seed>-a<attempt>`, where session is `co` (coordinator) or `st` (size/time), condition is `add` (decay 1), `add0` (no decay) or `shuf` (shuffled labels, decay 1). Host and session label are recorded separately in config and registry. A rerun gets a new attempt number or session prefix and is never pooled as a new seed.
- The three-run first preview (seed 30000, widths 32/128/512) runs in the coordinator session; the other 21 runs are assigned to the size/time lane in a seeded mixed-width order.
- Matplotlib 3.11.2 is locked in the `analysis` dependency group; `requirements.txt` (default groups) is unchanged.

## Measured speed before launch (this session, 4 vCPU Xeon, OpenBLAS Haswell kernel, 1 thread)

Training only, 1,000 full-batch updates: width 32 0.47 s, width 128 1.25 s, width 512 4.76 s. A 50k-update width-512 run should therefore take about 4-5 minutes including logging and checkpoints. These are pre-launch estimates; actual run times are in each run's `runtime.json`.
