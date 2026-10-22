# r1 independent review

Reviewer's plan and pre-stated counterexamples: [QUESTION.md](QUESTION.md). Status values are PASS, CONCERN, BLOCK and NOT YET CHECKED. A finding blocks only what it names.

| section | status | last updated |
|---|---|---|
| §0 shared code | PASS, with three usage concerns (C0.1–C0.3) | 2026-10-22, common at `0c7bd19` |
| §1 R1 composition | NOT YET CHECKED | — |
| §2 R2 shared subcomputation | NOT YET CHECKED | — |
| §3 R3 history and recovery | NOT YET CHECKED | — |
| §4 R4 competing routes | NOT YET CHECKED | — |

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

NOT YET CHECKED. Waiting for M0/M1.

## §2 R2 shared subcomputations

NOT YET CHECKED. Waiting for M0/M1.

## §3 R3 training history and recovery

NOT YET CHECKED. Waiting for M0/M1.

## §4 R4 competing predictive routes

NOT YET CHECKED. Waiting for M0/M1.
