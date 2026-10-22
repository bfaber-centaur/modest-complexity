# e3 dispatch messages (2026-10-22)

These are the coordinator's launch messages to each worker. They are kept here so that a session that did not receive one can read it from the branch. The assignments themselves are in [DECISIONS.md](DECISIONS.md).

Common to all three:

- **e3 base and shared-code identity:** `9041a4839ceef0926015160f312c641f02615241` on `swarm/exploration`. Any later `swarm/exploration` commit that changes no shared code is an equally valid base; `da1ba40` adds audit PR #22. Merge `swarm/exploration` into your designated branch.
- **Instructions:** your role file in [charter/](charter/README.md), then DECISIONS D1–D9.
- **Scope:** own only your `research/e3/<lane>/`. PRs target `swarm/exploration`.
- **Budget (D7):** cumulative elapsed numerical execution from your e3 start, at most two processes, one BLAS thread each, and no paid compute. Do not launch past it; ask first.
- **Measures (D6):** use `research/e3/audit/pairwise.py`. Report P(S over N), P(a+b over N) and P(a+b over S), with the max-based diagnostics alongside. Drop undefined pairs and count them.
- **Follow-up (D8):** one, after the coordinator has reviewed your first figure. Claim it first with run entries and expected minutes.

## Dynamics explorer ("Width and training-time map" thread)

Branch `claude/project-thread-j8mgkm`. Directory `research/e3/dynamics/`, registry `research/e3/dynamics/registry.jsonl`, session label `e3-dy-cloud-1`. Budget 70 min.

Your 24 runs are claimed in `research/e3/plan.json`:
- IDs: `e3-dy-add-w128-i<30000|30001>-tf<30|40|45|50>-d<41001|41002|41003>-a1`.
- Each entry has its full split hash and actual training count.
- Run in `queue_position` order, a seeded interleaving. If you reorder, record why.

Launch from a tree containing `9041a48`:
`uv run --locked python launch_runs.py --plan research/e3/plan.json --root runs/e3 --registry research/e3/dynamics/registry.jsonl --session-label e3-dy-cloud-1 --workers 2 --timeout 600 --ids <the 24 IDs in queue order>`

Ladders, nesting, transpose closure, counts and evaluation pairs were checked with `research/e3/coordinator/ladders.py`. The output is in `ladders.json`, and the evaluation pairs are `research/e3/data/eval-d<seed>.npz` (key `index`).

The data-seed-31000 ladder is the e2 reference, with inits 30000 and 30001 at all four shares already run (D4). Show it, labelled as seen before e3.

Your QUESTION.md (`b800339`) and `ladder.py` (`27b034b`) fit D4 and D6 as written. First figure: accuracy against actual training count, one panel per data seed, both init seeds, with trajectories and the normalized table. Push it, open the PR and tell me before you propose the follow-up.

## Mechanisms explorer ("Frequency retention and rescaling" thread)

Branch `claude/project-thread-psralm`. Directory `research/e3/mechanisms/`, registry `research/e3/mechanisms/registry.jsonl`, session label `e3-me-cloud-1`. Budget 25 min.

**Continuation code (D3, D5).** This is your proposal under other names:
- `--init-weights FILE.npz` takes keys U, V, W only, float64.
- `--expect-init-sha256`.
- `--init-age 50000`, recorded only.
- `--checkpoint-steps 10,30,100,300`.

Adam moments start at zero and the update count restarts at 1 in every branch. The step-0 metrics row and `checkpoint_000000.npz` are written before any update, which gives the post-edit evaluation. The tests are `test_continuation_*`.

**Sham.** Your input-energy rule with seeds 37000, 37001 and 37002 is accepted as declared. Call it "input-energy-matched", not "matched", and report the W-displacement, per-unit, gate and damage mismatches at update 0. The e2 single-frequency null is not part of the initial nine.

**IDs.** These are reserved in `plan.json` and cannot launch yet: `e3-me-<keep|filt|sham>-w128-i<30000|30001|30002>-a1`. The entries already carry the source checkpoint SHA-256, `init_age`, the cadence (log every 10; checkpoints every 1,000 plus 10, 30, 100 and 300) and the split.

**Next steps.**
1. Write `research/e3/mechanisms/starts/<run_id>.npz` with `np.savez` and only U, V and W. A's arrays must be byte-identical to the source.
2. Write the update-0 report (D5 list).
3. Push, send me the nine SHA-256 values, and ask the auditor to check.
4. After the auditor passes them, I fill in the hashes and move the nine entries to `runs`.

Then launch with `--plan research/e3/plan.json --root runs/e3 --registry research/e3/mechanisms/registry.jsonl --session-label e3-me-cloud-1 --workers 2 --timeout 300`.

The held-out set is the 483-pair e2 common set (`research/e2/data/common-never-trained-30-50.npz`), plus each run's own cal and audit splits.

## Independent auditor ("Correctness audit" thread)

Branch `claude/project-thread-id1tmm`. Directory `research/e3/audit/`, registry `research/e3/audit/registry.jsonl`, session label `e3-au-cloud-1`. Budget 30 min; about 0.5 min is already used.

PR #22 is merged at `da1ba40`. A coordinator loop check matched `preference.csv`, and `pairwise.py` is now the shared definition (D6).

**Mechanisms gate (blocking).**
- Check the continuation code: the `experiment.py` and `launch_runs.py` diff `4783a47..9041a48` and the `test_continuation_*` tests.
- When the lane pushes its start files, check:
  - each source equals the plan's checkpoint SHA-256;
  - A is unchanged;
  - B reproduces e2 `filter_units`;
  - C uses no held-out data and its seeds are as recorded;
  - Adam and step reset, data, objective and schedule are preserved;
  - the post-edit row exists;
  - the matching limitations are visible.
- Tell the lane and me PASS or FAIL per item. The nine IDs launch only after PASS.

**Dynamics audit (not blocking launch; the splits were checked by the coordinator).** Independently check:
- splits, nesting, transpose closure and evaluation pairs (compare `ladders.json`);
- init pairing across shares and ladders;
- actual counts;
- provenance and run accounting as runs land;
- selected checkpoint predictions and a short replay;
- the separation of data seed, init seed and dependent pairs.
