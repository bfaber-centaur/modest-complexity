# e2 dispatch messages (2026-10-21)

The coordinator's launch message to each worker, kept here so a session that did not receive it can read it from the branch. Assignments themselves are in [DECISIONS.md](DECISIONS.md).

Common to all three:

- e2 base: `6aaa3a2b6898abd0b317aa775cd7939e7530abdd` on `swarm/exploration` (later commits that change no shared code are equally valid). Shared code identity: `fe284a5` (unchanged).
- Instructions: your role file in [charter/](charter/README.md) (the full e2 contract plus your role; read all of it), then DECISIONS.md D2-D7.
- Keep your designated branch and merge `swarm/exploration` into it; own only your `research/e2/<lane>/`; PRs target `swarm/exploration`.
- Papers and glossary: the context zip in this project's files folder (`uploads/hearth/4611f976-4a58-4b39-a07b-b1bad03fcb19`). Extract it into a new empty scratch directory; do not commit it.
- Budget (D3): 30 minutes of numerical execution counted from your e2 start, at most two processes, one BLAS thread. Report e1 and e2 usage separately.
- Runs (D5): send the coordinator plan entries with reason and expected minutes; launch only after they are in `research/e2/plan.json` at a pushed commit, with `--plan research/e2/plan.json`.
- QUESTION.md: say which observations you had already seen when you wrote the question (D2).

## Dynamics explorer ("Width and training-time map" thread)

Role file `charter/02-dynamics-explorer.txt`. Branch `claude/project-thread-j8mgkm`, directory `research/e2/dynamics/`, registry `research/e2/dynamics/registry.jsonl`, session label `e2-dy-cloud-1`, run IDs `e2-dy-...`.

The common never-trained set is `research/e2/data/common-never-trained-30-50.npz` (key `index`). It has 483 pairs, equals the 50% split's cal ∪ audit and lies inside the e1 audit split. The training sets nest, and the 179 pairs the 50% split adds are all 118 e1 calibration pairs plus 61 e1 audit pairs, so the amount of data and the identity of the added pairs change together. Scores for the six width-128 runs are in `common_set_scores.csv`. Use the set for the first trajectory figure, keeping each condition's operand-associated mask defined from its own training pairs. The 30% width-128 runs have sparse published checkpoints (e1 i30002 only 300 and 50k); a denser trace needs a claimed run.

## Mechanisms explorer ("Frequency retention and rescaling" thread)

Role file `charter/03-mechanisms-explorer.txt`. Branch `claude/project-thread-psralm`, directory `research/e2/mechanisms/` (`research/mechanisms/` stays as the e1 record; its code is reusable), registry `research/e2/mechanisms/registry.jsonl`, session label `e2-me-cloud-1`.

The proposed cycle fits route A: unit-group ablation by dominant frequency against size- and norm-matched random unit sets at both shares, then retrospective tracing. The per-unit concentration was seen before the question, so QUESTION.md has to say so. Name the shams' remaining mismatches (gate changes, functional disruption, selection from the 50k state). Report the common set next to each run's own split for any 30%/50% contrast. D7 records "confirmed in its first half" as an observation plus a candidate explanation.

## Independent auditor ("Correctness audit" thread)

Role file `charter/04-independent-auditor.txt`. Branch `claude/project-thread-id1tmm`, directory `research/e2/audit/` (`research/audit/` stays as the e1 record), registry `research/e2/audit/registry.jsonl`, session label `e2-au-cloud-1`.

1. 50% runs: Part 4 F10-F15 already covers an early and a late checkpoint with independent code. Cite it in the e2 claim table and add only what the brief asks beyond it.
2. Restricted rank: `heldout_rank.py` is the auditor's own, and both mechanisms reimplementations follow its definition. Check the definition itself: the mask uses training pairs only, how operand positions and transposes are handled, ties and set sizes, the per-split chance reference (0.070 vs 0.135), like-for-like comparison across shares, and a null that keeps the mask.
3. Check `common_set.py` (it uses `experiment.forward`, so it is not independent): nesting, the 483 pairs equal to 50% cal ∪ audit, and the scores of 0-0.004 vs 0.878-0.913.
4. Attack one consequential interpretation, for example the mechanisms per-unit concentration reading or "the rank precedes accuracy".

The small-p empty-split guard is deferred as a known gap: it does not affect p = 31 and would force a new code pin mid-round.
