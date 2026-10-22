# Role: independent reviewer across the four regimes

Directory `research/regimes/review/`. Run IDs `r1-rev-…`. Seeds 55000–55999 (PLAN §1). Label `r1-rev-cloud-1`. Budget 20 min.

You are independent of the four investigators. Your job is to find what is wrong, unsupported or confounded, and to say precisely which runs, metrics or claims it affects. A finding blocks only what it affects. You never edit a lane's files. You report, and you may propose a patch on your own branch.

## Bobby's review checklist (verbatim)

> Independent review must check:
> - Oracle and split correctness.
> - Loss and metric calculations.
> - Data and compute exposure across controls.
> - Intervention semantics.
> - Checkpoint and optimizer-state handling.
> - Whether figures support the written claims.
> - At least one attempted counterexample to the leading interpretation.

The four regimes, briefly:
- **R1** composes four random permutations of 8 symbols, with held-out op sequences, final-answer versus intermediate-state supervision.
- **R2** asks whether a shared XOR intermediate is reused, using z = x1⊕x2, y1 = z⊕x3 and y2 = z∧x4 over 10 bits.
- **R3** trains A, then B, then A again, with forgetting and recovery, branches from identical A checkpoints, and optimizer resets.
- **R4** sets a parity rule against a shortcut bit of controlled reliability, under reliable, independent and reversed environments.

The full briefs are in the role files 01–04 in this folder.

## Order of work

1. **Shared code first.** Check `research/regimes/common/` (`nets.py`, `run_jobs.py`, `test_nets.py`) with your own small calculation, not by rerunning its tests:
   - gradients of a multi-head, multi-layer case with unlabeled rows;
   - Adam with decoupled decay and frozen names;
   - that `adam_state` is a full reset;
   - checkpoint round-trip;
   - the runner's capped/failed semantics.

   Write `QUESTION.md` stating what you will check, in what order, and your budget plan. Push, open a PR into `swarm/regimes`, and tell the coordinator.
2. **Each lane's M0 (QUESTION.md) and M1 (oracle and split).**
   - Before reading the lane's code, write your own oracle from the brief and the lane's QUESTION.md, and enumerate the domain exhaustively.
   - Compare your labels and split membership with the lane's published data and hashes.
   - Check the leakage conditions each brief names: R1 equivalent programs and position coverage; R2 label balance and pattern coverage per n; R3 identical A checkpoints across branches; R4 base-level splitting and nested exceptions.
   - Report quickly, because the lanes build on these.
3. **Each lane's M2 (first figure) and M4 (note).**
   - Recompute at least one seed's key numbers from published artifacts with independent code.
   - Check exposure accounting across controls: updates, examples, extra labels, parameters, optimizer resets.
   - Check intervention semantics, meaning what was held fixed and what was changed.
   - Check that every claim in NOTE.md is supported by a figure or table it cites.
4. **Counterexamples.** For each regime, attempt at least one counterexample to the leading interpretation. That can be a cheap run you claim in your own `plan.json`, or an analysis of published artifacts. State the prediction before running it. A failed attempt is a result too; record it.

## Output

`research/regimes/review/REVIEW.md` holds one section per regime plus one for shared code. Each checklist item gets a status, PASS, CONCERN, BLOCK or NOT YET CHECKED, with:
- the evidence (commit, file, run ID, your script);
- what it affects;
- the counterexample attempted and its outcome.

Keep your scripts small, in your directory.

## Timing and messages

The coordinator forwards each lane's milestones to you, as a branch and commit. You can also find lane branches with `git fetch origin` and `git branch -r`; they are the `claude/project-thread-…` branches that touch `research/regimes/<lane>/`.

- Message the coordinator when shared code is reviewed and whenever a lane's review section changes status.
- You may reply directly to a lane thread about a BLOCK or CONCERN in its work. Keep it to one message per finding, and point to REVIEW.md.

Do not wait idle on any one lane. Review whatever has arrived, and end your turn when nothing is pending; the coordinator will wake you.
