# Composition follow-ups c3: index

Bobby's packet: [charter/](charter/) (verbatim). Decisions, design constants, budgets, seeds: [DECISIONS.md](DECISIONS.md). Launch texts and threads: [DISPATCH.md](DISPATCH.md). Milestone log: [MILESTONES.md](MILESTONES.md). Coordinator runs: [plan.json](plan.json), [registry.jsonl](registry.jsonl), [runs/](runs/).

| item | where | status |
|---|---|---|
| integration branch | `swarm/composition-c3` from c2 handoff `7fe75e6` (head of c2 PR #61) | complete; handed off in PR #70 |
| c2 evidence for the premises | DECISIONS D2 | recorded; nothing refuted |
| shared code pin | [`common/`](../common/) at `be41d16` (initial runs), re-pinned at `82cef77` for the scaffolding follow-up (D10) | be41d16: reviewer PASS (PR #62), notes N1–N4; 82cef77: reviewer PASS (PR #66), CX-R1 held (a reset also enlarges the next steps), note N5 |
| learned correction | `correction/` | merged (PR #63); reviewer PASS (REVIEW §2), wording W1–W3 and C6 applied; follow-up F2 (8 new-draw tied/final bases) done; `corr_core` call site fixed for the re-pin signature (5b92e1b) |
| temporary scaffolding | `scaffolding/` | merged (PR #65); M2 reviewer PASS (REVIEW §3); follow-up (Adam reset vs carried at S = 1k, 6 runs) reviewer PASS (REVIEW §5); wording W-F1/W-F2 and figure note applied by the lane (`511a381`, from PR #69) |
| reviewer | `review/` | pin (PR #62), correction (PR #64), re-pin (PR #66), scaffolding M2 (PR #67) and follow-up (PR #68) reviewed, all PASS; synthesis PASS with nits (PR #71, REVIEW §6), nits applied |
| synthesis | [SYNTHESIS.md](SYNTHESIS.md) | final draft; reviewer PASS with nits N-Y1–N-Y5, applied |
| final draft PR to the parent (D1) | [PR #70](https://github.com/bfaber-centaur/modest-complexity/pull/70), `swarm/composition-c3` → `swarm/composition-c2` (c2's #61 still open) | draft, assigned to Bobby; the coordinator does not merge it |
