# e4 dispatch messages (2026-10-22)

These are the coordinator's launch messages, kept so that a session that did not receive one can read it from the branch. The assignments themselves are in [DECISIONS.md](DECISIONS.md).

Common to all three:

- **e4 base:** `ee028d96560a09736ab7846d95cb7a43f80c0aaf` on `swarm/exploration`. Shared code is unchanged since `72cfb5d` (D3). Merge `swarm/exploration` into your designated branch.
- **Instructions:** your role file in [charter/](charter/README.md), then DECISIONS D1–D10 and [LITERATURE.md](LITERATURE.md).
- **Scope:** own only `research/e4/<lane>/`. PRs target `swarm/exploration`.
- **Windows and measures (D5):**
  - fixed states 0, t_fit (70/50/40), 100, 300, 1,000, 2,000, 5,000 and 20,000;
  - held-out set: the 483 common pairs;
  - measures: `research/e3/audit/pairwise.py` plus the σ-margin.
- **Budget (D9):** elapsed numerical execution, at most 2 processes, 1 BLAS thread each. Do not launch past it; ask first.
- **Follow-up (D8):** one, after the coordinator has seen your first figure. Claim it first.
- **Launch:** from a clean tree at the base, with `launch_runs.py --plan research/e4/plan.json`, your registry and `--workers 2`. Put run output in a local root outside git, such as `runs/e4`, then publish into `research/e4/<lane>/runs/<ID>/`.

## Dynamics explorer ("Width and training-time map", `claude/project-thread-j8mgkm`)

- Registry `research/e4/dynamics/registry.jsonl`, label `e4-dy-cloud-1`, budget 20 min.
- **Claimed (D6):** `e4-dy-filt-wd0p{1,3}-w128-i{30000,30001,30002}-a1`, at 20k updates, with checkpoints 10, 30, 100, 300, 500, 1,500, 2,500 and every 1,000.
- Run them in `queue_position` order, with `--timeout 300`.
- Your QUESTION.md and `decay.py` (`e5fd1c2`) fit D5 and D6. The equivalence replay you asked for is done (D3); you may repeat it unclaimed.
- First figure: update axis plus exposure axis, with wd 0 and wd 1 as endpoints and non-crossing runs censored. Push it, open the PR and tell me before proposing the follow-up.

## Mechanisms explorer ("Frequency mechanisms lane", `claude/project-thread-psralm`)

- Registry `research/e4/mechanisms/registry.jsonl`, label `e4-me-cloud-1`, budget 20 min.
- **Approved:** the scope in your QUESTION.md (`d584af7`). Start on the published e3 checkpoints now.
- **Claimed (D7):** `e4-me-filt-wd{1,0}-replay-w128-i{30000,30001,30002}-a1`. These are dense diagnostic replays from update 0 to 5,000, with checkpoints every 10 to 300, every 50 to 3,000 and every 500 to 5,000.
  - They replace the update-100 resumes you asked for; replays hold those states, the rise and the peak.
  - Check them bitwise against the published checkpoints.
  - Commit everything except the checkpoint files (about 25 MB per run), plus a SHA-256 list of the checkpoints.
- Every decomposition states its object, its basis rule and its reconstruction error.
- One intervention, after the first figure. If it needs a shared-code change, send the patch and a test first.

## Independent auditor ("Correctness audit", `claude/project-thread-id1tmm`)

- Registry `research/e4/audit/registry.jsonl`, label `e4-au-cloud-1`, budget 25 min.
- Write your QUESTION.md first. Then:
  1. Audit the mechanisms decomposition: its object, basis rule, reconstruction identity, and whether its channels can be separated.
  2. Build the strongest cheap control for the main explanation once each lane has stated it.
  3. Check the dynamics runs (provenance, replay equality, censoring) and any intervention.
- Claim your runs with me before launching.
