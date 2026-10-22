# c3 dispatch manifest (2026-10-22)

The c3 coordinator is the thread "Four-regime expedition" (`cmsg_01APdvkRg6kVtZy2e9JoDcdT8LMi3arqG7m3Q9w6XU9F6V`), as in c2. All three c3 roles reuse c2 threads, so no new thread is needed. Every launch carries Bobby's packet message (`cmsg_01APdvkRg6kVtZy2e9JoDcdTYZspzxZdnEueRqJCsDp7Qd`). The packet is also committed verbatim under [charter/](charter/).

If threads cannot be used, a human can paste each launch text below into a new Claude Code session on this repository. Together with the files it names, each text is self-contained.

**Pin:** `swarm/composition-c3` at `C3_PIN`.

| # | role | thread (reused) | role file | directory | label |
|---|---|---|---|---|---|
| 1 | learned-correction investigator | "Hidden-state interchange" (c2 mechanisms) | `02-learned-correction.txt` | `research/composition-c3/correction/` | `c3-corr-cloud-1` |
| 2 | temporary-scaffolding investigator | "Composing small programs" (c2 time) | `03-temporary-scaffolding.txt` | `research/composition-c3/scaffolding/` | `c3-scaf-cloud-1` |
| 3 | independent reviewer | "Regime expedition review" (c2 reviewer) | `04-reviewer.txt` | `research/composition-c3/review/` | `c3-rev-cloud-1` |

**Why these threads.**
- The correction lane builds on the class-mean intervention that the mechanisms thread designed, and on its checkpoints.
- The scaffolding lane needs the r1 and c2 full-state trajectories that the time thread produced.
- The reviewer has checked both, and wrote the r1 per-step-readout arm. That arm is one of the correction lane's bases, so this is disclosed.

## Launch text (exact; ROLE, FILE, DIR from the table)

> You are the ROLE in composition follow-ups c3 in bfaber-centaur/modest-complexity. Bobby's packet is attached. c2 is complete and reviewed, and its handoff is pinned.
>
> 1. Fetch branch `swarm/composition-c3` (pin `C3_PIN`) and merge it into your designated branch.
> 2. Read `research/composition-c3/coordinator/charter/FILE`, which is your role prompt verbatim from Bobby.
> 3. Read `research/composition-c3/coordinator/DECISIONS.md`. It holds your assignment, what reviewed c2 evidence says about your question (D2), the shared-code pin, design constants, run-ID prefix, seed range, budget and gates.
>
> You own only DIR. Your PRs target `swarm/composition-c3`. Report each milestone in one line to the coordinator thread "Four-regime expedition" (`cmsg_01APdvkRg6kVtZy2e9JoDcdT8LMi3arqG7m3Q9w6XU9F6V`). Commit QUESTION.md before any result. New training or fitting waits for the reviewer's PASS on the pin, which the coordinator relays. Leave c2, r1, e4, `swarm/composition-c2`, `swarm/exploration` and `main` alone.

For the reviewer only, append:

> Your first task is the pin check (DECISIONS D3): c3train's schedule and resume recipe check, c3correct, test_c3.py, and the verification run. Report PASS, CONCERN or BLOCK to the coordinator.

## Thread and branch assignments

| # | thread id | designated branch |
|---|---|---|
| 1 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTHq7m8eNJQc51iKX371EFA6` | `claude/project-thread-gqcd3n` |
| 2 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTU8s1uzfPigWN1q5DxvtmMS` | `claude/project-thread-xb3nfz` |
| 3 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTAKMwUsYk2djqx7Pz1rukHx` | `claude/project-thread-wvait2` |
