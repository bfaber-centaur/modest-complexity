# c2 dispatch manifest (2026-10-22)

The c2 coordinator is the thread "Four-regime expedition" (`cmsg_01APdvkRg6kVtZy2e9JoDcdT8LMi3arqG7m3Q9w6XU9F6V`). Two r1 threads are reused, as the project coordinator suggested, and three are new. Every launch carries Bobby's packet message (`cmsg_01APdvkRg6kVtZy2e9JoDcdTDSGFgnxNEX2Y9zwhBs7LTs`). The packet is also committed verbatim under [charter/](charter/).

If threads cannot be started, a human can paste each launch text below into a new Claude Code session on this repository. Together with the files it names, each text is self-contained.

**Pin:** `swarm/composition-c2` at `b3e8132`.

| # | lane | thread | role file | directory | label |
|---|---|---|---|---|---|
| 1 | time | "Composing small programs" (reused) | `02-time.txt` | `research/composition-c2/time/` | `c2-time-cloud-1` |
| 2 | sharing | new: "Tying transitions and readouts" | `03-sharing.txt` | `research/composition-c2/sharing/` | `c2-share-cloud-1` |
| 3 | generalization | new: "Longer programs and new tasks" | `04-generalization.txt` | `research/composition-c2/generalization/` | `c2-gen-cloud-1` |
| 4 | mechanisms | new: "Hidden-state interchange" | `05-mechanisms.txt` | `research/composition-c2/mechanisms/` | `c2-mech-cloud-1` |
| 5 | reviewer | "Regime expedition review" (reused) | `06-reviewer.txt` | `research/composition-c2/review/` | `c2-rev-cloud-1` |

## Launch text (exact; ROLE, FILE, DIR from the table)

> You are the ROLE in composition round c2 in bfaber-centaur/modest-complexity. Bobby's launch packet is attached.
>
> 1. Fetch branch `swarm/composition-c2` (pin `b3e8132`) and merge it into your designated branch.
> 2. Read `research/composition-c2/coordinator/charter/FILE`, which is your role prompt verbatim from Bobby.
> 3. Read `research/composition-c2/coordinator/DECISIONS.md`. It holds your assignment, the shared-code pin, your run-ID prefix, seed range and budget, and the gates.
>
> You own only DIR. Your PRs target `swarm/composition-c2`. Report each milestone in one line to the coordinator thread "Four-regime expedition" (`cmsg_01APdvkRg6kVtZy2e9JoDcdT8LMi3arqG7m3Q9w6XU9F6V`). New training waits for the reviewer's PASS on the pin, which the coordinator relays. Leave r1, e4, `swarm/exploration` and `main` alone.

For the reviewer only, append:

> Your first task is the pin check (DECISIONS D3). Report PASS, CONCERN or BLOCK to the coordinator.

## Thread and branch assignments

Threads 2–4 were started by the project coordinator at 03:58Z; 1 and 5 received the launch text from this thread at 03:56Z. Branches are filled in as the workers push.

| # | thread id | designated branch |
|---|---|---|
| 1 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTU8s1uzfPigWN1q5DxvtmMS` | `claude/project-thread-xb3nfz` |
| 2 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTAaEVfAVzHaVKDX6fqHWFyJ` | |
| 3 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTDWFArk7gDAQdi4r2c532Ui` | |
| 4 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTHq7m8eNJQc51iKX371EFA6` | |
| 5 | `cmsg_01APdvkRg6kVtZy2e9JoDcdTAKMwUsYk2djqx7Pz1rukHx` | |
