# Size/time lane: expedition e1

Session label `st-cloud-1`. Branch `claude/project-thread-j8mgkm` (this session's designated branch, used in place of `swarm/size-time`). Started from `swarm/exploration` at `c7c1c2ad8ac0bb23c9d0f84f9f1c9438764d03c6`, whose shared code is identical to Stage 0 `ebd874e17028f2c2f04de5ecea9fe4b51b77e425`.

## Claim (committed before execution)

This lane claims the 21 plan entries with `owner: size-time`, queue positions 4-24, and runs them in queue order with two workers:

```
e1-st-add-w16-i30001-a1   e1-st-add-w256-i30002-a1  e1-st-add0-w128-i30000-a1
e1-st-shuf-w128-i30000-a1 e1-st-add-w256-i30001-a1  e1-st-add0-w128-i30001-a1
e1-st-add-w64-i30002-a1   e1-st-add-w128-i30001-a1  e1-st-add0-w128-i30002-a1
e1-st-add-w128-i30002-a1  e1-st-add-w16-i30000-a1   e1-st-add-w16-i30002-a1
e1-st-add-w64-i30000-a1   e1-st-add-w32-i30002-a1   e1-st-add-w512-i30002-a1
e1-st-add-w512-i30001-a1  e1-st-add-w64-i30001-a1   e1-st-shuf-w128-i30001-a1
e1-st-add-w256-i30000-a1  e1-st-shuf-w128-i30002-a1 e1-st-add-w32-i30001-a1
```

Status events go to [registry.jsonl](registry.jsonl) (append-only, written by `launch_runs.py`).

## Execution provenance

Runs execute from a separate Git worktree checked out (detached) at `c7c1c2a`, so every run's `config.json` records that exact commit and a clean tree, while lane commits (this file, registry, figures, published runs) go to the branch from the main checkout. The registry path passed to the launcher is the main checkout's `research/size-time/registry.jsonl`.

Environment: uv 0.12.23 (the installed 0.11.32 had no CPython 3.12.14 download), `uv sync --locked --group analysis`. `uv run --locked python -m unittest test_exploration test_instrumentation`: 9 tests OK on this host before any run.
