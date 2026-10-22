# c2 sharing lane: adaptive choices

Every choice made after [QUESTION.md](QUESTION.md) was committed, with its time (UTC, 2026-10-22), what it was based on, and whether any c2-share outcome had been seen. Pre-registered content stays in QUESTION.md.

| time | choice | based on | outcomes seen? |
|---|---|---|---|
| 04:10 | Recipe flags (`--task-id`, `--hidden`, `--lr`, `--wd`, `--eval-every`) written explicitly into every plan argv; values unchanged from the defaults | reviewer pin check C1, relayed by the coordinator | no |
| ~04:35 | Follow-up chosen: matched no-decay comparison (all four cells at wd 0, 12 runs), by the indicators declared in QUESTION.md; proposed in QUESTION.md "Follow-up" and appended to plan.json, not launched | initial comparison: group diagnostics (decay/Adam step ratio and cosine after the fit; √v̂ < eps fraction) and the endpoints | yes, all initial-comparison outcomes |
| ~04:30 | Copying the lane's local checkpoints to the project-files mount was attempted and refused by the session's permission check; checkpoints other than the committed UT/UU `ckpt-10000` stay local and are regenerable (about 75 s per run) | — | — |
| ~04:42 | Follow-up approved by the coordinator as committed (wd 0, all four cells, 12 runs); launched with the explicit recipe flags already in plan.json. Write-up note from the coordinator: wd 0 is a different recipe, so a change in the tying effect says only that the effect depends on decay under this recipe | coordinator relay | yes |
