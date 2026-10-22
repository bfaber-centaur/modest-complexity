# Adaptive choices (generalization lane)

Every choice made after seeing an outcome, in order, with the time and what it was based on. Choices fixed in QUESTION.md before any evaluation are not repeated here.

| time (UTC) | choice | based on |
|---|---|---|
| 04:05 | P2 reference read from `recovered/` instead of the project mount (QUESTION.md amended) | coordinator committed the P2 checkpoint at 8fca55f so the reference is reproducible from Git; before any evaluation |
| 04:07 | Frozen evaluation re-run as a2 with an explicit step map `[1, 2, 3, 3, …]` | a1 failed: c2rnn's default map rejects tied/tied models past length 3 (pin note D10); tied groups ignore the map |
| 04:10 | Figure annotations moved; legend wording | visual inspection of the exported PNG (label collisions) |
| 04:19 | Calibration checkpoint not committed (hash kept in its MANIFEST) | not an input to any analysis; keeps the lane's committed checkpoints to the 16 research runs |
| 04:21 | Follow-up chosen: frozen first-error anatomy (F1, F2) | Part A and B showed a gradual decline with length in every tied/inter model; the charter's optional follow-up targets the first failing length |
| 04:22 | Follow-up amended with the held-out-window measure and F3 before it ran | reviewer G1 on Part A (programs with held-out 3-op windows are less accurate) |
| 04:23 | Both figures labelled so the held-out-only length-3 point is the like-for-like comparison | reviewer G2 |
| 04:26 | Original-task ¬N trajectories in the replication figure scored from r1's logged prediction strings | r1's metrics do not log N/¬N; the strings determine it exactly |
