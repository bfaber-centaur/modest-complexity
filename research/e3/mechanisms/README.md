# e3 mechanisms: figures first

Question, arms, matching rule and expected outcomes: [QUESTION.md](QUESTION.md) (`5bfa578`). Assignment: DECISIONS D5. Session `e3-me-cloud-1`, branch `claude/project-thread-psralm`. The start files passed the audit gate (D10); the continuation packet below is not audited yet. Reading: [INTERPRETATION.md](INTERPRETATION.md).

## Start files and the update-0 report (before any continuation update)

`edits.py` (committed at `371ed43`, run from that clean commit) writes `starts/<run_id>.npz` (np.savez, U, V, W only, float64) and [starts/update0.csv](starts/update0.csv) / [update0.json](starts/update0.json). Each source checkpoint is checked against the plan's SHA-256 and through `load_run`, which checks manifest hashes, the split hash and a replay of the logged metrics (maximum difference 0.0). The arm A arrays are byte-identical to the source's U, V and W.

Update 0 is measured after the edit and before any update. Displacement is ‖Δ‖/‖source‖ per matrix, and gate flips are over all 961 pairs × 128 units.

| run | displacement U / V / W | kept energy U / W | gate flips | RMS logit change | train loss / acc | common acc (483) | own cal+audit acc |
|---|---|---|---|---|---|---|---|
| keep i30000 / i30001 / i30002 | 0 | 1 / 1 | 0 | 0 | 3.5–3.7e-6 / 1.000 | 0.000 / 0.000 / 0.004 | 0.000 / 0.000 / 0.003 |
| filt i30000 | 0.761 / 0.764 / 0.797 | 0.42 / 0.37 | 0.288 | 4.08 | 1.13 / 0.689 | 0.431 | 0.437 |
| filt i30001 | 0.739 / 0.740 / 0.779 | 0.45 / 0.39 | 0.276 | 4.00 | 0.96 / 0.732 | 0.540 | 0.562 |
| filt i30002 | 0.737 / 0.737 / 0.779 | 0.46 / 0.39 | 0.275 | 3.98 | 0.90 / 0.789 | 0.580 | 0.607 |
| sham i30000 | 0.764 / 0.761 / 0.751 | 0.42 / 0.44 | 0.285 | 3.70 | 1.92 / 0.431 | 0.010 | 0.012 |
| sham i30001 | 0.741 / 0.738 / 0.719 | 0.45 / 0.48 | 0.275 | 3.52 | 1.74 / 0.482 | 0.010 | 0.008 |
| sham i30002 | 0.736 / 0.738 / 0.709 | 0.46 / 0.50 | 0.281 | 3.55 | 1.82 / 0.425 | 0.010 | 0.009 |

The filt rows reproduce e2 packet 2 exactly (0.431 / 0.540 / 0.580).

The sham is **input-energy-matched**, not matched:
- **Input.** Total kept input energy equals the filter's within 0.06% (λ = 1.78, 2.07, 2.09; seeds 37000–37002), so the input displacements agree within 0.003.
- **Gate flips.** These come out close (0.275–0.288 against 0.275–0.285), although the rule does not target them.
- **Mismatches.**
  - W is displaced less by the sham: 0.71–0.75 against 0.78–0.80.
  - The logit change is smaller: 3.5–3.7 against 4.0–4.1.
  - The immediate training damage is larger: train accuracy 0.43–0.48 against 0.69–0.79, train loss 1.7–1.9 against 0.9–1.1.
- **Per unit.**
  - In 37–47 of 128 units the dominant frequency outweighs all the others, so those units cannot be matched.
  - Per-unit kept energy against the dominant energy has median 1.6–1.7, with 5–95% range 0.22–2.3.
  - The median unit keeps 12–14 of its 14 non-dominant frequencies, so for most units the sham amounts to removing the dominant frequency.

Checks: `test_edits.py` passes 4 tests.
- The projection reproduces e2 `filter_units`.
- Sham orders are seeded and never include the dominant frequency.
- The stopping rule works on a hand-made case.
- The sham removes the dominant frequency from U, V and W, keeps the DC, and matches the total input energy.

Reproduce: `uv run --locked --group analysis python research/e3/mechanisms/edits.py`. The start files' SHA-256 values are in `update0.csv`. `np.savez` stamps the zip with the write time, so a rerun gives equal arrays but a different file hash; the committed files are the ones claimed.

## Continuation packet 1: the nine runs (2026-10-22)

**Runs.** All nine runs completed 20,000 updates; none failed or was capped. Each took 34–35 s. They were launched with two workers, arms and seeds interleaved, between 00:28:03 and 00:30:56 UTC (registry [registry.jsonl](registry.jsonl)). The producing commit is `59c822e`; its shared code (`experiment.py`, `launch_runs.py`, `publish_run.py`, `pyproject.toml`, `uv.lock`) is identical to the `9041a48` pin. The recorded tree status lists only untracked lane files: the registry the launcher was writing, and in five runs the plotting script, which was not yet committed. Published under [runs/](runs/) are checkpoints 0, 10, 30, 100, 300, 1000, 2000, 5000, 10000, 15000 and 20000, plus the first update with train accuracy ≥ .99; the others are listed with hashes in each manifest.

**Checks** (`continuation.py`, every run, before any measure):
- Published files match their manifests.
- `config.json` matches the plan: start-file hash, `init_age` 50000, 20,000 steps, lr, wd and width.
- Step-0 weights equal the start file exactly.
- The step-0 logged metrics replay with a maximum difference of 0.0.
- The split hash is the plan's.
- The 483 common pairs do not overlap the training pairs.

`test_continuation.py` passes 3 tests on the start files: update-0 values equal `update0.csv`, the frequency measures are right for each arm, and the pairwise and σ measures are unchanged when W is scaled by a positive factor.

**Figures.** [figures/continuation.png](figures/continuation.png) shows held-out and train accuracy, train cross-entropy, P(a+b over N) and distance. [figures/structure.png](figures/structure.png) shows frequency structure, P(S over N) and the σ-margin. Both have SVG versions. Each cell is split in two. The left panel is what the edit did: the 50k source, then the edited weights at update 0. The right panel is what training did afterwards, over updates 10–20,000 on a log axis. Seeds are separate columns, with no pooling and no smoothing.

Data: [results/checkpoints.csv](results/checkpoints.csv) has 9 source rows and 104 checkpoint rows. [results/dense.csv](results/dense.csv) is every 10 updates, from metrics.jsonl. [results/inputs.json](results/inputs.json) holds the hashes, commits and command. Analysis commit `94b9626`, run from a clean tree.

Reproduce:
```
uv run --locked --group analysis python research/e3/mechanisms/continuation.py
uv run --locked --group analysis python research/e3/mechanisms/plot_continuation.py
```

**Accuracy on the 483 common pairs** ("50k" is the source before the edit; the other columns are continuation updates), with three other measures at update 20,000, where B and C are still fitting:

| run | 50k | 0 | 10 | 100 | 300 | 1000 | 5000 | 20000 | P(a+b>N) / P(a+b>S) at 20k | σ-margin over N at 20k | input concentration at 20k | units with update-0 dominant freq. at 20k | distance from update 0 / from 50k at 20k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| keep i30000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.69 / 0.14 | −0.44 | 0.40 | 0.99 | 0.07 / 0.07 |
| keep i30001 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.72 / 0.15 | −0.32 | 0.44 | 0.98 | 0.06 / 0.06 |
| keep i30002 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.004 | 0.006 | 0.74 / 0.18 | −0.29 | 0.45 | 0.97 | 0.10 / 0.10 |
| filt i30000 | 0.000 | 0.431 | 0.538 | 0.735 | 0.621 | 0.240 | 0.025 | 0.031 | 0.88 / 0.45 | 0.41 | 0.61 | 0.91 | 1.15 / 0.96 |
| filt i30001 | 0.000 | 0.540 | 0.644 | 0.789 | 0.679 | 0.217 | 0.021 | 0.031 | 0.87 / 0.44 | 0.32 | 0.60 | 0.90 | 1.11 / 0.92 |
| filt i30002 | 0.004 | 0.580 | 0.700 | 0.863 | 0.739 | 0.267 | 0.027 | 0.010 | 0.87 / 0.39 | 0.22 | 0.59 | 0.92 | 1.14 / 0.92 |
| sham i30000 | 0.000 | 0.010 | 0.008 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.67 / 0.15 | −0.51 | 0.40 | 0.27 | 1.32 / 0.85 |
| sham i30001 | 0.000 | 0.010 | 0.002 | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.70 / 0.15 | −0.36 | 0.42 | 0.19 | 1.25 / 0.76 |
| sham i30002 | 0.004 | 0.010 | 0.006 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.66 / 0.15 | −0.50 | 0.39 | 0.23 | 1.19 / 0.71 |

Chance is 1/31 ≈ 0.032.

### Observations

1. **A, the optimizer reset alone, moves almost nothing.**
   - Common accuracy stays at 0.000 / 0.000 / 0.004–0.006.
   - Train cross-entropy stays at 3.5–3.7e-6.
   - P(a+b over N) changes by at most 0.003.
   - Distance from the start reaches 0.06–0.10 by update 20,000.
2. **B, first ~100 updates: training raised held-out accuracy above what the edit gave.** These are two separate effects:
   - The edit itself took common accuracy from 0.000 / 0.000 / 0.004 to 0.431 / 0.540 / 0.580 at update 0, before any update.
   - Training after the edit then added another 0.25–0.30, reaching 0.735 / 0.789 / 0.863 at update 100. The edit did not produce this rise.
   - Dense audit accuracy (544 pairs, which include the 483) peaks at 0.746 / 0.798 / 0.884 at updates 80 / 70 / 70.
   - That is just after train accuracy first reaches 1.0 (updates 70 / 50 / 40), when train cross-entropy is about 0.2.
3. **B, afterwards: the gain is then erased.**
   - Common accuracy is 0.62–0.74 at update 300, 0.22–0.27 at 1000, 0.02–0.03 at 5000 and 0.010–0.031 at 20,000, which is chance.
   - Audit accuracy drops below its update-0 value at updates 490–600, and below half of it at 880–1090.
   - After update 5000 it never exceeds 0.053.
   - Meanwhile train cross-entropy keeps falling, from 0.2 at update 100 to 3.5–3.7e-5 at 20,000, parallel to C's curve and slightly below it.
   - **Update 20,000 is not an end state.** B and C are still fitting there: their train cross-entropy (B 3.5–3.7e-5, C 5.0–5.4e-5) is about 10–15× A's 3.6e-6 and still falling steeply. The held-out fall was complete by about 5,000 updates, long before that.
4. **C shows no held-out accuracy at any point.**
   - Common accuracy is 0.010 at update 0 and 0.000–0.002 from update 100 on.
   - Train accuracy recovers to 1.0 by updates 100–110, later than B's 40–70.
5. **B's frequency structure partly survives the erasure.**
   - Input concentration falls from 1.0 to 0.59–0.61, still above A's 0.40–0.45.
   - 90–92% of live units keep their update-0 dominant frequency.
   - The units' share at that frequency is 0.58–0.60.
6. **In C, the removed frequency partly grows back.**
   - The share at each unit's 50k dominant frequency rises from about 0 to 0.24–0.31.
   - 73–81% of units change dominant frequency.
   - Input concentration returns to 0.39–0.42, about A's level.
7. **At update 20,000, B keeps a rank preference for a+b over unrelated labels, but not over operand-associated ones.**
   - P(a+b over N) is 0.87–0.88, against 0.69–0.74 for A and 0.66–0.70 for C.
   - The median σ-margin over N is +0.22 to +0.41, against −0.29 to −0.44 for A and −0.36 to −0.51 for C.
   - P(a+b over S) is only 0.39–0.45 (A 0.14–0.18, C 0.15). a+b is ranked below most labels seen with an operand in training, so accuracy is at chance.
   - P(S over N), the preference for operand-associated over unrelated labels, also comes back. The edit lowered it from 0.94–0.95 to 0.58–0.59 in B and 0.75–0.76 in C. By update 1000 training had restored it to 0.91–0.92 in B and 0.96–0.97 in C; at 20,000 it is 0.92–0.95 in all nine runs.
8. **Neither edited branch returns to its source.**
   - At update 20,000, B is 1.11–1.15 from update 0 and 0.92–0.96 from the 50k source.
   - C is 1.19–1.32 from update 0 and 0.71–0.85 from the source.
9. **The three seeds agree.** All three B branches rise and then erase, with peak and erasure times within about 100 updates of each other. All three C branches stay at chance.

### Limitations

- **Replicates.** There are three checkpoints from one split at one width. They share their evaluation pairs, so pairs are not independent replicates. There is no rerun of any run.
- **The reset.** Every branch includes the optimizer reset. A shows the reset alone does not move a converged memorizer, but B and C mix recovery from the edit with fresh-moment Adam steps.
- **The sham.** C is input-energy-matched only. Its immediate damage is larger (train accuracy 0.43–0.48 against 0.69–0.79) and its W displacement smaller. Its later train recovery means C and B differ in more than which frequencies were kept.
- **Measurement density.**
  - The common-set, preference and frequency measures exist only at 11–12 checkpoints per run.
  - Only train, cal and audit accuracy and loss are dense, every 10 updates.
  - The peak values above come from the dense audit log; the common-set values are at checkpoints.
- **Peak accuracy.** The B peak is an observation of a trajectory. It is not a method: choosing the update with the best held-out accuracy would be selection on held-out data.
- **Budget.** About 3.5 of the session's 25 minutes are used: the nine runs took 2.9 min elapsed, and the edits, tests and analysis seconds.

## Follow-up 1: weight decay 0 (DECISIONS D11, 2026-10-22)

The question and expectation are in [QUESTION.md](QUESTION.md#follow-up-1-decisions-d11-weight-decay-0), pushed at `ad6e923` before any wd-0 run.

**Runs.** There are six runs, `e3-me-<filt|sham>-wd0-w128-i<30000|30001|30002>-a1`. They use the same B and C start files and hashes as packet 1, with `--wd 0` and nothing else changed. All six completed 20,000 updates; none failed or was capped. Each took 34–35 s, between 00:38:53 and 00:40:37 UTC.

The producing commit is `ad6e923`, with shared code identical to the pin. The recorded tree status lists only lane files: the registry being appended, and in four runs the lane's analysis, plotting and notes files, which I was editing during the runs. `experiment.py` and the other shared files were unmodified. Checkpoints are published as in packet 1.

**Analysis.** `continuation.py` and `plot_continuation.py` were committed at `d156aec` and run from that commit; the recorded tree status lists only the output files being rewritten. The tables now carry `wd`, `branch` (for example `filt-wd0`) and the weight `norm`. The packet-1 rows of `checkpoints.csv` are unchanged cell for cell, and the packet-1 figures are pixel-identical.

New figure: [figures/weight_decay.png](figures/weight_decay.png) (and `.svg`). It shows B and C at wd 1 (solid, packet 1) and wd 0 (dashed), with the same layout as the packet-1 figures.

**Accuracy on the 483 common pairs**, wd 0 runs (wd 1 values from packet 1 in brackets), with three other measures at update 20,000:

| run | 0 | 100 | 300 | 1000 | 5000 | 20000 | train CE at 20k | P(a+b>N) / P(a+b>S) at 20k | input concentration at 20k | weight norm 0 → 20k |
|---|---|---|---|---|---|---|---|---|---|---|
| filt-wd0 i30000 | 0.431 | 0.758 (0.735) | 0.743 (0.621) | 0.692 (0.240) | 0.634 (0.025) | 0.561 (0.031) | 3.6e-8 (3.7e-5) | 0.99 / 0.90 | 0.87 (0.61) | 37.9 → 75.5 |
| filt-wd0 i30001 | 0.540 | 0.805 (0.789) | 0.778 (0.679) | 0.739 (0.217) | 0.660 (0.021) | 0.565 (0.031) | 3.2e-8 (3.5e-5) | 0.99 / 0.89 | 0.85 (0.60) | 39.2 → 75.9 |
| filt-wd0 i30002 | 0.580 | 0.872 (0.863) | 0.849 (0.739) | 0.812 (0.267) | 0.737 (0.027) | 0.619 (0.010) | 3.2e-8 (3.5e-5) | 0.99 / 0.92 | 0.87 (0.59) | 39.3 → 75.9 |
| sham-wd0 i30000 | 0.010 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 5.6e-8 (5.4e-5) | 0.55 / 0.08 | 0.31 (0.40) | 39.4 → 79.3 |
| sham-wd0 i30001 | 0.010 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.002) | 5.2e-8 (5.0e-5) | 0.50 / 0.06 | 0.27 (0.42) | 41.0 → 78.4 |
| sham-wd0 i30002 | 0.010 | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 0.000 (0.000) | 5.2e-8 (5.0e-5) | 0.50 / 0.06 | 0.27 (0.39) | 41.3 → 78.8 |

### Observations

1. **The early rise in B is the same at wd 0.**
   - Common accuracy at update 100 is 0.758 / 0.805 / 0.872, against 0.735 / 0.789 / 0.863 at wd 1.
   - Dense audit accuracy peaks at 0.767 / 0.812 / 0.886, at updates 100 / 110 / 80.
2. **Without decay, B keeps most of its gain through 20,000 updates.**
   - Common accuracy is 0.69–0.81 at update 1000, 0.63–0.74 at 5000 and 0.56–0.62 at 20,000. At wd 1 the same start files gave 0.22–0.27, 0.02–0.03 and 0.01–0.03.
   - Audit accuracy never falls below its update-0 value.
   - B at wd 0 still declines slowly and steadily on the log axis, by 0.20–0.25 from its peak over 20,000 updates.
3. **B at wd 0 fits the training pairs harder, not less.**
   - Its train cross-entropy at 20,000 is 3.2–3.6e-8, about 1000× lower than at wd 1, and it is still falling.
   - The weight norm doubles, from 38–39 to 75–76. At wd 1 it stayed near 38–41 for about 1,000 updates and then grew to 57.
4. **B's structure and preferences hold at wd 0.**
   - All live units keep their update-0 dominant frequency. Input concentration is 0.85–0.87 at 20,000, against 0.59–0.61 at wd 1.
   - P(a+b over N) stays at 0.99, and P(a+b over S) is 0.89–0.92 at 20,000 (wd 1: 0.39–0.45).
   - P(S over N) still rises from 0.58 to 0.87, short of the 0.92–0.95 seen at wd 1 and in A.
5. **C at wd 0 stays at chance throughout** (0.000 from update 100 on).
   - Its P(a+b over N) drifts down to 0.50–0.55, and P(a+b over S) to 0.06–0.08.
   - The removed frequency grows back less than at wd 1: share 0.11–0.13 against 0.24–0.31.
   - Input concentration stays at 0.27–0.31.
6. **Distance does not track retention.** At update 20,000, B at wd 0 is 1.12–1.19 from its update-0 weights, about as far as B at wd 1 (1.11–1.15), yet it kept its held-out accuracy.
7. **The three seeds agree** on every point above.

### Limitations

- **Scope.** These are the same three start files, the same split and the same evaluation pairs as packet 1. They are not new replicates of the sources.
- **20,000 is not an end state.** At wd 0 every run is still fitting, with train cross-entropy still falling and the norm still growing. B's slow decline might continue past 20,000, so "keeps most of its gain" holds only up to 20,000 updates.
- **Decay was the only change, but its effects are coupled.** Turning it off also changes the weight norm, the effective step size relative to the weights and the logit scale. The accuracy and pairwise measures are scale-free; the train cross-entropy is not.
- **Budget.** The follow-up used about 1.8 min elapsed for runs and seconds for analysis. The session total is about 5.5 of 25 min.
