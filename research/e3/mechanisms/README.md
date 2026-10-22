# e3 mechanisms: figures first

Question, arms, matching rule and expected outcomes: [QUESTION.md](QUESTION.md) (`5bfa578`). Assignment: DECISIONS D5. Session `e3-me-cloud-1`, branch `claude/project-thread-psralm`. Nothing here is audited yet.

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
