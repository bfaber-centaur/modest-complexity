# e4 audit

Session `e4-au-cloud-1`, thread "Correctness audit", branch `claude/project-thread-id1tmm`. The base is `swarm/exploration` `52e7879` (D3), with D11 and D12 merged. Questions and rules were written before any measurement: [QUESTION.md](QUESTION.md) (`33bf731`; addenda 1 `2d81d64`, 2 `4e459bf`, 3 `8a39ee1`, 4 `0c45b39`).

Software results are PASS / FAIL / NOT CHECKED. Readings are SUPPORTED LOCALLY / AMBIGUOUS / CONTRADICTED.

| QUESTION section | status |
|---|---|
| 1. Control C1: random-feature starts through the B protocol | done, below |
| 2. Mechanisms decomposition (M1–M4, R1–R4, C2; addenda 1–2) | done, below (lane PR #35 at `52f25d0`) |
| 3. Dynamics runs (Y1–Y7), decay-off runs (D13), restricted-decay patch and its runs (D15; addendum 3) | done, below (PR #33, #36, #37, #40) |
| Control X0: does the exempt-reference rescue need B's start? (addendum 4, D16, 12 runs) | done, below |

## Control C1: does the rise and collapse need anything the failed runs learned? (QUESTION §1)

![c1](figures/c1.png)

Vector version: [c1.svg](figures/c1.svg).

Look before reading:
- Top left (wd 1): do the orange curves (random-feature starts) rise and fall like the blue ones (B)? Where do the triangles and crosses sit?
- Top right (wd 0): where do the two colours end?
- Bottom right: compare the end of the filled orange and blue lines with the grey ones (the unedited A).

**The control.** Each init seed gets one start file, written by [rf_starts.py](rf_starts.py) before any run (`aa09dd4`):
- the untrained U and V of that init;
- a closed-form ridge readout fitted to the 299 training labels;
- the same per-unit dominant-frequency filter that made B;
- U and V rescaled jointly, and W separately, to B's matrix norms. That is a scale-only change, so it changes no answer.

So this start has no trained features. The only trained content is a readout fitted to the same 299 labels as B's sources. Each start was continued with the e3 B recipe (fresh Adam, lr 1e-3, 20,000 updates) at wd 1 and wd 0. These are the claimed runs `e4-au-rf[-wd0]-w128-i{30000,30001,30002}-a1` (D11, D12).

**Results, per seed.** Common accuracy is on the 483 never-trained pairs at published checkpoints. The peak and the decline times use own held-out accuracy every 10 updates, with the dynamics lane's definitions.

| init | arm | common acc, update 0 → 100 | rise to 100 (class) | first update with train acc 1 (wd 1) | own held-out peak (update), wd 1 | wd 1: first below update-0 level / below 0.10 | wd 1 common acc at 20k | wd 0 crossings | wd 0 common acc at 20k |
|---|---|---|---|---|---|---|---|---|---|
| 30000 | random features | 0.195 → 0.580 | 0.385 (generic) | 140 | 0.624 (240) | 1,330 / 2,060 | 0.054 | none (censored > 20,000) | 0.561 |
| 30000 | B | 0.431 → 0.735 | 0.304 | 70 | 0.751 (80) | 600 / 1,680 | 0.031 | none (censored > 20,000) | 0.561 |
| 30001 | random features | 0.143 → 0.596 | 0.453 (generic) | 140 | 0.645 (250) | 2,000 / 2,910 | 0.035 | none (censored > 20,000) | 0.621 |
| 30001 | B | 0.540 → 0.789 | 0.248 | 50 | 0.807 (70) | 470 / 1,540 | 0.031 | none (censored > 20,000) | 0.565 |
| 30002 | random features | 0.277 → 0.646 | 0.369 (generic) | 140 | 0.693 (200) | 1,140 / 2,250 | 0.046 | none (censored > 20,000) | 0.609 |
| 30002 | B | 0.580 → 0.863 | 0.284 | 40 | 0.878 (70) | 470 / 1,800 | 0.010 | none (censored > 20,000) | 0.619 |

**Re-crossings.** The random-feature i30001 wd 1 run re-crosses both thresholds once. The i30002 wd 1 run re-crosses 0.10 once. B re-crosses neither.

**At 20,000 updates, wd 1, scale-free measures.**

| arm | P(a+b over S) | P(a+b over N) | P(S over N) | σ-margin over N |
|---|---|---|---|---|
| random features | 0.45–0.46 | 0.88–0.89 | 0.91 | 0.01–0.08 |
| B | 0.39–0.45 | 0.87–0.88 | 0.92–0.93 | −0.04–0.11 |
| unedited A | 0.14–0.18 | 0.70–0.75 | 0.94–0.95 | −0.60 to −0.45 |

At wd 0 the two filtered arms also end alike: P(a+b over S) 0.89–0.93 and σ-margin 1.38–1.70.

**Verdict under the pre-stated rule: G is SUPPORTED LOCALLY**, in all three seeds and all three components.
- **Rise.** The random-feature rise to update 100 is 0.37–0.45, more than B's 0.25–0.30 and far above the half-of-B threshold (0.12–0.15).
- **Collapse.** Every wd 1 random-feature run falls below its update-0 level (1,140–2,000) and below 0.10 (2,060–2,910).
- **Hold.** Every wd 0 random-feature run is censored at both thresholds.

**What stays specific to B** (observations, three seeds):
- It starts 0.24–0.40 higher.
- It peaks 0.13–0.18 higher in own held-out accuracy (0.75–0.88 against 0.62–0.69).
- It fits every training pair two to three times sooner (40–70 updates against 140), and it peaks and collapses earlier in updates.
- By 20,000 updates, at either wd, nothing listed above separates the arms. That includes the "rank trace", P(a+b over S) about 0.4–0.46 at wd 1.

**Reading (interpretation, three seeds, one split).**
- **The phenomenon is supplied.** The rise, the collapse under decay 1, the retention without decay and the residual rank preference do not need anything the failed runs learned. Untrained features, a readout fitted to the same 299 labels, and the filter are enough. So the computation between B's peak and its collapse is supplied by the filter plus fitting.
- **What a mechanism found in B describes.** It describes filtered, frequency-concentrated networks trained on these pairs with AdamW. It is not a property of the failed runs. The failed runs contribute a head start: more correct answers at the edit, a higher peak and a faster fit.
- **The rank trace.** e3 read the trace that survives the collapse against the unedited and sham arms. It is equally present when the start came from random features, so it marks the filtered start, not something inherited from the failed run.

**Not shown.**
- That the two arms collapse by the same mechanism. Only their trajectories in scale-free measures were compared.
- That the failed runs contain nothing. Their head start is real in all three seeds.
- Anything beyond three init seeds, one split, width 128 and this recipe.

**Strongest surviving alternative.**
- Two different mechanisms could produce such similar curves.
- In particular, B's decline could start from the structure it inherited from the failed run, while the random-feature start declines from structure built in its first 200 updates.
- The random-feature start's later peak (200–250 against 70–80) and later crossings (2,060–2,910 against 1,540–1,800 for 0.10) are consistent with either.

**Cheapest discriminating check.** Run the mechanisms lane's channel decomposition on these six runs at the same published updates (0, 10, 30, 100, 300, 1,000, 2,000, 5,000, 10,000, 15,000, 20,000). This is analysis only.
- If the same channels change in the same order, the mechanism belongs to the filtered network.
- Where B and the random-feature start differ, that difference is what the failed runs contribute.

**Limits.**
- **Concentration.** The random-feature units keep less of their energy under the filter (U 0.19, W 0.085, against B's 0.42–0.46 and 0.37–0.39). This works against G, so the G verdict is conservative.
- **Timing of the rise.** It was measured at update 100, before the random-feature start's own fit (140) and peak (200–250). Measured to its first fit checkpoint (110–140), the rise is 0.39–0.48. This also works against G.
- **Ridge strength.** Only one value was used (r = 1e-6). Its choice is explained in QUESTION.md and was not made on held-out accuracy.

**Software checks.** All PASS. Sources: [starts/update0.json](starts/update0.json) and [results/c1-check.json](results/c1-check.json).
- **Start files.**
  - U and V equal `experiment.init` for the seed, and every published step-0 checkpoint of that init at width 128 that `load_run` accepts: B's source for 30000 and 30001, and the e1 no-decay run for all three. B's i30002 source has no step-0 file in the repository.
  - The filtered network reproduces e2 audit cycle 3 (0.195 / 0.143 / 0.277).
  - Rescaling leaves the argmax on all 961 pairs, and every scale-free measure, unchanged to within 1.1e-15. The norms equal B's to 1e-12 relative.
  - The split hash is `aa8f7504…`, and the common pairs do not overlap training.
- **Provenance**, for each of the six runs:
  - the config's start-file hash equals the plan's and the file's, and the step-0 weights equal the start file bitwise;
  - wd, lr and steps are as planned, `init_age` is 0 and there is no resume;
  - the registry has one running and one complete event, and 20,000 updates completed;
  - the published files match their manifests;
  - the step-0 logged accuracies equal my forward pass (maximum difference 0.0).
- **Producing code.**
  - The producing commit is `ccb4e16`. Its shared code is identical to `72cfb5d` (`git diff` is empty for `experiment.py`, `launch_runs.py`, `report_map.py`, `publish_run.py`, `instrumentation.py`, `pyproject.toml` and `uv.lock`).
  - The configs differ from B's only in the start file and its hash, `init_age`, run ID, output path, session label and runtime record.
- **Incidental, on existing runs.** My decline times for B reproduce the dynamics lane's addendum (`e5fd1c2`) exactly: 600 / 470 / 470 and 1,680 / 1,540 / 1,800 at wd 1, censored at wd 0.

**Budget.** About 2.0 of 25 min:
- six runs, 108 s elapsed with two workers (one BLAS thread each, launched 02:17:10–02:18:58 UTC);
- start files and analysis, about 12 s.

**Reproduce**
```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
uv run --locked python research/e4/audit/rf_starts.py
uv run --locked python launch_runs.py --plan research/e4/plan.json --root runs/e4 --registry research/e4/audit/registry.jsonl \
  --session-label e4-au-cloud-1 --workers 2 --timeout 300 --ids e4-au-rf-w128-i30000-a1 e4-au-rf-wd0-w128-i30001-a1 \
  e4-au-rf-w128-i30002-a1 e4-au-rf-wd0-w128-i30000-a1 e4-au-rf-w128-i30001-a1 e4-au-rf-wd0-w128-i30002-a1
for r in runs/e4/e4-au-rf-*-a1; do uv run --locked python publish_run.py $r research/e4/audit/runs --steps 0,10,30,100,300,1000,2000,5000,10000,15000,20000; done
uv run --locked python research/e4/audit/c1_check.py research/e4/audit/results
uv run --locked --group analysis python research/e4/audit/plot_c1.py research/e4/audit/figures/c1 research/e4/audit/results
```

## Mechanisms channel split (QUESTION §2, addenda 1–2; lane packet PR #35, `52f25d0`)

![channels](figures/channels.png)

Vector version: [channels.svg](figures/channels.svg).

Look before reading:
- Left: how far do the thick lines (reference channel alone) sit above the thin ones (full network) after 2,000, compared with the grey lines (A) and the crosses (a permuted basis)?
- Middle: do orange (random-feature start) and dark blue (B) channels move together?
- Right: what happens to the share the decayed start can make up (●), and to the direction (□)?

**What was run.** [channels_check.py](channels_check.py) is my implementation of the lane's design (`d584af7`). It covers every published state of B (wd 1 and 0), A, C (wd 1 and 0) and the six C1 runs. [lane_check.py](lane_check.py) runs the lane's own `channels.py`:
- against mine;
- on the C1 runs, which the coordinator assigned to me (D14 item 4);
- for the basis checks of addendum 2.

The rules were fixed before computing: addendum 1 (`2d81d64`) and addendum 2 (`4e459bf`). When addendum 2 was written, my own channels had already been computed (it says so).

**Software checks, all PASS.**

| # | check | result |
|---|---|---|
| M1 | Reconstruction identity | offset + reference + off-reference = label-centred logits to 1.7e-15 relative, in every state (mine). The lane's `channels()` equals mine to 7.3e-16 on 175 states (B, A, C1). The label-axis DC is a per-pair constant (spread 2e-16). No unit is dead at update 0, so the lane's live-unit rule changes nothing |
| M2 | Basis and labels | k_j read from the update-0 weights equals e2 `dominant()` of the 50k source (the frequency the filter kept) for all three sources; the wd 0 start gives the same k_j. μ uses training inputs only. Task-informed: the lane's alignment and coherence (a+b−c mode, over all 961 pairs including training pairs) |
| M3 | Invariances | Accuracy, P measures and the σ-margin change by at most 6.7e-16 under ×3.7, under a random per-pair constant, and under (2U, 2V, 3W). Linear margins scale by exactly 6 under the last. The lane's channel margins (logits) and weight-part norms are scale-dependent. Its accuracies and P measures are not |
| M4 | Decay is a uniform factor | One `experiment.adam` update with zero gradient gives exactly (1 − lr·wd)·w for wd 1, 0.3, 0.1 and 0 (bitwise). −log ρ_t = E(t) to 1.4e-14 |

**The lane's readings.**

| lane reading | verdict | evidence |
|---|---|---|
| The label offset (Janati's mean term) is not involved in the decline | SUPPORTED LOCALLY | From update 100 on, removing μ·W changes accuracy by at most 0.041 in B and 0.031 in C1 (from 300). Its mean margins are at most 0.017 of the RMS logit. At update 0, removing it lifts accuracy by up to 0.23 in B and in C1. So a channel with near-zero mean margin can move the argmax a lot (R4) |
| "Outvoted, not erased": the reference channel alone still gets about half the common pairs right after the collapse | SUPPORTED LOCALLY as a statement about the readout at the reference frequencies, with three qualifications (below) | At 5,000 and 20,000, B's reference-only accuracy is 0.47–0.52, against A's 0.16–0.28. That is 0.22–0.36 above A in every seed, and the normalised margins do not overlap (C2 rule met). Permuting k_j across units gives 0.034–0.047 (median of 8, max 0.077), so the basis does not manufacture it |
| What decay changes, compared with wd 0, is mostly the reference part | AMBIGUOUS: it depends on the scale | In raw logits the wd 1 − wd 0 difference at 2,000 is 3.5–3.8 in the reference channel and 0.6–0.7 in the off-reference one (lane). But the wd 0 networks are 1.65× larger in ‖(U, V)‖·‖W‖ (1,372–1,409 against 836–851), which inflates their raw margins. Per unit of that product (a scale no channel enters, post hoc), the differences are 1.6–1.7 (reference) and 1.3 (off-reference). The off-reference channel is then 2.5× larger at wd 1. Normalising by the RMS logit instead couples the channels: any growth in one shrinks the others |
| The useful component's loss is not mainly weight shrinkage, since its input weights keep their norm | AMBIGUOUS | The norms hold, but decay acts on those weights exactly on schedule (M4: ρ = 0.15 from 100 to 2,000). They keep their norm because the gradient replaces them. On the readout side, the decayed start can be at most 15–16% of ‖W_ref‖ at 2,000 and 0.8% at 5,000. Benchmarked against passive decay from update 100 (R1), the reference channel's S-margin at 2,000 is 26–36× larger. The channel is actively rebuilt, not passively shrinking |
| Phase alignment and coherence at the a+b mode: not seen | AMBIGUOUS: the index could not show it | The lane's index over all 961 pairs gives A (0% held-out) 0.98–0.99, as high as B. On the 483 common pairs, B's coherence goes 0.97–0.98 at 100 → 0.94–0.96 at 2,000 → 0.91–0.93 at 20,000, and A's is 0.81–0.85. That is a modest decline, not a large decoherence. A full-table projection onto a+b−c gives A implied held-out accuracy 1.0 over all pairs or over training pairs, and 0.0 over common pairs. A table-level a+b mode cannot show that a computation exists |

**Qualifications to "outvoted, not erased".**
1. **Not specific to the failed runs.** The lane's own `channels.py` on the six C1 runs gives the same channels in the same order, SUPPORTED LOCALLY under addendum 2(b) in all three seeds:
   - From 300 to 2,000 at wd 1, both channels fall, with the reference share 0.33 / 0.35 / 0.40 against B's 0.41 / 0.45 / 0.42.
   - The offset carries under 1%.
   - At wd 0 the reference channel rises.
   - The reference-only accuracy at 5,000–20,000 is 0.46–0.52, against B's 0.47–0.52.
2. **"Not erased" describes the direction, not the edited weights.** W_ref,t = ρ_t W_ref,0 + ΔW_ref,t exactly.
   - By 5,000 the decayed start is at most 0.76–0.82% of ‖W_ref‖ in every run (s_t < 0.05; addendum 2c), so the edited readout is gone.
   - The rebuilt readout still points along it: cosine 0.81–0.83 for B and 0.72–0.75 for C1 at 5,000, and c_t 0.62–0.71 ≫ ρ = 0.0067.
3. **The split is per unit, not per frequency (R3).** The set of reference frequencies covers all 15 bins in every run (B, A, C, C1), so the pre-stated "in K / outside K" split is degenerate. Every off-reference component sits at some other unit's reference frequency.
   - Only unit identity separates "competitor" from "useful signal on another unit".
   - Functionally the off-reference channel behaves like a memorizer: alone it scores 0.00–0.004 (lane), and its margin falls against S, not against N.

**Other checks.**
- **R2.** The off-reference channel is exactly zero at update 0 by construction and grows at both wd. The evidence for "competitors overtake" is the excess at wd 1 (2.5× by 2,000 per unit norm product), not the growth itself. The lane compares against wd 0, so this is met.
- **R4.** The exact bilinear split of the reference channel's S-margin change (B, wd 1, 100 → 2,000) depends on the order:
  - features carry 0.90–0.99 of it if they change first, and 0.43–0.55 if the readout changes first;
  - the readout carries 0.01–0.10 or 0.45–0.57.
  So at least about 43% of the reference channel's fall comes through h (input weights and gates). Readout and features cannot be separated further this way, consistent with the lane's "cannot tell which".

**Strongest surviving alternative.** One integrated re-fitting process, cut into two channels by the Fourier basis (the lane's own alternative). It is consistent with three findings above:
- under decay both channels are almost entirely gradient-built by 5,000;
- the rebuilt reference readout keeps the start's direction;
- the same split appears when the start came from random features.

**Cheapest discriminating check.** The lane's approved intervention (D14), restricting decay to or away from the reference parts. It tests whether decay acting on the reference slot is necessary, while the gradient keeps rebuilding it. My review of its shared-code patch is still to come.

**Not checked here.** The lane's gate-change measure, its weight-part norms, its dense replays (the coordinator and the lane checked replay equality) and its current-frequency variant.

**Budget.** About 15 s of analysis in total (channels_check 3.5 s, lane_check 3.7 s, each run twice). Cumulative about 2.3 of 25 min.

**Reproduce**
```
uv run --locked python research/e4/audit/channels_check.py research/e4/audit/results
uv run --locked --group analysis python research/e4/audit/lane_check.py research/e4/audit/results
uv run --locked --group analysis python research/e4/audit/plot_channels_audit.py research/e4/audit/figures/channels research/e4/audit/results
```

## Control X0: does the exempt-reference rescue need B's start? (QUESTION addendum 4, D16)

![X0](figures/x0.png)

**Why.** In the mechanisms intervention (PR #40), B with the reference parts exempt from decay (xref) reaches 0.998–1.000 on the common pairs at 5,000, where uniform wd 1 gives 0.02–0.03. The exemption is label-free, but its rule encodes the known Fourier structure of the task: one input frequency per unit, with the readout kept at the same frequency. X0 runs the same exemption from starts that lack B's structure. The rule was pushed at `0c45b39` before any run, and the coordinator claimed 12 runs (D16).

**Result.** Common-pair accuracy (483 pairs). Labels at 5,000: rescued ≥ 0.90, not rescued ≤ 0.30, otherwise intermediate.

| arm (start; projection) | seed | 2,000 | 5,000 | label | same start, uniform wd 1, at 5,000 |
|---|---|---|---|---|---|
| I (untrained init; its own k_j) | 30000 | 0.037 | 0.915 | rescued | 0.000 (e1 run) |
| | 30001 | 0.043 | 0.954 | rescued | 0.000 |
| | 30002 | 0.081 | 0.923 | rescued | 0.000 (own held-out; no checkpoint) |
| RF (C1 start; the init's k_j) | 30000 | 0.977 | 0.996 | rescued | 0.029 |
| | 30001 | 0.988 | 1.000 | rescued | 0.054 |
| | 30002 | 0.990 | 0.998 | rescued | 0.035 |
| A (unedited 50k start; B's files) | 30000 | 0.696 | 0.994 | rescued | 0.000 |
| | 30001 | 0.785 | 0.988 | rescued | 0.000 |
| | 30002 | 0.799 | 1.000 | rescued | 0.004 |
| S (B start; k_j permuted across units) | 30000 | 0.186 | 0.325 | intermediate | 0.025 (B) |
| | 30001 | 0.087 | 0.141 | not rescued | 0.021 |
| | 30002 | 0.126 | 0.217 | not rescued | 0.027 |
| B xref (the lane's runs, for reference) | 30000–30002 | 0.961–0.988 | 0.998–1.000 | | 0.021–0.027 |

**Software checks: PASS.**
- **Provenance.** Each run has one running and one complete event and 5,000 updates. Decay is on `rest`, and the projection hash equals the plan's and the file's. Start-file hashes match. The producing commit `63262e1` has the shared files of the pin `8108d8b`.
  - The recorded tree is dirty only in audit-owned files: the registry being appended, my check script (which I edited during the launch) and one result file. No shared code is involved.
- **Update 0** equals the start bitwise, or `independent.initial_weights` for I.
- **Replay.** My replay with my own projection reproduces all 30 logged rows to update 300 in every run.
  - Accuracies are identical, and the checkpoints at 100 and 300 agree to ≤ 1.5e-11 relative.
  - The A starts are the loosest: rows to 5e-10, checkpoints to 1.5e-11. This is the known logit-rounding difference at tiny train cross-entropy (the 50k start) and is within tolerance.

**Readings.**
- **I rescued, so "the 1.00 is built in" is SUPPORTED LOCALLY.**
  - From untrained weights and 299 training pairs, the exemption reaches 0.92–0.95 in 5,000 updates. Uniform wd 1 from the same init stays at 0.000.
  - It is slower than from the trained starts: 0.04–0.08 at 2,000, with the climb between 2,000 and 4,000.
  - So xref's 1.00 on B is a property of the intervention. It is not evidence of a computation present in B.
- **RF and A rescued.** Neither the failed runs' features nor B's edit is needed. A gets there from the unedited 50k start, which collapses to 0.000 under uniform wd 1.
- **S is mixed** (one intermediate, two not rescued). By 5,000, exempting a frequency that is not the unit's own has not rescued B, so the match between unit and frequency matters at this horizon.
  - All three S runs are still rising at 5,000 (bottom of the curve around 2,000–3,000), so a later rescue is not excluded.
  - The bottom row of the figure shows what S is: most of each unit's actual content decays (the decaying share falls only from 0.95 to 0.41–0.46), while the exempted frequency starts near empty.

**What X0 does to the lane's intervention readings.**
- The numbers are correct (X1–X5 below).
- The **onlyref** reading (decaying only the reference parts suffices for the collapse) is not touched by X0.
- The **xref** reading needs a qualifier. "Decay acting on the reference parts is necessary for most of the collapse" was read from f ≥ 0.7. Its strongest alternative now has direct support: the exemption does not just remove a cause of B's collapse, it drives any of these starts, untrained weights included, to the solution.
  - xref's f of 1.37–1.60 (above wd 0) therefore mixes "less decay where it hurts" with "structure supplied by the projection".
  - It should not be read as the share of the collapse that decay on the reference parts causes.

**Budget.** 12 runs, 67.5 s elapsed on 2 workers (134.8 s of process time). Checks and figure took 7 s.

## Section 3: dynamics runs, decay-off runs, the restricted-decay patch and its runs (QUESTION §3, addendum 3)

Rules: §3 (`33bf731`) for D6, and addendum 3 (`8a39ee1`, pushed before any of these runs was opened) for D13, the patch and D15. All replays are mine: `research/audit/independent.py` gradients and evaluation, with my own AdamW and my own explicit-DFT projection; nothing training-related is imported from `experiment.py`. "Replay-equal" means: identical train/cal/audit accuracies at every logged row; cross-entropies, norm, adaptive and decay norms within 1e-8 relative + 1e-14; weights within 1e-10 relative.

### D6, decay strength (dynamics packet 1, PR #33): Y1–Y7 all PASS

| # | result |
|---|---|
| Y1 | PASS. Six runs, each with running and complete events and 20,000 updates. Start-file hash = plan = config = file. The config differs from `e3-me-filt-w128-i<seed>-a1` only in wd, the checkpoint list and bookkeeping. Producing commit `52e7879` (clean), whose shared files equal `72cfb5d` |
| Y2 | PASS. Step 0 is bitwise equal to the start file and to the e3 wd 1 run's step 0. Split `aa8f7504…`. Row 0 is equal except wall-clock fields, and `data.npz` is equal |
| Y3 | PASS. `init_age` 50,000, no resume. A 10-update replay from the start with zero moments equals the update-10 checkpoint to ≤ 4.2e-16 relative |
| Y4 | PASS. i30001 at wd 0.3 and i30002 at wd 0.1, 300 updates: 30 of 30 rows with identical accuracies, the rest within 5.1e-14. Checkpoints within 7.1e-15 |
| Y5 | PASS. All 194 published checkpoints of the 12 runs in the lane's table (the six D6 runs plus the e3 wd 1 and wd 0 comparators): common accuracy and σ-margin equal the lane's to 4.4e-16 |
| Y6 | PASS. Peak update, T at both thresholds and re-crossings (none) equal the lane's for all 12 runs. All 12 ratios r equal to 1e-12, with the same classification: 10 exposure clock, and 2 faster (wd 0.1 at the update-0 threshold, i30001 and i30002). No decayed run is censored; the wd 0 runs are censored at > 20,000 |
| Y7 | PASS. The exposure is t·log of the per-update factor that `adam()` applies (Z1 a confirms the factor) |

The lane's MIXED verdict follows from its pre-stated rule applied to correct numbers. I do not re-judge it.

### D13, decay switched off mid-decline (dynamics packet 2, PR #36)

| # | result |
|---|---|
| Y1′ | PASS. Six runs, each with running and complete events and 20,000 updates. `resume` is the e3 wd 1 checkpoint at s, with the same hash in the config, plan and file. The config differs from the source only in wd (0), the resume and init fields, the checkpoint list and bookkeeping. Producing commit `83d8b94` (clean), whose shared files equal `72cfb5d`, with no decay flags |
| Y2′ | **FAIL as written; PASS on the state.** The metric row at s differs from the source's row in one more field than wall-clock: `optimizer_update` is null in the resumed run, because it takes no update at s, while the source's row describes update s. The accuracies, cross-entropies, margins and norm are all equal. The checkpoint at s (weights, both moments, step) is bitwise equal to the source's in all six runs. The rule should have excluded that field |
| Y3′ | PASS. My replay from the source checkpoint, with carried moments and the update count continuing at s + 1, is replay-equal at every logged row and at the first checkpoint after s (500 or 1,300) to ≤ 1.2e-15. **The control is informative:** with the count restarted at 1, the row at s + 10 already fails and the checkpoint is 4.5–5.4% away |
| Y4′ | PASS. A_1 and A_0 reproduce Q2's table. A_off, the Δ's and f equal the lane's exactly (gap 0). The reading, recovery label and T(0.10) agree |

**Values.**
- f is 0.055, 0.143 and 0.144 at s = 300, and −0.010, −0.000 and 0.092 at s = 1,000.
- All six read (i), "decay must stay on".
- There is no recovery: the highest accuracy after the switch is at s + 10, and it is below A_1(s). T(0.10) is censored at > 20,000 in all six.
- At 20,000 the runs are at 0.43–0.48 (s = 300) and 0.19–0.22 (s = 1,000). For comparison, wd 1 is at 0.015–0.035 and wd 0 at 0.56–0.63.

**Fragility** (pre-stated as reported, not judged). Every variant stays ≤ 1/3 in all six runs, so the reading does not depend on the window or the measure:
- f is 0.04–0.22 with W = 500 and −0.17–0.12 with W = 2,000;
- on the common pairs it is −0.03–0.17, at the nearest checkpoint shared by all three runs (W = 700 for s = 300).

**Intervention checklist.**
- The immediate effect is nil: the state is bitwise equal at the switch.
- The perturbation is wd only.
- The switch times come from the D5 window, which was set on held-out accuracy. That is a disclosed choice of time.

**Reading.** (i) is SUPPORTED LOCALLY as the lane reports. Q2's own caveat stands: Δ_0 comes from the wd 0 run, which is a different, higher state.

### The restricted-decay patch (`e9cbc84`, pinned at `8108d8b`): Z1–Z4 all PASS

| # | result |
|---|---|
| Z1 | PASS. **(a) The no-flag step is unchanged.** `adam()` of `72cfb5d` and of the pin, fed the same states and gradients, give bitwise equal weights and moments and identical measured norms over 300 updates, from all three B starts at wd 1 and at wd 0.3. **(b) The full CLI path is unchanged.** A diagnostic run at the pin without flags (i30001, wd 1, 300 updates; commit `0c45b39`, shared files = pin) equals the published D7 replay bitwise at 0, 50, 100 and 300, and 31 of 31 metric rows are equal except wall-clock fields |
| Z2 | PASS. My explicit-DFT P equals `decay_parts` to ≤ 3.5e-15 of the input norm, for both `on` values, on the start weights and on 10 random matrices. P is idempotent (1e-15) and orthogonal (Pythagoras 2e-16, cross term 1e-16). The rest keeps every column's DC (7e-16). At update 1, the squared decay norm splits into the reference and rest parts to 3e-16 |
| Z3 | PASS. A zero-gradient step gives exactly w − lr·wd·D(w) (bitwise), and matches my P to ≤ 1.8e-17. With a real gradient, the moments are bitwise those of the no-flag step, so they stay gradient-only. Both steps rebuild bitwise as decay plus the same Adam update, so the decay sits outside the update. The measured `decay_norm` equals ‖lr·wd·D(w)‖ to 3e-16 |
| Z4 | PASS. The projection-file hashes equal plan.json and the lane's json. k equals my dominant input frequency of the B start at update 0, computed from U and V only, with no labels. The live mask equals mine (128 of 128). P(W₀) = W₀ to 2.5e-15. P holds 96.5–96.9% of the squared norm of U and V; the rest is DC |

**Disclosure.** Z2's "relative" is taken against the input norm. The rest of W₀ is zero, so an output-relative gap is roundoff divided by roundoff (≈ 1); that is what the first pass reported, and I changed the normalization after seeing it.

### Restricted-decay runs, xref and onlyref (mechanisms intervention, PR #40): X1–X5 all PASS

| # | result |
|---|---|
| X1 | PASS. Running and complete events, 5,000 updates. The config equals the D7 wd 1 replay's except the projection fields and bookkeeping. `decay_on` is `rest` for xref and `reference` for onlyref. Producing commit `23bb74a` (clean), whose shared files equal the pin's |
| X2 | PASS. Update 0 equals the start file, and row 0 equals the replay's except wall-clock fields. The immediate effect is nil |
| X3 | PASS. My replay with w ← w − lr·wd·D(w) + Adam update, using my own P, reproduces every logged row to 5,000 in all six runs: 500 of 500 per run with identical accuracies, the rest within tolerance (worst 1.6e-12). Every published checkpoint agrees to ≤ 5.3e-14. The late gap is 0.21–8.2 times that of my no-flag replay of the D7 wd 1 run from the same start, within the pre-stated 10× |
| X4 | PASS. The logged `decay_norm` is in every row compared above, so it equals ‖lr·wd·D(w_{t−1})‖ from my replayed state at all 500 rows. The orthogonal split at update 1 is shown in Z2 |
| X5 | PASS. My f equals the lane's `intervention.csv` to ≤ 4.9e-7. **Deviation:** the lane's table is rounded to 6 decimals, so the pre-stated 1e-12 could not be applied; the gap is within half a unit of the last published digit |

**f per source.**

| arm | f at 2,000 | f at 5,000 | reading under INTERVENTION.md as written |
|---|---|---|---|
| xref | 1.51, 1.39, 1.31 | 1.60, 1.53, 1.37 | ≥ 0.7 in all three: decay acting on the reference parts is necessary for most of the collapse (qualified by X0, above) |
| onlyref | −0.11, −0.09, −0.14 | −0.04, −0.03, −0.04 | ≤ 0.3: decaying only the reference parts suffices |

**Perturbation size and training damage.**
- **Decay per update.** xref decays 13–23% of what uniform decay removes. Its logged decay norm is 0.005–0.009, against 0.037–0.045.
- onlyref starts at 96–99% of uniform decay and falls to about 20% by 5,000 (0.009 against 0.045), as its reference parts empty.
- **Distance from the replay.** The weights are 0.21 (xref) and 0.05 (onlyref) relative away at 300, and 0.88–1.05 and 1.26–1.45 at 5,000.
- **Training.** Train accuracy is 1.0 from update 40–70 on in both arms. Train cross-entropy at 5,000 is 2e-4 (xref) and 5e-4 (onlyref).

**Budget.** The patch checks took 8 s and the diagnostic run 0.8 s. The D6 checks took 3.4 s, D13 11 s (two passes) and D15 80 s (nine 5,000-update replays). Together with X0, the audit has used about 5.3 of 25 min in total.

**Reproduce**
```
uv run --locked python research/e4/audit/dynamics_check.py patch research/e4/audit/results
uv run --locked python experiment.py --out DIAG --task add --data-seed 31000 --init-seed 30001 --control-seed 32000 --diagnostic-seed 33000 \
  --expect-split-sha256 aa8f7504bf7d925eb90c69183f10daf5778cce9aad8e460b2c907811526d1f24 \
  --init-weights research/e3/mechanisms/starts/e3-me-filt-w128-i30001-a1.npz --init-age 50000 --steps 300 --wd 1 \
  --log-every 10 --checkpoint-steps 50,100 --diagnostics none --run-id diag-e4-au-noflag-i30001 --purpose diagnostic-audit
uv run --locked python research/e4/audit/dynamics_check.py noflag DIAG research/e4/audit/results
uv run --locked python research/e4/audit/dynamics_check.py d6 research/e4/audit/results
uv run --locked python research/e4/audit/dynamics_check.py d13 research/e4/audit/results
uv run --locked python research/e4/audit/dynamics_check.py d15 research/e4/audit/results
uv run --locked --group analysis python research/e4/audit/xref_control.py   # X0 projection files (already committed; rerunning changes the npz bytes)
uv run --locked python research/e4/audit/x0_check.py research/e4/audit/results
uv run --locked --group analysis python research/e4/audit/plot_x0.py research/e4/audit/figures/x0 research/e4/audit/results
```
