# e2 review index (figures first)

Round e2 is open-question exploration under the [charter](charter/README.md); assignments and the run-ID procedure are in [DECISIONS.md](DECISIONS.md). e1 is complete and archived at [research/coordinator/INDEX.md](../../coordinator/INDEX.md).

e2 base for workers: `6aaa3a2b6898abd0b317aa775cd7939e7530abdd`. Shared code identity: `fe284a5d90f741aab8c1273a55b11e5352ec38c5` (unchanged).

## Shared check: both training shares on the pairs neither trained on

Question: the 30% (e1) and 50% (follow-up 2) width-128 runs were each evaluated on their own held-out split. Scored on the same pairs, do they still differ, and when?

![common set](figures/common-set.png)

Look before reading: compare the solid and dotted line of each blue run, and where the grey runs sit relative to the chance line.

Vector version: [common-set.svg](figures/common-set.svg). Source: [common_set_scores.csv](common_set_scores.csv) (every published checkpoint of the six runs: common-set accuracy, cross-entropy and median margin, plus each run's own cal/audit accuracy), [common_set.json](common_set.json).

Observations:

- The training sets nest: 30% train (299 pairs) ⊂ 50% train (478 pairs); the 179 added pairs are all 118 e1 calibration pairs and 61 e1 audit pairs.
- Never trained by either: 483 pairs, 248 unordered groups, transpose-closed, 15–16 pairs per class. This is exactly the 50% split's cal ∪ audit and lies inside the e1 audit split (`research/e2/data/common-never-trained-30-50.npz`, SHA-256 `1535a21ab192ae95d37a5abb5eb783e74f51bd18a0500fbe0218d910914434d3`).
- At 50,000 updates on these 483 pairs: 30% runs 0.000 / 0.000 / 0.004; 50% runs 0.913 / 0.878 / 0.909 (own audit split 0.923 / 0.868 / 0.904). Common-set cross-entropy at 50k: 9.9–10.5 against 0.39–0.50.
- e1 i30002 has only steps 300 and 50,000 published (markers without a line).

Status: software checks on the 50% runs PASS in audit Part 4 (F10–F15, independent code). The common-set result was recomputed with independent code in e2 audit G7 (PASS: all fields and all 51 rows match). "Amount of data" versus "which pairs" is AMBIGUOUS (D6).

Reproduce: `uv run --locked --group analysis python research/e2/coordinator/common_set.py`

## Lanes

| lane | directory | status |
|---|---|---|
| dynamics | [research/e2/dynamics/](../dynamics/) | both cycles integrated. Packet 1: correct-label margins on the common pairs ([share](../dynamics/figures/share/margins.png), [width](../dynamics/figures/width30/margins_width30.png)). Packet 2: nested share boundary 30/40/45/50% at width 128, six new runs ([boundary](../dynamics/figures/boundary/margins_boundary_summary.png)). Not audited |
| mechanisms | [research/e2/mechanisms/](../mechanisms/) | packet 1 (PR #15): unit groups by dominant frequency, [figure](../mechanisms/figures/groups/groups.png), audited in audit cycle 2. Packet 2 (PR #17): weights-only per-unit filter, [figure](../mechanisms/figures/unitfilter/unitfilter.png); coordinator spot check and no-decay counterexample in D10; closing cycle 2 |
| audit | [research/e2/audit/](../audit/) | cycle 1 (PR #16): restricted-rank definition and nulls, `common_set.py` recomputed (G7 PASS), "rank precedes accuracy" SUPPORTED LOCALLY (partly by construction), "early signature of success" CONTRADICTED as a level. Cycle 2 (PR #18): mechanisms packet 1 reproduced (G10-G12 PASS, G13 correction), no-decay and shuffled controls, dense onset with three seeds per share. Now auditing mechanisms packet 2 |

Run claims: [research/e2/plan.json](../plan.json): six dynamics runs at 40% and 45% share (D8), all complete, registry `research/e2/dynamics/registry.jsonl`. Extra e1/e1f2 checkpoints: [e1-checkpoints/](e1-checkpoints/) (D9).

## Base

e2 base: `6aaa3a2b6898abd0b317aa775cd7939e7530abdd` (2026-10-21), the commit that opened e2 on `swarm/exploration`. Later `swarm/exploration` commits are valid bases while they change no shared code (DECISIONS.md D4).
