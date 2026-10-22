# e4 mechanisms: proposed intervention (not claimed, not run)

Proposed on 2026-10-22, after the first figure ([README.md](README.md)) and before any intervention code or run. The coordinator decides. Strengths and selections below use no held-out data.

## Question

At wd 1, between the peak and the collapse, the reference-frequency part of the readout loses about 35–40% of its held-out margin over operand-associated labels. Over the same updates, the off-reference part builds a contribution against a+b of about the same size. At wd 0 the reference part grows instead.

Is decay's action on the reference parts necessary for the collapse? Or does decay on everything else suffice, with the competitor overtaking even a reference part that is not decayed?

## Design: one manipulation, where decoupled decay acts

The design has two complementary arms, each with three sources. Everything else equals the D7 replays: the same start files, fresh Adam, lr 1e-3, wd 1, 5,000 updates, metrics every 10 updates, and the D7 checkpoint list.

| arm | decay applies to | run IDs |
|---|---|---|
| exempt-reference | each parameter minus its reference part: w ← w − lr·wd·(w − P(w)) | `e4-me-filt-wd1-xref-w128-i{30000,30001,30002}-a1` |
| reference-only | only the reference part: w ← w − lr·wd·P(w) | `e4-me-filt-wd1-onlyref-w128-i{30000,30001,30002}-a1` |

**The projection P.**
- For U and V, P keeps each unit column's component at frequency k_j along the token axis.
- For W, P keeps each unit row's component at k_j along the label axis.
- DC goes with "the rest".
- k_j is each unit's dominant input frequency at update 0. That is the frequency the filter kept, and it is the same basis rule as the channel split. It is fixed for the whole run and stored per source in a small committed file whose hash goes in the plan.

**Freezing.** Nothing is frozen. Every parameter gets its Adam update; only where the decoupled decay acts changes.

**Immediate effect.** There is none: update 0 is the unchanged start file. Every difference from the replays is therefore an effect of training under the changed decay. Gates, weight parts and the logged decay norm are reported, so the comparison covers magnitude as well.

**Cost.** Six runs of about 16 s each, about 1 min elapsed with 2 workers.

**Shared-code change.**
- `experiment.py` gets an optional `--decay-projection FILE` (per-unit k) and a `--decay-on {rest,reference}` switch.
- Without the flag, the step is bitwise unchanged.
- The test checks two things: without the flag, a step equals the current step; with it, the decay leaves the excluded part untouched and scales the other part by exactly 1 − lr·wd.
- It comes as a patch on this branch, with the test, before any claim (D3).

## Measurement

The same `channels.py` measures, on every checkpoint, compared seed by seed with the wd 1 and wd 0 replays.

**Primary quantity.** f = (acc_arm − acc_wd1) / (acc_wd0 − acc_wd1). Accuracy is on the 483 common pairs at update 5,000, computed per source. For reference, at 5,000 acc_wd1 is 0.021–0.027 and acc_wd0 is 0.634–0.737. 2,000 is reported too.

## Pre-stated reading

| arm | result | reading |
|---|---|---|
| exempt-reference | f ≥ 0.7 in all three sources | Decay acting on the reference parts is necessary for most of the collapse. Off-reference growth under full decay elsewhere does not by itself outvote a reference part that is not decayed. |
| exempt-reference | f ≤ 0.3 in all three sources | Decay on the reference parts is not necessary. What decay does to the rest suffices: the competitor overtakes even an undecayed reference part. |
| reference-only | f ≤ 0.3 | Decaying only the reference parts suffices for the collapse. |
| reference-only | f ≥ 0.7 | Decaying only the reference parts does not suffice. |
| either arm | mixed or intermediate f | Both contribute. |

The channel margins say which part moved in each arm.

**Caveat, stated now.** The parts are coupled through the loss. Exempting the reference part also changes how much gradient reaches the off-reference part, for example through smaller training losses. A "necessary" result therefore says that decay on the reference part matters, not that its shrinkage alone is the mechanism. If only one arm is affordable, run the exempt-reference arm.

## Outcome (added 2026-10-22 after the runs; nothing above was changed)

The coordinator approved the design (D14) and claimed the runs at pin `8108d8b` (D15). The six runs completed, launched at `23bb74a`.

f at 5,000, per source 30000, 30001 and 30002:
- **exempt-reference** (xref): 1.60, 1.53, 1.37. That is f ≥ 0.7 in all three, so decay acting on the reference parts is necessary for most of the collapse.
- **reference-only** (onlyref): −0.04, −0.03, −0.04. That is f ≤ 0.3 in all three, so decaying only the reference parts suffices for the collapse.

Common accuracy at 5,000 is 0.998–1.000 for xref and 0.000 for onlyref. Details, the post hoc decay-size comparison and the strongest alternative (the exemption is task-informed) are in [README.md](README.md).
