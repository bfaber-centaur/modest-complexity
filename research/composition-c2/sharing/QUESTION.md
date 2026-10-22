# c2 sharing lane: question, predictions, measures (M0)

Written 2026-10-22 before any c2-share measurement or run. Role prompt: [charter/03-sharing.txt](../coordinator/charter/03-sharing.txt). Assignment, pin, seeds and budget: [DECISIONS.md](../coordinator/DECISIONS.md) D3–D7. Adaptive choices made after this file are logged separately in [ADAPTIVE.md](ADAPTIVE.md).

## Question

Within the same three-step recurrent computation on r1's original task, how do the concrete interventions "tie the transition across steps" and "tie the readout across steps" change held-out composition under r1's 10k recipe?

This estimates the effect of these tying interventions under this recipe. It is not a parameter-efficiency comparison, and it cannot show that sharing is necessary in general.

## Design: a 2×2 at the original task, split and width

Everything not listed here is r1's recipe as pinned in `common/` at `b3e8132` (D3):
- task `ops51000-ls51100`, 48 training programs × 8 starts = 384 rows, 16 held-out programs = 128 rows;
- width 128, ReLU, full-batch Adam (lr 1e-3, b1 0.9, b2 0.999, eps 1e-8), decoupled wd 0.1 on every parameter, 10,000 updates;
- every cell gets the same labels: step1, step2 and final, with explicit equal step weights `--step-weights 1 1 1`.

| cell | transition (`A`, `B` op embedding, `bh`) | readout (`V`, `bv`) | parameters | runs (init seeds 51200–51202) |
|---|---|---|---|---|
| TT | tied | tied | 19,208 | reused: `r1-comp-rnn-inter-s5120{0,1,2}-a1` (recovered `ckpt-10000`, r1 metrics) |
| TU | tied | per-step | 21,272 | reused: `r1-rev-comp-rnn-untied-s5120{0,1,2}-a1` (reviewer's r1 arm) |
| UT | per-step | tied | 53,256 | new: `c2-share-ut-s5120{0,1,2}-a1` |
| UU | per-step | per-step | 55,320 | new: `c2-share-uu-s5120{0,1,2}-a1` |

The start embedding (`Ws`, `bs`) is one copy in every cell.

**Replays (labelled as such).** `c2-share-replay-tt-s5120{0,1,2}-a1` and `c2-share-replay-tu-s5120{0,1,2}-a1` rerun the two reused cells with `c2train.py --diag` on this host. They are reproductions of the original runs, not new seeds. Their purpose is to give all four cells optimization diagnostics from the same code path and host, and same-host timing. The scientific values for TT and TU are the original runs'. A replay counts as reproducing its original only if every logged r1 metric and every array of `ckpt-10000` (weights, both Adam moments, step) is identical; any difference is reported, not pooled.

**What the intervention includes.** These are part of what "untying" changes and are reported, not controlled away:
- *Initialization identity.* `c2rnn.init` draws r1's tied parameters and copies an untied group three times. All four cells of a seed start as the same function. Checked here: the step-0 metric rows of the four cells, `held_final_pred` included, must be identical per seed.
- *Gradient aggregation.* A tied parameter receives the sum of its three per-step gradient terms. Untied copy k receives only step k's term. For a transition copy, step k's term carries backpropagated loss from steps k..3; for a readout copy, only step k's loss. Adam then normalizes each coordinate separately, so the summed-versus-single difference reaches the update mainly through the moment estimates and, wherever √v̂ is not large against eps = 1e-8, through the gradient's scale.
- *Decay.* Decoupled wd 0.1 multiplies every parameter, every untied copy included, by (1 − lr·wd) per update. An untied group therefore decays three times as many parameters, each at the tied parameter's rate, against a gradient made of one term instead of three.
- *Data reuse.* Each training row applies a tied transition three times per update, and an untied copy once. Each untied copy sees one position's (state, op) inputs only.
- *Parameter count.* Untying the transition adds 34,048 parameters, the readout 2,064. Width is matched; parameter count is not, and no dummy parameters are added.

**Supervision exposure** is identical in all cells: 384 rows × 3 labels × 10,000 full-batch updates (3.84M row presentations, 11.52M label presentations).

## Measures

**Primary**, per run, logged every 50 updates by `c2train.py` (r1 keys plus `held_subset`):
1. held-out final accuracy on 128 rows;
2. held-out final accuracy on the 42 ¬N rows, where no one-op-slot training neighbour carries the true answer, and on the 86 N rows;
3. training accuracy on 384 rows at the final step (and step1, step2);
4. the 10,000-update endpoint of 1–3.

For TT and TU, N/¬N at every logged step are computed from r1's logged `held_final_pred` and `c2task.neighbour_answerable`. They are checked against the replays' logged `held_subset`.

**Derived.**
- First update at which training final accuracy is 1.0.
- First update at which held-out final accuracy is ≥ 0.95 at five consecutive logged evaluations (the time lane's threshold). A run that never meets it is censored at 10,000.
- Paired differences per seed: transition effect TT − UT (tied readout) and TU − UU (per-step readout); readout effect TT − TU and UT − UU; interaction (TT − TU) − (UT − UU).

**Decoder-across-position table** (D5), `c2probe.decode_table` unchanged, probe seed 64300, at `ckpt-10000` of all 12 runs and at `ckpt-0` (which must give the same table for all four cells of a seed). Programs have length 3 and start from the start state, so every untied copy is used at its own step; no cross-position step map is needed.
- *Reading limits.* Held-out h1 and h2 states are training-prefix states (C1.1), so entries scored at j = 1, 2 measure training states. A low off-diagonal entry does not show that position j lacks the symbol; an invertible position-specific code remains possible. A high entry does not show that the network uses that code.

**Secondary (descriptive, not in the first figure).**
- *Readout-across-position table.* Readout copy k applied to h_j (k, j in 1..3), accuracy against the symbol at j, on training and held-out rows. This is the model's own readout, not a fitted probe; in tied-readout cells every k is the same readout.
- *Optimization diagnostics.* From `--diag` every 50 updates: global gradient and update norms, √v̂ median, max and fraction below eps, and the training final-step margin. From the checkpoints at 0, 1k, 2k, 4k, 7k and 10k, per parameter group (start, transition, readout; each copy separately):
  - parameter norm;
  - Adam-direction step norm lr·‖m̂/(√v̂ + eps)‖, decay step norm lr·wd·‖p‖, and the cosine between the two;
  - fraction of coordinates with √v̂ < eps;
  - cosine similarity between untied copies, which all start at 1.
- *Same-host timing.* ms per update and peak RSS for each cell.

## Predictions (before any c2-share measurement)

Known before writing, from r1 (not predictions): TT reaches 1.00 / 0.96 / 1.00 held-out final at 10k and ¬N 1.00 / 0.88 / 1.00. TU reaches 0.63 / 0.52 / 0.52 and ¬N 0.24 / 0.07 / 0.05, still rising. Both fit training by 400–700 updates.

1. **Fit.** UT and UU fit training (train final accuracy 1.0 at all three steps) by 1,000 updates on every seed.
2. **Transition tying under a tied readout.** UT is below TT on every paired seed at 10k, by at least 0.10. UT lies in 0.30–0.90.
3. **Transition tying under per-step readouts.** UU is at or below TU on every seed at 10k. UU lies in 0.15–0.55.
4. **Interaction.** The two tyings are super-additive: (TT − TU) > (UT − UU) on every seed. Untying either group removes most of the benefit of tying the other.
5. **Where the gains are.** Any cell below 0.70 held-out at 10k has ¬N ≤ 0.35: its above-chance accuracy is mostly on rows a neighbour copy can answer.
6. **Decoding.** Among positions 1–3, the mean off-diagonal entry is ≥ 0.8 for TT and ≤ 0.5 for UU, with UT and TU in between. Position 0 is never read by a readout, and no prediction is made for it.
7. **Timing.** Untied-transition cells cost 10–20% more per update than tied-transition cells, as in the coordinator's calibration.

What would change the reading:
- UT ≥ 0.9 on every seed would mean that, under this recipe, a shared readout with intermediate supervision suffices without a shared transition.
- UU ≥ TU on every seed would mean that the shared transition adds nothing once the readouts are per-step.
- Prediction 4 failing would mean the two effects are roughly additive or that one subsumes the other.

## Controls and checks

- Init identity: identical step-0 rows across cells per seed, and identical decode tables at `ckpt-0`.
- Gradient aggregation: at init on each seed, the untied copies' gradients sum to the tied gradient. `test_c2.py` covers this at a test seed; it is repeated here at 51200–51202 as a cheap oracle check.
- Replays reproduce the original TT and TU runs, arrays included.
- Reference levels, drawn on the figure: chance 0.125; a neighbour copier scores 0 on ¬N.
- r1's `rnn-final` (0.27–0.32) and MLP `inter` (0.20–0.25) are quoted for context only. They differ in supervision or architecture and are not cells of this design.
- The pinned tests run in their own process before any training: `uv run --locked python -m unittest -v research/composition-c2/common/test_c2.py`.

## First figure (before any expansion)

`figures/sharing_grid.png` (+ `.svg`, source CSV, regeneration command):
- **2×2 trajectory panels** laid out like the design table: rows are transition tied / per-step, columns readout tied / per-step. Each panel shows, for every seed separately, held-out final (solid), ¬N (dotted) and training final (thin dashed) accuracy against updates 0–10,000, with chance and the 0.95 threshold drawn. Axes are identical across panels.
- **Endpoint panel**: 10k held-out final, N and ¬N, and training final for every run, by cell. Lines join the paired seeds.

`figures/decode_table.png`: the decoder-across-position table for each cell and seed at 10k, with the `ckpt-0` table for reference. Every value is in the CSV.

## Follow-up (one, adaptive)

It is chosen after the initial comparison, proposed in writing as a "Follow-up" section appended here, and approved by the coordinator before launch. It addresses the largest remaining ambiguity, with one of two candidates selected by these indicators:
- **Regularization (matched no-decay comparison at wd 0).** Selected if the cells separate after the training set is fit, in a window where the decay step is comparable to the Adam-direction step in the groups that differ between cells (cosine near −1, ratio near 1). That means the late dynamics are a decay/gradient balance that untying changes.
- **Optimization exposure (loss-scale sensitivity).** Selected if, where the cells separate, untied copies are eps-limited (a large fraction of √v̂ < eps) while their tied counterparts are not. That means the summed-versus-single gradient scale, not tying as such, may set the step size.
- If neither indicator holds, the follow-up proposal says so and names the ambiguity it targets instead.

Longer training belongs to the time lane and is not a candidate here.

## Interpretation limits (fixed in advance)

- One task draw, one split, three paired init seeds: exploratory coverage, not statistical power.
- Cells differ in parameter count, decayed-parameter count, gradient aggregation and per-copy data reuse at once. A difference between cells is the effect of the whole tying intervention under this recipe, not a pure effect of sharing or of parameter count.
- Every value is a 10k-update value. A cell that is still rising has not plateaued, and a finite-budget gap is not impossibility.
- Linear decodability is not use. A successful tied model is not thereby shown to implement a clean symbolic state machine.

## Literature

Starting point: Liška, Kruszewski and Baroni (2018), "Memorize or generalize? Searching for a compositional RNN in a haystack", arXiv:1802.06467. Methods read on 2026-10-22:
- 8 random bijective 3-bit lookup tables and their 64 pairwise compositions, as character-level sequence-to-sequence episodes for a 60-unit LSTM plus a 10-unit sigmoid layer, trained by per-episode SGD-style updates with Adam;
- the intermediate result is part of the required output string;
- the held-out test is 2 unseen *inputs* for each of the 64 seen composed tasks;
- with direct supervision of a hand-designed hidden-state code (an automaton encoding), the network composes (96%);
- without it, about 2% of 50k initializations exceed 80% zero-shot accuracy, and success depended on data order rather than initialization;
- the paper reports, without data, that training on a subset of composed tasks did not generalize to unseen composed tasks.

**Relation to this lane.**
- *Not a reproduction.* The architecture, task format and held-out unit differ. Here whole programs are held out by a Latin square, so the test is unseen compositions, the case that paper reports as failing.
- *Extension.* The paper's recurrence always shares weights across time. Untying transitions and readouts in a small full-batch ReLU RNN is not done there.
- *Novelty.* No claim is made without a targeted literature check.

## Follow-up (proposed 2026-10-22 04:25Z at `d8735fc`, after the initial comparison; approved by the coordinator 04:26Z, launched 04:26:52Z)

**What the initial comparison left most ambiguous.** Only TT composed within 10k (0.96–1.00, ¬N 0.88–1.00). TU, UT and UU reached 0.52–0.84 and were still rising. The tyings interact strongly, and TU < UU on every seed. All cells fit by 700 updates, so every difference arises in the post-fit phase. In that phase:
- decay is a large part of the dynamics: in tied groups the decay step is 0.44–0.94 of the Adam-direction step and opposes it (cosine −0.40 to −0.94), less closely so in per-step transition copies (0.10–0.74);
- Adam is not eps-limited anywhere: √v̂ < eps for at most 8.8% of the coordinates in any group.

By the indicators declared above, the regularization candidate is selected and the loss-scale candidate is not. Untying changes how decay is spread (three times as many decayed parameters in an untied group, each opposed by one gradient term instead of three). So the open question is whether the tying effects at 10k depend on decay.

**Design.** The matched no-decay comparison. All four cells at wd 0, three paired init seeds (51200–51202), everything else identical to the initial runs:
- task, width, lr, step weights 1 1 1, 10,000 updates, `--diag`;
- checkpoints at 0, 1k, 2k, 4k, 7k and 10k.

IDs are `c2-share-wd0-{tt,tu,ut,uu}-s5120{0,1,2}-a1`: 12 runs, and TT/TU at wd 0 are new runs. Each wd 0 run is paired with the same cell and seed at wd 0.1; they start from the identical `ckpt-0`. The cost at the observed rate is about 7.2 min wall at two workers, bringing the lane to about 14.8 of 20 min.

**Measures.** The same as the primary ones:
- held-out final, ¬N, N and train accuracy trajectories and 10k endpoints;
- paired differences within wd 0, and wd 0 − wd 0.1 per cell and seed;
- the decoder table at 10k, probe seed 64300;
- group diagnostics.

Figure: `figures/followup_wd0.png`, laid out like the primary figure with the wd 0.1 endpoints beside the wd 0 ones.

**Predictions (before any wd 0 run).**
- F1. All 12 runs fit training (all three steps at 1.0) by 1,000 updates.
- F2. TT at wd 0 is below TT at wd 0.1 at 10k by at least 0.2 held-out on every seed: decay drives much of TT's late rise.
- F3. The spread across cells (max − min held-out at 10k, per seed) is smaller at wd 0 than at wd 0.1 on every seed.
- F4. TT is still the highest cell at wd 0 on at least 2 of 3 seeds.

**Reading.**
- If TT at wd 0 still reaches ≥ 0.9 while the other cells stay below, the tying effect within 10k does not depend on decay under this recipe.
- If all cells are close and low at wd 0, the tying advantage seen at 10k depends on decay within this budget. This is not a claim that decay is necessary in general.
- A reversal of TU < UU at wd 0 would tie that ordering to the decay accounting.

**Limits.** wd 0 removes decay from every parameter. It is not a removal of only the decay-count difference between tied and untied groups, and it changes the post-fit dynamics of every cell. Values are at 10k only; longer training stays with the time lane.
