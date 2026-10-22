# R1 composing small programs: note (M4)

**Question.** A small network learns 48 of the 64 length-3 programs over four random permutations of eight symbols. Under what conditions does it answer the 16 unseen programs?

Design, predictions and the follow-up's pre-registration are in [QUESTION.md](QUESTION.md). Every number below comes from the run records in `runs/` (IDs given) or the CSVs in `figures/`.

## Design in brief

**Split.** A Latin square holds out 16 whole programs at every start symbol. In training:
- every op appears 12 times at each position;
- every ordered pair appears 3 times at each slot pair;
- every (state, op) transition occurs at every position.

**Baselines, all at or below chance on this split.**

| baseline | held-out accuracy |
|---|---|
| chance | 0.125 |
| every drop-one-op or single-op heuristic | ≤ 0.125 |
| every training-fitted lookup on a proper subset of slots | ≤ 0.125 |
| a 3-of-4-slot neighbour vote | 0.077 |

**Leakage.** No products collide, no ops commute, and the train and held-out rows do not overlap. The reviewer independently passed the oracle and the split (REVIEW.md §1). It also showed that the 48 training programs' final-answer tables alone determine the four permutations uniquely. So final-only training lacks no information about the primitives; intermediate labels make them explicit.

**Model and conditions.**
- **Model.** `common/nets.py` 20→128→128 ReLU MLP, full-batch Adam (lr 1e-3, decoupled wd 0.1), 10,000 updates.
- **Conditions.** `final` (answer head), `inter` (+ step1 and step2 heads), `shuf` (the same heads, labels permuted across rows).
- **Seeds.** Paired init seeds 51200–51202; the trunk starts bitwise identical across conditions.
- **Positive controls.** P1 fits all 512 rows (1.0). P2 trains the 32 transitions and applies them three times with the composition hard-wired; it scores 1.0 held-out.

## Observations

### Initial comparison

Primary figure: [figures/heldout.png](figures/heldout.png).

**Every run fits.** Training accuracy is 1.0 in all nine runs by about 250 updates, so the low held-out numbers are failed generalization, not failed optimization.

| held-out final accuracy at 10,000 updates | 51200 | 51201 | 51202 |
|---|---|---|---|
| `final` | 0.164 | 0.172 | 0.141 |
| `inter` | 0.242 | 0.203 | 0.250 |
| `shuf` | 0.125 | 0.109 | 0.102 |

**The paired differences are small and of consistent sign.**
- `inter − final` is +0.08, +0.03 and +0.11.
- `shuf − final` is −0.04, −0.06 and −0.04.

**Trajectories.** In two of the three `final` runs, held-out accuracy starts below chance (0.06–0.09 around updates 100–500, near the neighbour-vote level) and then drifts upward. `final` and `inter` were still drifting upward at 10,000 updates.

**Intermediate heads generalize; the final answer does not.** In `inter`, held-out step1 accuracy is 1.0 and step2 is 0.98–1.0, yet held-out final accuracy stays at 0.20–0.25. Outcome 5 of QUESTION.md is the closest description.

**What the errors are** (`heldout_error_types.csv`).
- 97–100% of wrong held-out answers equal the label of a training row that differs in exactly one op slot. If wrong answers were uniform, that would be about 73%.
- In `inter`, 80–86% equal the label of a row with the same prefix and a different op 3, against about 35% if uniform. In `final` the figure is 66–70%.
- So the networks answer an unseen program with an answer that belongs to a seen neighbour. In `inter`, that neighbour typically shares the correctly computed state after op 2.

### Decoding (descriptive, `r1-comp-probe-a1`)

Supporting figure: [figures/decoding.png](figures/decoding.png).

**Step1.**
- Step1 is ≥ 0.96 decodable on held-out rows from hidden layer 1 even at initialization, because random ReLU features of (s, a) suffice.
- From the input itself it is 0.41.

**Step2, held-out, from hidden 1.**
- 0.40–0.42 at initialization.
- 0.52–0.57 after `final`, 0.75–0.83 after `inter`, 0.42–0.47 after `shuf`.

**Step2, held-out, from hidden 2.**
- 0.55–0.66 after `final`.
- 0.98–0.99 after `inter`, where that layer is directly supervised.

Training without intermediate labels therefore makes step2 more decodable than in the untrained network, but not fully.

### Interchange intervention (training-program pairs, where the model is correct)

The intervention swaps the probe's 7-dimensional step2 subspace into a base row from a source row with a different step2. Pairs were chosen so that the composed counterfactual, `P[c_base](step2_source)`, differs from both rows' answers.

| swap site | output follows the counterfactual | random 7-d subspaces | base answer kept |
|---|---|---|---|
| hidden 1 | 1–3% (`final`), 4–7% (`inter`) | ≤ 1% | 77–93% |
| hidden 2 | up to 12–16% (`inter`) | ≤ 1% | 42–48% |

At hidden 2 in `inter`, 35–38% of outputs become the *source* row's answer, against 14–15% for random subspaces. The decoded step2 subspace has some causal effect beyond a random subspace. But the networks mostly do not compute the final answer by applying op 3 to the decoded step2.

### Adaptive follow-up: a shared-transition RNN, kept separate

Figure: [figures/followup_rnn.png](figures/followup_rnn.png). It was pre-registered at b842eb2, before any follow-up run.

**Model.** A ReLU RNN that applies one shared transition `h_t = ReLU(h_{t-1}A + e_op B + b)` per op, with one shared readout. Width 128, 19,208 parameters, the same split, optimizer and seeds.

| held-out final accuracy at 10,000 updates | 51200 | 51201 | 51202 |
|---|---|---|---|
| `rnn-inter` (readout trained on h1, h2, h3) | 1.000 | 0.961 | 1.000 |
| `rnn-final` (readout trained on h3 only) | 0.273 | 0.312 | 0.320 |

**`rnn-inter`.**
- Training accuracy reached 1.0 by about 500 updates. Held-out accuracy rose gradually *after* that: 0.34–0.36 at 1,000 updates, 0.48–0.66 at 4,000, 0.83–0.93 at 7,000.
- The prediction of ≥ 0.8 for all seeds held.

**`rnn-final`.**
- It sits above every MLP condition and was still rising at 10,000 updates. It reached 0.27–0.32, at the low edge of the predicted 0.3–0.9.
- The descriptive prediction failed. The shared readout applied to h1 and h2 decodes step1 and step2 at about chance (0.06–0.15 at 10,000 updates), so without intermediate supervision the RNN did not form a state code shared across positions.

## Reading

**In this architecture, the information is not enough.** Final-answer and intermediate-state supervision both left the MLP well short of composing. The training labels determine every primitive, the MLP represents the intermediate state of held-out programs (decodable, and correct at the intermediate heads), and still it mostly answers with a seen neighbour's answer.

**The two structural conditions that made it compose.** One transition shared across positions, plus intermediate states supervised through one shared readout. Together they gave near-perfect zero-shot composition, which emerged slowly after the training set was fit. Sharing the transition alone helped modestly over the MLP within 10,000 updates.

**This does not show** that the shared transition is the component that matters as distinct from the shared readout. These runs do not separate the two.

## Limitations

- **One draw.** There is one task draw (seed 51000) and one split (seed 51100). Three init seeds, full-batch and deterministic, are the only replication. Seed-to-seed spread is small in every condition.
- **One recipe.** The recipe is fixed: width 128, lr 1e-3, wd 0.1, 10,000 updates. MLP `final`, MLP `inter` and `rnn-final` were all still rising at the end, so every number is a 10,000-update number, not an asymptote.
- **Decoding.** Probes are linear and fitted on 384 rows. A low held-out probe accuracy does not show that the information is absent.
- **The interchange.** It tests only the linear probe subspace. The base answer survived the swap in most pairs, which may mean the state is also encoded redundantly in the complement.
- **The RNN changes two things.** It differs from the MLP in shared transition and shared readout at once. The RNN and MLP seeds are not paired.
- **Unpaired controls.** `shuf` was run only for the MLP. The RNN arms have no information control.

## Next question

Two candidates:
- Does `rnn-final` reach `rnn-inter`'s level with longer training? It was still rising at 10,000 updates, as `rnn-inter` itself did between 1,000 and 10,000.
- Does an RNN with per-step readouts that are *not* shared still compose under intermediate supervision?

The first separates "slower" from "different". The second separates the shared transition from the shared readout. Either costs a few minutes of CPU.
