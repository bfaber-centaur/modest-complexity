# R1 composing small programs: question and design (M0)

Written 2026-10-22 before any training run. Task properties and non-trained baselines below come from `task.py` (enumeration only, no training). Run IDs `r1-comp-…`, seeds 51000–51999, session label `r1-comp-cloud-1`.

## Question

A two-hidden-layer ReLU MLP is trained on 48 of the 64 length-3 programs over four fixed random permutations of eight symbols, at every start symbol. Does its accuracy on the 16 held-out programs rise above what non-compositional strategies achieve? Does supervising the intermediate states (after op 1 and after op 2) change that, beyond what the same number of label-free auxiliary loss terms does?

## Task, split, architecture, objective

**Task.**
- Eight symbols. Ops are four permutations `P[k]` drawn with `np.random.default_rng([51000, k])`.
- The rule was fixed before the draw: reject only if an op is the identity or two ops are equal. The first draw (51000) passed.
- A program `(a, b, c)` maps a start symbol `s` to `P[c][P[b][P[a][s]]]`. There are 64 programs × 8 starts = 512 rows.

**Input.** Slot one-hots, 20 units: start (8), then op slots 1, 2 and 3 (4 each). The input does not tell the network that the slots are the same four ops, or which order they apply in.

**Split.**
- Seed 51100 draws one of the 576 4×4 Latin squares `L` uniformly. The 16 programs `(a, b, L[a][b])` are held out at every start symbol, so whole sequences are held out.
- Train: 48 programs × 8 starts = 384 rows. Held out: 128 rows.
- Square used: `[[0,1,3,2],[3,2,0,1],[2,0,1,3],[1,3,2,0]]`.

**Coverage of the training programs.**
- Every op appears 12 times at every position.
- Every ordered op pair appears exactly 3 times at slot pairs (1,2), (2,3) and (1,3).
- All 32 (state, op) transitions occur at each of positions 1, 2 and 3.
- All 128 prefixes `(s, a, b)` occur, and all 128 `(step1, b, c)` contexts.
- Final labels are balanced: 48 of each symbol in training and 16 of each in the held-out rows.

**Equivalence and leakage checks** (`task.py`).

| op | table | cycle type | fixed points | order |
|---|---|---|---|---|
| 0 | 7 0 1 2 4 6 5 3 | 5,2,1 | {4} | 10 |
| 1 | 5 0 7 2 6 1 3 4 | 5,3 | none | 15 |
| 2 | 6 3 0 5 1 2 7 4 | 8 | none | 8 |
| 3 | 6 1 4 3 0 5 7 2 | 5,1,1,1 | {1,3,5} | 5 |

- No op is the identity or an involution, and no pair of ops commutes.
- The 16 length-2 products are distinct, and so are the 64 length-3 products. No held-out program computes the same permutation as a training program.
- Train and held-out rows do not overlap.
- 10 of the 16 held-out programs repeat an op: (0,0,0) is the only all-same one; three have a=b, three b=c, and three a=c. The other 6 use three distinct ops. This repetition pattern is one structure breakdown.

**Non-trained baselines on the held-out rows.** Each has an expected accuracy, with a uniform tie break where it applies.

| baseline | held-out accuracy |
|---|---|
| chance | 0.125 |
| identity `s` | 0.102 |
| single-op heuristics `P[a](s)`, `P[b](s)`, `P[c](s)` | 0.063, 0.102, 0.063 |
| drop-one-op heuristics (drop op 1, 2 or 3) | 0.125 each |
| best training-fitted lookup on any proper subset of slots | 0.125 (all 14 subsets ≤ 0.125) |
| plurality vote of the 9 training rows sharing 3 of 4 slots (a lazy, kernel-like interpolation) | 0.077 |
| nearest seen program, chosen using the held-out table itself (optimistic oracle bound) | 0.453 |

So, apart from the oracle-selected bound, nothing that reuses seen rows or partial programs beats chance on this split. Held-out accuracy well above 0.125 needs the start symbol and all three op slots combined in a way that was never seen as a row. Interpolating from neighbouring rows is expected to land *below* chance.

**Architecture.**
- `common/nets.py`: 20 → 128 → 128 ReLU, with softmax heads on the last hidden layer. He-normal trunk, zero biases.
- Heads: `final` (8 classes), plus `step1` and `step2` (8 each) in the conditions that use them.
- This architecture does not encode that the slots share ops or that they apply in sequence.
- A note on depth: a stage-wise interpreter needs one conjunction per op, which is three hidden layers. With two hidden layers, a solution that only uses seen combinations has to compute one triple conjunction, for example `(s, a, b)` → step2 in layer 1 and then `(step2, c)` in layer 2. Every such triple and pair does occur in training.

**Objective and optimizer.**
- Sum over the condition's heads of the per-head mean cross-entropy, each with weight 1.
- Full-batch Adam: lr 1e-3, decoupled weight decay 0.1, β = (0.9, 0.999).
- 10,000 updates. Evaluation every 50 updates on all 512 rows.
- Fixed and untuned. No minibatch order exists, so the init seed is the only randomness within a condition.

## Conditions (initial comparison), all at init seeds 51200, 51201, 51202 (paired)

| condition | heads trained | labels per training row | extra parameters |
|---|---|---|---|
| `final` | final | 1 | 0 |
| `inter` | final + step1 + step2 (true states) | 3 | 2 heads × (128·8 + 8) = 2,064 |
| `shuf` | final + step1 + step2, with the (step1, step2) label pairs permuted across the 384 training rows by one fixed permutation (seed 51300) | 3 | 2,064 |

**Exposure.**
- All three conditions use the same 384 inputs, the same 10,000 full-batch updates (3.84 M row presentations) and the same trunk (19,200 parameters) plus the final head (1,032).
- `nets` draws each head from (seed, name), so for a given init seed the trunk and the final head start bitwise identical across conditions.
- Extra signal in `inter`: two 8-way labels per row. The step1 labels alone give all four permutation tables in full, because every `(s, a)` occurs. So `inter` is given the primitives explicitly, as well as a hint about the route.
- `shuf` keeps the label count, the marginals, the joint (step1, step2) distribution and the three loss terms, but carries no information about the inputs. It adds a memorization burden that `inter` does not.
- I have not checked whether the final labels alone determine the four permutations. I expect they nearly do, which would mean final-only training lacks no information in principle.

## Plausible outcomes and what distinguishes them

Each reading assumes training accuracy on the final head reaches 1.0 in every run. A run that does not reach it is reported as an optimization failure, not as a generalization result.

1. **Memorization without composition.** Held-out final accuracy stays at or below about 0.125 in all conditions. Near the 0.077 neighbour-vote level would mean interpolation from seen rows.
2. **Intermediate labels enable composition.** `inter` held-out final accuracy is well above both `final` and `shuf` for each paired seed.
3. **Both compose.** `final` is also well above 0.125, for example above 0.5.
4. **Seed-dependent.** One of three seeds generalizes and the others do not, as in Liška et al., where about 2% of 50k RNNs exceeded 80%. Three seeds give no rate, only whether this happens at all.
5. **Prefix states generalize, the last step does not.** In `inter`, held-out step1 and step2 accuracy are high, because the `(s, a)` and `(s, a, b)` contexts were all seen, but held-out final accuracy stays near chance. That would fit the depth note: with heads on the last layer, applying op 3 to the computed step2 needs a (step2, c) conjunction in that same layer.
6. **Extra loss terms, not information.** `shuf` ≈ `inter` would mean the extra loss terms, not their information, drive any gain. `shuf` < `final` would mean the random labels cost capacity.

**The distinguishing measurement** is the per-seed paired difference in held-out final accuracy, `inter − final` and `shuf − final`, over training and at 10,000 updates. It is read next to the training accuracy of every head, and next to held-out step1 and step2 accuracy in `inter`.

**My prior,** recorded so it can be wrong:
- `final` held-out stays within about 0.05–0.25 at 10,000 updates (outcome 1).
- In `inter`, step1 and step2 generalize well, and held-out final is modestly above `final` (outcomes 2 and 5 mixed).
- `shuf` ≈ `final`.
- Confidence is low. If held-out accuracy is still rising at 10,000 updates, that is the main reason to change duration in the follow-up.

## Controls and known confounders

**Controls.**
- The non-trained baselines above, drawn on the figure.
- `shuf`, the information control.
- Positive control P1 (`full`): the same network and heads trained on all 512 rows. It checks that the 64 tables and the intermediate heads can be fit, and that the metrics read 1.0. It says nothing about generalization.
- Positive control P2 (`interp`): a 12 → 128 → 128 network trained on the 32 one-step transitions `(state, op) → next state` that the training rows' intermediate labels expose. It is then applied three times, feeding back its argmax, on every row. It checks that the held-out evaluation can register success and that the training data carry every transition needed. The composition is hard-wired here, so it validates the pipeline, not the MLP.

**Known confounders and limits.**
- There is one task draw and one split, so every result is conditional on this permutation set and this Latin square.
- The recipe is fixed and untuned (width, depth, lr, weight decay, duration).
- Argmax accuracy is the headline. Cross-entropy is logged beside it.
- Init seeds are the only replication. Checkpoints within a run are not replications.

## Measurements

**Accuracy.**
- Train and held-out accuracy, plus cross-entropy, for every head, every 50 updates.
- Held-out final predictions for all 128 rows at every evaluation, so the structure breakdowns can be computed at any step: per program, by repetition pattern, by start symbol, and whether the prediction equals a heuristic's or a neighbour's answer.

**Decoding (descriptive).**
- Linear softmax probes (`nets` with no hidden layer) for step1 and step2.
- Read from the input, from hidden layer 1 and from hidden layer 2, at the final checkpoint of each run.
- Fit on training-program rows and scored on held-out-program rows.
- Baselines: the input layer, and the same network at initialization.
- Decodability is not use.

**Targeted causal test, only if decoding shows a held-out-decodable step2** in some layer, and the final head generalizes beyond chance.
- An interchange intervention on that probe's subspace: in a base row, replace the component of the layer's activation in the span of the probe's (centred) class directions with a source row's component.
- The source row has the same op 3 but a different step2.
- Measure how often the output becomes `P[c](source step2)` rather than the base answer.
- Controls: a random subspace of the same dimension, and a whole-layer patch, which also carries other slots' information, reported only as a reference.
- If decoding suggests no candidate, the test is skipped and the note says why.

## Proposed first figure

`figures/heldout.png`.

**Top row:** held-out final accuracy against updates, one panel per condition (`final`, `inter`, `shuf`). Each seed is its own coloured line. Training final accuracy is a dashed line in the same colour. Reference lines:
- chance (0.125);
- the neighbour vote (0.077);
- the oracle-selected nearest-program bound (0.453);
- the P2 positive control's held-out accuracy.

**Bottom row:**
- held-out accuracy at 10,000 updates by repetition pattern and per program, one dot per seed per condition;
- held-out step1 and step2 accuracy in `inter` and `shuf`.

Failed or capped runs appear as gaps, labelled with their run ID.

## Budget and stopping rules

The budget is 25 min of elapsed numerical time.

**Planned initial spend, about 6 min:**
- one calibration run (`train.py` at 1,000 updates, seed 900000+, about 10 s);
- P1, 512 rows at 10,000 updates (about 45 s);
- P2 (a few seconds);
- 9 main runs at 2 concurrent (about 3 min);
- probes and interventions (about 1 min).

That leaves about 15 min for the one follow-up.

**Rules.**
- Timeouts are about 3× the calibrated time, and at most 900 s.
- Capped and failed runs are kept and shown. An infrastructure failure gets at most one re-attempt, under a new attempt ID.
- No recipe changes inside the initial comparison.
- I stop and ask the coordinator before exceeding 25 min.

## Literature read

Liška, Kruszewski & Baroni (2018), *Memorize or generalize? Searching for a compositional RNN in a haystack* (arXiv 1802.06467): the abstract and the full text from the arXiv PDF.

What bears on this design:
- **What they held out.** They held out inputs to seen compositions, two of eight per composed task, not whole compositions. Footnote 8 says that networks trained on a subset of composed tasks did not generalize to unseen composed tasks, with data not shown. That is this lane's setting.
- **Intermediate outputs.** Their targets included the intermediate step's output, and they say results were stable without it.
- **Supervision.** Hand-designed supervision of the hidden state gave 96% zero-shot accuracy.
- **Rate of success.** Without it, about 2% of 50k LSTMs exceeded 80% on held-out inputs.
- **What made the difference.** Success depended on the order of task presentation, not on initialization. This lane trains full-batch, so no such order exists, and the init seed is the only within-condition randomness.

## Follow-up

To be written here, with its motivation and prediction, after the first figure and before any follow-up run.
