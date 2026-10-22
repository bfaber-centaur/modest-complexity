# Role: R1 investigator, composing small programs

Directory `research/regimes/composition/`. Run IDs `r1-comp-…`. Seeds 51000–51999 (PLAN §1). Label `r1-comp-cloud-1`. Budget 25 min.

## Bobby's brief (verbatim)

> Question: Under what conditions does learning primitive operations support unseen compositions?
>
> Task:
> - Eight symbols.
> - Four fixed random permutations acting as operations.
> - Input: starting symbol and ordered operation sequence.
> - Target: result of executing the sequence.
> - Begin with length-three programs: 512 total input/program combinations.
>
> Design:
> - Hold out entire operation sequences, not merely starting symbols.
> - Every primitive must appear in every position during training.
> - Balance shorter subsequences where feasible; report actual coverage.
> - Check for equivalent programs, permutation identities, and accidental leakage.
> - Begin with one small architecture that does not hard-code the correct interpreter.
> - Compare final-answer supervision with intermediate-state supervision.
> - Keep data exposure explicit and account for the additional supervisory signal.
>
> Measurements:
> - Accuracy on held-out compositions, broken down by structure.
> - Intermediate-state decoding as a descriptive measurement.
> - If decoding suggests a mechanism, attempt a targeted causal intervention.
> - Do not equate decodability with use.
>
> First figure: held-out composition performance across seeds and supervision conditions.
>
> Starting literature: https://arxiv.org/abs/1802.06467

Read the paper's abstract, and its body if it bears on a concrete design choice. Record what you read and what you could not retrieve.

## Coordinator's notes

These are things to consider, not requirements. Your QUESTION.md decides.

**A balanced held-out set exists.**
- There are 4³ = 64 programs. Take a 4×4 Latin square L and hold out the 16 programs (a, b, L[a][b]). The remaining 48 training programs then contain every ordered op pair exactly 3 times at each of the position pairs (1,2), (2,3) and (1,3), and every op 12 times at every position.
- Randomizing the square is the split's data seed.
- Whatever you choose, report coverage as counts.

**Leakage and equivalence checks worth computing before training**
- Each permutation's cycle type, fixed points and order. Whether any is the identity or an involution.
- Which pairs commute.
- Collisions among the 64 length-3 products, and among length-2 products.
- Per held-out program, the best agreement over 8 starts with any training program's full table. This is a "nearest seen program" baseline.
- The accuracy of partial-program heuristics, such as dropping one op. Random permutations have fixed points, so these beat 1/8.

These baselines belong on the first figure next to chance, 1/8.

**Architecture**
- The simplest choice that does not hard-code the interpreter is an MLP on slot one-hots (start: 8; three op slots of 4). `nets.py` with two hidden layers fits.
- Intermediate supervision can be extra heads (state after op 1, after op 2) on the same last layer. That keeps the architecture identical across conditions and changes only the loss.
- A recurrent cell that applies one shared transition per op is a strong structural prior. It is a natural adaptive follow-up, not the starting point.

**Exposure accounting**
- Same inputs, same updates, same parameters, apart from the extra head weights, which you should state. Intermediate labels add two 8-way labels per example.
- Two controls separate the *information* in those labels from the *extra loss terms*:
  1. Shuffled intermediate labels: same count, no information.
  2. Training programs' length-1 and length-2 prefixes given as extra final-answer examples, with a blank-slot encoding. This is the same information in another form, but it changes the input format.
- Pick at most one for the initial comparison. Note that intermediate labels on the 48 training programs expose every (state, op) transition at positions 1 and 2.

**Structure breakdown.** Possible breakdowns of held-out accuracy:
- by position of the "novel" pair;
- by whether the overall permutation equals some training program's;
- by start symbol;
- by whether the answer equals a partial-program heuristic.

**Decoding is descriptive**
- Decode the state after op 1 or 2 from each hidden layer, on training programs and on held-out programs.
- Use input-layer and untrained-network decoders as baselines. The state after op 1 is a table of two one-hot slots, so linear decodability from the input itself is a real baseline.

**Causal test, if decoding suggests one**
- An interchange intervention that swaps only the decoded subspace between two inputs and checks whether the output follows the swapped intermediate state.
- Patching a whole layer also carries the later op slots' information, so it does not test the intermediate state.

**Outcomes worth distinguishing in advance**
- Held-out accuracy sits at the nearest-program or heuristic baseline in both conditions.
- Intermediate supervision lifts held-out accuracy but final-only does not.
- Both generalize.
- Seed-dependent: a few seeds generalize and most do not.

Also say what training accuracy must look like for each reading to hold.

## First figure

Held-out composition accuracy by supervision condition. Draw:
- each of three paired init seeds as a separate mark or trajectory over updates;
- train accuracy beside held-out accuracy;
- chance, the nearest-program baseline and the heuristic baselines;
- capped and failed runs explicitly.
