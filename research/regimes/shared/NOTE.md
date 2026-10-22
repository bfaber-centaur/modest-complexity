# R2 shared subcomputations: note (M4)

Design and the follow-up's prediction: [QUESTION.md](QUESTION.md). Commands and budget: [README.md](README.md).

## Question

When a network that learned y1 = (x1⊕x2)⊕x3 learns y2 = (x1⊕x2)∧x4 faster, is it reusing a representation of z = x1⊕x2? Or does a z-free task give the same benefit?

## Design

- **Network.** 10 → 64 → 64 ReLU MLP (`common/nets.py`), full-batch Adam at 3×10⁻³, with the optimizer reset at every phase.
- **Data.** A pool of 512 inputs and a fixed test set of 512. Each arm is pretrained on the pool for 2,000 updates, then trained on y2 for 2,000 updates on nested D_n with n = 8 to 512.
- **Primary metric.** Held-out accuracy on the x4 = 1 half, where y2 = z and every z-blind rule scores 0.5.
- **Two transfer modes.**
  - Fine-tuning: trunk and a new head.
  - Frozen readout: a new head only, on the phase-1 features.
- **Pretraining arms.**
  - `rel`: y1.
  - `part`: x2⊕x3⊕x4. It shares two of y2's bits, including the gate x4, and has no z.
  - `unrel`: x5⊕x6⊕x7.
  - `same`: y2 itself, the positive control.
  - `split`: the follow-up. A 4-class pair (x1⊕x3⊕x5, x2⊕x6⊕x7) that shares exactly {x1, x2} with y2 and has no z.
- **Other arms.** `fresh`, plus joint versus separate training.
- **Replicates.** Three paired replicates. Each arm shares the init and data seeds within a replicate.
- **Runs.** 29 in all (smoke, calibration, 24 initial, 3 follow-up): none failed or capped, 4.1 min of the 25-minute budget.

## Observations

All numbers are paired per replicate, at the final step, from `figures/final.csv`.

1. **Controls hold.**
   - Every pretraining run reaches 1.00 test accuracy, and its predictions flip only on its task's bits.
   - `fresh` reaches 1.00 at n = 512.
   - `same` transfers: frozen readout is 1.00 at n ≥ 32.
   - All arms are at 1.00 by n = 256 under fine-tuning, so differences live at n = 16–128.
2. **Frozen features show rel-specific transfer.** This is `figures/transfer.png`, bottom row.
   - A linear y2 head on y1 features reaches 0.98–1.00 at n ≥ 128.
   - `rel` beats `part` on every replicate at n = 32, 64 and 128: +0.17/+0.23/+0.35, +0.27/+0.14/+0.26 and +0.22/+0.30/+0.15.
   - `rel` beats `split` on every replicate at the same n: −0.19 to −0.38 for `split` minus `rel`.
   - `unrel` features do worse than random features: −0.09 to −0.16 against `fresh` at n = 64–128.
3. **Fine-tuning tells a different story** (top row).
   - `rel` beats `fresh` by +0.12 to +0.32 at n = 32–64, on all replicates.
   - `part` (no z, shares the gate bit) transfers about as well. `rel` minus `part` is +0.13/−0.04/+0.04 at n = 64, and positive on all replicates only at n = 16, where pattern coverage differs by replicate (reviewer C2.1).
   - `split` (shares {x1, x2}, no z) gives nothing over `fresh`: −0.04 to +0.09 at n = 32–64. That contradicts my pre-stated prediction 2, that bit overlap alone would transfer.
   - `unrel` transfers negatively: −0.09 to −0.10 at n = 64.
4. **Frozen beats fine-tuning at small n.** At n = 32, fine-tuning ends below the frozen readout for `rel` (0.70/0.71/0.94 against 0.77/0.81/0.95) and for `same` (0.79/0.89/0.99 against 1.00). The fine-tuning arm therefore partly measures how much phase 2 disturbs pretrained features, as the coordinator suggested.
5. **Probes: z is linearly decodable but not singled out.** Ridge probes, layer 2, test accuracy (`figures/followup.png`, right panel).
   - y1-pretrained features encode z, x1⊕x3 and x2⊕x3 all at 1.00. The untrained network scores 0.80–0.84, `part` 0.80–0.93, `unrel` 0.68–0.71.
   - `split` scores 0.96/0.89/0.87. Its best replicate also has its best frozen readout (0.97 at n = 512, against 0.78 and 0.82).
   - Across arms and replicates, frozen-readout success tracks linear z availability. That is decodability plus readout success, not a demonstration of use.
6. **Steering did not show reuse.** Moving each input to the other z class mean along a mean-difference direction:
   - Never changed y1 in the y1 network, at either layer, in any replicate. A uniform shift along a z direction cannot flip a linear readout of z⊕x3, so this test could not detect y1's use of z.
   - In every y2-trained network (all arms), the network's own layer-2 z direction flips y2 on 0.99–1.00 of x4 = 1 inputs and on 0.00–0.22 of x4 = 0 inputs. That holds for every y2 solver, not specifically for reuse. Its own layer-1 direction flipped nothing, except in one `unrel` replicate (0.46 of x4 = 1).
   - The y1 network's z direction, applied inside the fine-tuned `rel` network, flipped nothing at either layer. Its cosine with the final network's direction was 0.66–0.75 at layer 2 and 0.80–0.94 at layer 1.
   - The x1⊕x3, x5⊕x6 and random directions flipped nothing anywhere.
7. **Joint training** gave at most a small y2 benefit over `joint-unrel`: +0.03 to +0.12 at n = 64, mixed at n = 32. y1 alone and y1 in the joint network were similar (`figures/joint.png`).
8. **Irrelevant inputs** (`figures/flip.png`).
   - At n = 64, `rel` y2 predictions depend on x1 and x2 closest to the exact 0.5 (0.41–0.47 per bit), against `fresh` 0.24–0.30 and `unrel` 0.15–0.22.
   - The largest flip rate from any y2-irrelevant bit (x3, x5–x10) at n = 64, across replicates: `rel` 0.08, `part` 0.15, `fresh` 0.18, `unrel` 0.25.
   - At n ≤ 16 these reach 0.25–0.53. By n = 256 they are at most 0.002 in every arm.

## Reading

- **Frozen features.** The data favor "y1 pretraining builds a linearly available z (with its sibling XORs), and a linear y2 readout uses it". Controls with the same exposure, the same shared bits (`split`) or the same overlap count (`part`) do not reproduce it. This matches outcome 4, but the selective intervention that would separate 4a from 4b failed to be informative.
- **Fine-tuning.** The rel advantage is not specific to z: `part` matches it. `split` does not, which points at sharing the gate bit x4, or at fewer y2-irrelevant pretraining bits (`split` trains on four, `rel` and `part` on one), and not at the pair {x1, x2}.
- **Not shown.** That the fine-tuned network computes y2 through the representation y1 built.

## Limitations

- **Scale.** One architecture, one lr, one update count, three replicates.
- **`split` changes more than the pair.** It has four classes (more supervision) and four y2-irrelevant bits.
- **Coverage.** Small-n patterns differ by replicate (4/6/8 of 8 at n = 8).
- **Probes** are linear, on standardized features, fit on the pool.
- **Steering** used class-mean shifts, which cannot reveal use of z inside an XOR.

## Next question

Does an input-specific interchange intervention or ablation of the z subspace show selective use?

- **Frozen y1 features.** Ablate the z probe direction(s) from them and refit the readout. It should collapse to about 0.5 on x4 = 1, while ablating x1⊕x3 leaves it intact.
- **y1 network.** Patch the layer-1 z component from an input with z flipped.

Separately: does a smaller phase-2 lr, or a head warm-up before unfreezing, make fine-tuned transfer as z-specific as the frozen readout?
