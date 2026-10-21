# e2 dynamics lane

Session `e2-dy-cloud-1` ("Width and training-time map" thread), branch `claude/project-thread-j8mgkm`, base `48ea420` on `swarm/exploration` (contains the e2 base `6aaa3a2`; shared code `fe284a5` unchanged). Question log: [QUESTION.md](QUESTION.md). Interpretations, kept separate: [INTERPRETATIONS.md](INTERPRETATIONS.md).

Execution budget (D3): e1 usage stays as recorded in [research/size-time/README.md](../../size-time/README.md) (21 runs, 22 min 57 s wall with two workers). e2 so far: analysis only, under 1 minute of numerical execution; no e2 runs yet.

Audit status: nothing in this directory has been audited. The numbers come from this lane's own forward pass (`margins.py`), which checks every consumed checkpoint against its e1 manifest hash, every checkpoint's own-split audit accuracy against the run's logged value, and 51 common-set accuracies against the coordinator's `common_set_scores.csv` (all agree to 1e-12).

## Packet 1: where the correct label stands on pairs neither share trained on

Question: on the 483 pairs that neither the 30% nor the 50% training set contains, how does the correct label's logit compare with the strongest wrong label, separately for labels seen with an operand of the pair in training and labels never seen with either operand, and when do the two shares start to differ?

![share comparison](figures/share/margins.png)

Look before reading: compare the grey and blue lines in "beats every N label" and in the S-minus-N gap panel, and find the step where they separate.

Label sets, identical for every run because the training sets nest (30% ⊂ 50%): for a common pair (a, b), S = labels of 30% training pairs (a, b') or (a', b) (mean 16.0 labels, 4-25); D = labels of such pairs that only the 50% set contains (mean 6.7, 1-13); N = the remaining wrong labels, never seen with a or b (mean 7.2, 1-20). No set is empty for any pair, and the correct label is in none (for addition it cannot be; asserted). A run's own operand-associated set is S at 30% and S ∪ D at 50%, so D is the only place the two masks differ. "Beats X" = no label in X has a higher logit; the dashed no-preference line is the mean of 1/(|X|+1). For the 50% runs, "beats every N label" is the audit's restricted-rank "top among absent" statistic on the same pairs (no-preference value 0.135 in both).

Observations, width 128, three init seeds per share (ranges over seeds):

| step | share | accuracy | beats every N | beats every S | margin over N | margin over S | gap S − N | gap D − N |
|---:|---|---|---|---|---|---|---|---|
| 1,000 | 30% | 0.000 | 0.15-0.16 | 0.00 | −1.39 to −1.30 | −5.69 to −5.39 | 4.06-4.38 | −0.22 to −0.10 |
| 1,000 | 50% | 0.000 | 0.17-0.21 | 0.00 | −1.38 to −1.17 | −5.60 to −5.24 | 4.08-4.20 | 2.35-2.66 |
| 2,000 | 30% | 0.000 | 0.14-0.16 | 0.00 | −1.53 to −1.32 | −5.96 to −5.86 | 4.35-4.50 | −0.26 to −0.11 |
| 2,000 | 50% | 0.002 | 0.29-0.39 | 0.00 | −0.87 to −0.52 | −4.97 to −4.34 | 3.81-4.04 | 2.32-2.50 |
| 5,000 | 30% | 0.000 | 0.16-0.19 | 0.00 | −1.59 to −1.41 | −6.67 to −6.48 | 4.97-5.24 | −0.21 to −0.12 |
| 5,000 | 50% | 0.02-0.07 | 0.62-0.78 | 0.02-0.09 | 0.53-1.53 | −3.34 to −2.21 | 3.68-4.06 | 2.12-2.59 |
| 10,000 | 30% | 0.000 | 0.22-0.24 | 0.00 | −1.51 to −1.42 | −7.24 to −7.14 | 5.70-5.88 | −0.16 to −0.05 |
| 10,000 | 50% | 0.19-0.39 | 0.88-0.95 | 0.21-0.41 | 2.38-3.34 | −1.43 to −0.44 | 3.75-3.88 | 2.31-2.37 |
| 50,000 | 30% | 0.000-0.004 | 0.29-0.37 | 0.00 | −1.86 to −1.11 | −9.62 to −9.35 | 8.16-8.25 | −0.37 to −0.15 |
| 50,000 | 50% | 0.88-0.91 | 0.99 | 0.89-0.92 | 7.81-8.25 | 3.10-3.66 | 4.54-4.70 | 2.56-2.67 |

Margins and gaps are medians over the 483 pairs, in logits; "gap S − N" is the strongest S label's logit minus the strongest N label's.

- Up to 1,000 updates the two shares are indistinguishable on every column except D, which the 50% runs already treat as operand-associated (gap D − N 2.4-2.7, against −0.2 to −0.1 at 30%).
- Between 1,000 and 2,000 updates they separate, before any 50% run answers held-out pairs: the 50% runs' margin over N rises by 0.4-0.8 logits and their "beats every N" fraction goes from 0.17-0.21 to 0.29-0.39, while the 30% runs' stay flat.
- At 50% the S − N gap stays between 3.7 and 4.8 from 1,000 to 50,000 updates, and the margin over S follows the margin over N about 4 logits behind. At 30% the gap roughly doubles (4.1-4.4 to 8.2) while the margin over N stays between −1.9 and −1.1.
- At 50%, 57-71% of common pairs at 5,000 and 10,000 updates beat every label outside the run's own set but lose to one inside it (lower-left panel). Wrong pairs in that state become correct by the next published checkpoint more often than other wrong pairs: 5,000 to 10,000, 0.24-0.42 against 0.10-0.15; 10,000 to 15,000, 0.38-0.59 against 0.13-0.19 ([margins_transitions.csv](figures/share/margins_transitions.csv)). They are also most of the wrong pairs then (61-77% at 5,000), and checkpoints are 5,000 updates apart.
- Control within the 30% runs: D and N are both outside their own set, and they are treated alike (gap D − N between −0.4 and 0.1 throughout; "beats every D" tracks "beats every N" against their no-preference values).

Width comparison at the 30% share, same pairs and label sets (S is each run's own set; D and N are both outside it):

![width comparison](figures/width30/margins_width30.png)

| width | beats every N at 50k | margin over N at 50k | margin over S at 50k | gap S − N at 50k |
|---:|---|---|---|---|
| 16 | 0.11-0.18 | −3.07 to −2.76 | −6.77 to −6.60 | 3.46-3.82 |
| 32 | 0.18-0.22 | −3.92 to −2.89 | −9.57 to −9.04 | 5.60-5.96 |
| 64 | 0.26-0.31 | −2.91 to −2.38 | −11.06 to −10.31 | 7.53-8.21 |
| 128 | 0.29-0.37 | −1.86 to −1.11 | −9.62 to −9.35 | 8.16-8.25 |
| 256 | 0.37-0.42 | −1.00 to −0.58 | −8.96 to −8.77 | 8.07-8.35 |
| 512 | 0.47-0.50 | −0.14 to −0.01 | −8.32 to −8.13 | 8.20-8.30 |

- At 30%, "beats every N" at 50,000 rises with width from about the no-preference value (0.135) at width 16 to 0.47-0.50 at width 512, and the margin over N from about −3 to about 0. After step 0, no run beats every S label on more than 1.1% of pairs.
- The S − N gap grows during training at every width and reaches about 8 logits for widths 64-512 (3.5-3.8 at width 16, 5.6-6.0 at width 32). Width 512 at 30% ends at the "beats every N" level that the 50% width-128 runs pass between 2,000 and 5,000 updates, with an S − N gap twice theirs.

Checkpoints: published checkpoints only. The e1 width-128 runs `e1-st-add-w128-i30001-a1` and `-i30002-a1` (this session's) gained 9 and 14 checkpoints in [e1-checkpoints/](e1-checkpoints/), copied from this session's local run directories after checking each SHA-256 against the run's e1 manifest (`publish_e1_checkpoints.py`; `supplement.json` records the manifest hash). The coordinator's `e1-co-add-w128-i30000-a1` and the 50% runs have their published steps only (1,000, 2,000, 5,000, 10,000, 15,000, 20,000, 30,000, 40,000, 50,000 for the 50% runs). Source tables: [share/margins.csv](figures/share/margins.csv), [width30/margins_width30.csv](figures/width30/margins_width30.csv), set sizes in `sizes.json`. Vector versions: `margins.svg`, `margins_width30.svg`.

Runs: 30% width 128 `e1-co-add-w128-i30000-a1`, `e1-st-add-w128-i30001-a1`, `e1-st-add-w128-i30002-a1`; 50% width 128 `e1f2-co-add-w128-i3000{0,1,2}-tf50-a1`; width comparison: all 18 e1 `add` runs at widths 16-512 (`research/coordinator/runs/`, `research/size-time/runs/`).

Regenerate:

```
C=research/coordinator/runs; P=research/size-time/runs
uv run --locked --group analysis python research/e2/dynamics/margins.py \
  --group "30% train" $C/e1-co-add-w128-i30000-a1 $P/e1-st-add-w128-i30001-a1 $P/e1-st-add-w128-i30002-a1 \
  --group "50% train" $C/e1f2-co-add-w128-i3000{0,1,2}-tf50-a1 --out research/e2/dynamics/figures/share
run() { for s in 30000 30001 30002; do for x in $C/e1-co-add-w$1-i$s-a1 $P/e1-st-add-w$1-i$s-a1; do [ -d "$x" ] && echo $x; done; done; }
args=(); for w in 16 32 64 128 256 512; do args+=(--group "width $w" $(run $w)); done
uv run --locked --group analysis python research/e2/dynamics/margins.py "${args[@]}" --name margins_width30 \
  --title 'Correct-label margins on the 483 common pairs: 30% training share, widths 16-512' --out research/e2/dynamics/figures/width30
```

Limitations: one split and one data seed; checkpoints 1,000-10,000 updates apart around the 50% onset (the coordinator's every-1,000 checkpoints of the 50% runs are hash-listed but not published); medians summarize per-pair distributions; logit scale grows with training, so margin sizes are compared within a step, not across steps.
