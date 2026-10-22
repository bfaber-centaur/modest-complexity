# c3 scaffolding lane: question and pre-registration (M0)

- **Charter.** [../coordinator/charter/03-temporary-scaffolding.txt](../coordinator/charter/03-temporary-scaffolding.txt).
- **Assignment, pin, constants, seeds and budget.** [../coordinator/DECISIONS.md](../coordinator/DECISIONS.md) D2–D5, D7–D9.

This file was written before any c3 measurement by this lane. The evidence it uses is published records only:
- r1's 0–10,000-update metrics of the source runs;
- c2's time and generalization lanes (NOTEs, CSVs, REVIEW.md §2–3);
- the coordinator's pilot table (D5).

## Question

After learning with intermediate-state supervision, does composition persist when training continues with final-answer supervision only?

## Task, model and recipe (fixed)

- **Task.** The original task and split, `ops51000-ls51100`: 384 training rows (48 programs × 8 starts), and 128 held-out rows (16 unseen length-3 programs), split into N (86 rows) and ¬N (42 rows).
- **Model.** Tied transition and tied readout throughout, width 128, 19,208 parameters.
- **Recipe.** Full-batch Adam (lr 1e-3, b1 0.9, b2 0.999, eps 1e-8) and decoupled weight decay 0.1 on every parameter.
- **What changes.** Only the step-weight schedule (D5). Each branch carries its weights, both Adam moments and the step count from the switch checkpoint. The final-step weight stays exactly 1.
- **Init seeds.** 51200–51202 are the original inits, reused on purpose. They are three paired seeds on one task draw. Branches from the same checkpoint are not independent replicates.

## Arms

For each seed and each switch time S ∈ {1,000; 10,000} (D5: common, fixed and pilot-informed), every arm starts from the same full-state checkpoint: `r1-comp-rnn-inter-s<seed>-a1/ckpt-S`, recovered, with its SHA-256 in `../../composition-c2/recovered/RECOVERY.json`.

| arm | schedule from S | runs | ID |
|---|---|---|---|
| A, keep | weights [1, 1, 1] throughout | reused: r1 tied/inter for 0–10k, then `c2-time-tt-inter-*` for 10k–50k | – |
| B, drop | [0, 0, 1] from the update at S | new, 6 | `c3-scaf-drop-at{1k,10k}-s<seed>-a1` |
| C, taper | intermediate weights 1 → 0 linearly over 2,000 updates (the update at S + k uses 1 − (k+1)/2000; [0, 0, 1] from S + 1,999) | new, 6 | `c3-scaf-taper-at{1k,10k}-s<seed>-a1` |
| D, final-only from init | [0, 0, 1] from update 0 | reused: r1 tied/final for 0–10k, then `c2-time-tt-final-*` for 10k–50k | – |
| switch state | the checkpoint at S, evaluated with no further training | – | – |

**Exact runs.**
- **Script and flags.** Every new run is `common/c3train.py --resume <ckpt-S> --resume-sha256 <RECOVERY.json> --resume-config research/regimes/composition/runs/<source>/config.json`. Every recipe flag is explicit: `--task-id ops51000-ls51100 --transition tied --readout tied --step-weights 1 1 1 --hidden 128 --lr 0.001 --wd 0.1 --eval-every 50`.
- **Schedule.** `--switch-step S --switch-weights 0 0 1`, with `--taper-updates 1` for B or `2000` for C.
- **Length and logging.** `--updates 50000 --diag`.
- **Checkpoints.** `--checkpoint-steps` at the fixed times 10000 (S = 1k only), 20000, 30000, 40000 and 50000, which are the keep arm's existing checkpoints. Each branch also saves the end of its taper window, S + 2,000 (3,000 or 12,000), with no keep counterpart.
- **Updates.** 6 × 49,000 + 6 × 40,000 = 534,000.

**Joining.** A branch's first logged row (step S) must equal the r1 source's published row at S in every r1 key. The reused arms' joins at 10k were checked in c2 (time lane `checks.json`, reviewer §2).

## Measures

**Logged at every 50 updates, in every arm.**
- **Training rows.** Final accuracy and CE, plus step-1 and step-2 accuracy and CE: the shared readout applied to h1 and h2 on training rows.
- **Held-out rows.** Length-3 final accuracy and CE, and the N/¬N split.
- **Weights.** The step weights of the update taken (`step_weights`).
- **Diagnostics** (from S, through `--diag`): gradient and update L2 norms, Adam √v̂ (median, max, fraction below eps), and the train final margin.
- **Not available.** Arms A and D have no diagnostics before 10k.

**Frozen evaluation of checkpoints.** This is done at the fixed times in every arm: S, 10k, 20k, 30k, 40k and 50k, plus the branch-only taper ends. It follows c2 generalization's definition, exhaustively: every program of length L with every start symbol, 8·4^L rows, with the tied forward iterated L times and the final step read.
- **Lengths 1–2.** All 32 length-1 and all 128 length-2 states are training intermediate labels: every (start, op1) and (start, op1, op2) prefix occurs in training. So in A–C, lengths 1–2 measure retention of *previously supervised* states. In D they were never supervised, so they test a readout of unsupervised states. Neither is novel generalization.
- **Length 3.** Held-out (128 rows, plus N and ¬N) and training programs.
- **Lengths 4–6.** Novel programs. On this draw almost every one is a new function (c2 generalization).
- **Check.** The lane's evaluation must reproduce c2 generalization's published L = 4–6 numbers for the keep and final-only checkpoints at 10k and 50k.

**Retention and acquisition.** These are defined per seed, S, arm, measure m and fixed time t.
- **Branch delta.** Δ_branch(t) = m_arm(t) − m(S), the change from the branch point.
- **Matched delta.** Δ_keep(t) = m_arm(t) − m_A(t), the difference from continued supervision at the same total update count.
- **Final-only delta.** Δ_D(t) = m_arm(t) − m_D(t), the difference from final-only from init at the same total update count.
- **Degradation, descriptive and per measure.** Δ_keep(50,000) ≤ −0.05, five percentage points, on held-out length-3 final, ¬N, or the L = 4, 5 or 6 accuracy, each reported separately. "Retained" means |Δ_keep(50,000)| < 0.05. A branch can exceed keep only where keep is below 0.95. Full curves are shown, so the threshold does not replace the data.
- **Acquisition** (for S = 1k, where every seed is partial). This is Δ_branch and Δ_D over time, plus the c2 time lane's threshold rule: the first of five consecutive logged evaluations with held-out final accuracy ≥ 0.95, at 50-update resolution, censored at 50,000.
- **Starting seeds.** Each is marked individually at each S.
  - At 1k all three are partial: held-out 0.34–0.36, ¬N 0.00–0.02.
  - At 10k, 51200 and 51202 are near-perfect at length 3 (1.000, ¬N 1.00). 51201 is partial (0.961, ¬N 0.88).
  - At L = 6, all three are partial at 10k: 0.86 / 0.56 / 0.72 (c2 generalization).

**Exposure and loss scale.** From each run's `config.json` → `exposure`: labels per head (the number of updates with nonzero weight, and weighted row presentations) from init and in this run. The weighted training loss is also reported: the sum over heads of weight times train CE. More labels are part of this intervention, not a confound to be claimed away.

**Neighbour compatibility of wrong held-out length-3 predictions.** This is the c2 time lane's measure, at the fixed times.

## Predictions

These were written after reading only the sources listed at the top. Confidence is low throughout.

1. **S = 10k, length 3.**
   - In all three seeds, both B and C stay within 5 pp of keep at 50k on held-out length-3 and ¬N, so they are "retained".
   - The partial seed 51201 still reaches the threshold by 50k.
   - Reasoning: final-only supervision keeps the length-3 answers on the training set. A shared transition that already composes is one solution of that loss, and nothing pushes away from it.
2. **S = 10k, lengths 1–2.**
   - With no loss on h1 and h2, the shared readout's accuracy on those states declines in B and C.
   - In at least two seeds, training step-1 or step-2 accuracy falls below 0.90 by 50k.
   - Keep stays at 1.0.
3. **S = 10k, lengths 4–6.**
   - Keep reaches about 1.0 by 20k–50k (c2).
   - B and C are degraded at L = 6 by at least 5 pp at 50k in at least two seeds: length extension stops improving, or declines, without the intermediate labels that tie the code of h1 and h2 to that of h3.
   - This is the least confident prediction. The opposite, retention at length 6, would be equally informative.
4. **S = 1k, acquisition.**
   - B and C keep rising after the switch, faster than D at equal total updates but slower than keep.
   - At 50k, held-out length-3 is between D (0.56–0.70) and keep (1.00) in all three seeds, so it is degraded relative to keep.
   - At most one seed meets the threshold.
   - ¬N at 50k is below 0.5. L = 4–6 at 50k is below 0.5.
5. **Taper against drop.** C is within 5 pp of B on every endpoint measure at both S. A 2,000-update taper is short against 40,000–49,000 updates after it.
6. **Optimizer.**
   - After a drop, the carried √v̂ still reflects the three-head gradients, so the update norm falls at first. Then √v̂ decays toward the new gradient scale over a few thousand updates.
   - The eps regime (every coordinate's √v̂ < eps) begins within ±5,000 updates of the c2 onset, 27,800–29,650, in every branch.

## Interpretation rules, fixed in advance

These hold for this task, recipe, horizon and three seeds.
- **Lasting benefit, a training-history effect.** At a given S, B (and C) stay above D by at least 5 pp at 50k on held-out length-3, in all three seeds. This does not show discovery without the extra labels, universal modularity, or a stable asymptote.
- **Persistence, retention.** At S = 10k, a measure is retained in all three seeds. Persistence at length 3 together with degradation at lengths 4–6 is reported as its own outcome, not as persistence.
- **Collapse.** A branch ends within 5 pp of D at 50k on held-out length-3.
- **Further improvement.** A branch exceeds its own switch state by at least 5 pp (Δ_branch ≥ 0.05) at 50k on a measure where it was below 0.95 at S.
- **Mixed.** Outcomes that differ across seeds are reported per seed and not pooled.
- **Two limits.** A drop mixes a change of loss direction and magnitude with optimizer-history effects, so the B–A difference is not attributed to "supervision" alone. Neither a plateau nor a crossing at 50k is an asymptote.

## Controls and checks

- **Pin and tests.** The pin `be41d16` is unedited. The shared tests `test_c3.py` (8) and `test_c2.py` (10) run in their own processes on this host and pass. They cover the schedule boundaries, gradient linearity in the weights, the drop branch equal to final-only from the same state, a bitwise mid-taper resume, and the r1 recipe check.
- **Before launch.** An on-host calibration with c3train (below).
- **Per-run checks.**
  - The joining check.
  - `step_weights` in `metrics.jsonl` against the declared schedule at every logged row, including S, S + 1,999 and S + 2,000.
  - Exposure against a recount.
- **Lane tests.** `test_scaf.py`, run in its own process, checks the retention arithmetic, the frozen-length evaluation against a plain per-row loop, and the schedule recount on hand-made inputs.
- **No checkpoint selected by held-out score.** All comparisons are at the fixed times.

## First figure (planned)

`figures/scaf_main.png`, with SVG and CSV beside it. There are two column blocks, S = 1k and S = 10k, colour shows the arm (A keep, B drop, C taper, D final-only), and every seed is drawn.

1. **Supervision schedule.** The intermediate weight against update.
2. **Held-out length-3 final accuracy and ¬N.** Against update aligned at the switch (t − S), and at equal total updates from 0 to 50k.
3. **Train losses.** Train final CE and the weighted training loss, on a log scale, with train step-1/step-2 accuracy (lengths 1–2, previously supervised).
4. **Frozen lengths 1–6 at the fixed times.** Accuracy at L = 4, 5, 6 against total updates, plus the length curve at 50k.
5. **Diagnostics.** Update norm, gradient norm, and median and max √v̂ against eps.

## Follow-up (to be appended before running)

There is one follow-up, proposed after the first figure and approved by the coordinator. The coordinator sized it in D8, with a 20-minute budget.
- **Default.** If B differs from A by at least 5 pp at 50k on any endpoint measure at one S, run reset-versus-carried moments in both keep and drop at that S. Both arms are reset at S, with a matched carried control (needs a coordinator re-pin).
- **Alternative.** One explicitly justified loss-scale control. A changed loss scale can move the decay balance and the Adam eps regime, and that will be reported.
- **No follow-up.** If neither is justified, there is none.

## Budget and calibration

- **Cap.** 40 minutes initial, plus 20 for the follow-up (D8): elapsed wall time with at least one job running, `--workers 2`.
- **Calibration first.** `c3-scaf-calib-{drop,taper}-s900210-a1`: fresh init (seed 900210), 2,000 updates, a switch at 500 (taper 1,000 for the taper run), `--diag`, two concurrent jobs.
- **Expected cost.** 534,000 updates at 4–8 ms per update is about 18–36 minutes with two workers.
- **Timeouts.** About 3× the calibrated run time, at most 900 s per run.
- **Reporting.** The coordinator is told at 80%.

## Deliverables

- **Records.** QUESTION.md, ADAPTIVE.md (dated adaptive entries), `plan.json` and `registry.jsonl`.
- **Code.** `scaf_analyze.py` (publish, check, frozen evaluation, measure, figure) and `test_scaf.py`.
- **Runs.** `runs/<ID>/` with the small run files and `MANIFEST.json`. Checkpoints at the fixed times are committed.
- **Figure.** The primary figure as PNG, SVG and CSV.
- **Write-up.** README.md, NOTE.md and a reviewer handoff list.
