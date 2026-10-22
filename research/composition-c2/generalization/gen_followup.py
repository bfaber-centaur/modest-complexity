#!/usr/bin/env python3
"""Follow-up (QUESTION.md "Follow-up", approved at 99c83c8): anatomy of the first errors of the
11 frozen tied/inter models at steps 4-6 of exhaustive length-6 programs.

For every row whose first wrong step t* is 4, 5 or 6: the transition at t* (true state before
step t*, op at t*), the wrong answer, and whether ops t*-2..t* form a held-out program of the
model's split. Rates use the rows "at risk" at step t (correct through step t-1). Input-level
description of which transitions fail first; no claim about mechanism.

  anatomy.csv   per model x transition: first errors at steps 4, 5, 6 and at-risk counts
  windows.csv   per model x length 4-6: accuracy of programs with / without a held-out window
  summary.json  measures 1-4 and the F1-F3 verdicts per model

uv run --locked python research/composition-c2/generalization/gen_followup.py --out DIR
"""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gen_eval  # noqa: E402  (puts common/ and regimes/common/ on sys.path)
import c2task  # noqa: E402

STEPS = (4, 5, 6)
REP_INTER = sorted(HERE.glob('runs/c2-gen-inter-ops*'))


def ranks(x):
    """Average ranks (ties share their mean rank), for a Spearman correlation without scipy."""
    x = np.asarray(x, float); order = np.argsort(x, kind='stable'); r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):
        r[x == v] = r[x == v].mean()
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def held_window(ops, held, end):
    """Rows whose ops at 0-based positions end-2..end form a held-out program."""
    return np.array([tuple(map(int, w)) in held for w in ops[:, end-2:end+1]])


def anatomy(e):
    T = c2task.load(e['task_id'])
    P, held = T['P'], set(map(tuple, T['held']))
    p, _, _ = gen_eval.load_checked(e['path'], e['sha256'])
    d = c2task.rows(P, c2task.programs(gen_eval.MAX_LEN))
    pred = np.stack([z.argmax(1) for z in gen_eval.fwd(p, d['ops'], s=d['s'])[1]], 1)
    ok = pred == d['states']
    first = np.where(ok.all(1), 0, (~ok).argmax(1) + 1)
    before = np.concatenate([d['s'][:, None], d['states']], 1)  # before[:, t-1] = state before step t
    trans = before[:, :-1]*c2task.N_OP + d['ops']               # transition id x*4 + k at each step
    err = np.zeros((32, len(STEPS)), int); risk = np.zeros((32, len(STEPS)), int)
    wrong_op = win = win_risk = n_first = n_risk = 0
    for j, t in enumerate(STEPS):
        at_risk = first == 0
        at_risk |= first >= t
        f = first == t
        err[:, j] = np.bincount(trans[f, t-1], minlength=32); risk[:, j] = np.bincount(trans[at_risk, t-1], minlength=32)
        x, k, yhat = before[f, t-1], d['ops'][f, t-1], pred[f, t-1]
        alt = np.stack([P[kk, x] for kk in range(c2task.N_OP)], 1)  # every op applied to the true state
        alt[np.arange(len(k)), k] = -1
        wrong_op += int((alt == yhat[:, None]).any(1).sum())
        hw = held_window(d['ops'], held, t-1)
        win += int(hw[f].sum()); win_risk += int(hw[at_risk].sum())
        n_first += int(f.sum()); n_risk += int(at_risk.sum())
    pooled = err.sum(1)
    top = np.sort(pooled)[::-1]
    rate = np.where(risk > 0, err / np.maximum(risk, 1), np.nan)
    s56 = spearman(rate[:, 1], rate[:, 2])
    s45 = spearman(rate[:, 0], rate[:, 1]) if err[:, 0].sum() >= 20 else None
    share = lambda n: float(top[:n].sum() / max(pooled.sum(), 1))
    summary = dict(task_id=e['task_id'], init_seed=e['init_seed'], first_errors_steps_4_6=int(pooled.sum()),
                   first_errors_by_step=dict(zip(map(str, STEPS), map(int, err.sum(0)))),
                   top1_share=share(1), top4_share=share(4), top8_share=share(8),
                   spearman_rate_step5_vs_step6=s56, spearman_rate_step4_vs_step5=s45,
                   wrong_answer_is_other_op_on_true_state=wrong_op / max(n_first, 1),
                   first_error_window_is_held_out=win / max(n_first, 1), at_risk_window_is_held_out=win_risk / max(n_risk, 1))
    summary.update(F1_top4_ge_0p5=summary['top4_share'] >= .5, F2_rho_gt_0p5=bool(s56 > .5),
                   F3_window_share_ge_0p375=summary['first_error_window_is_held_out'] >= .375)
    rows = [dict(run_id=e['run_id'], task_id=e['task_id'], init_seed=e['init_seed'], state_before=x, op=k,
                 **{f'first_errors_step{t}': int(err[x*4+k, j]) for j, t in enumerate(STEPS)},
                 **{f'at_risk_step{t}': int(risk[x*4+k, j]) for j, t in enumerate(STEPS)},
                 first_errors_steps_4_6=int(pooled[x*4+k])) for x in range(c2task.N_SYM) for k in range(c2task.N_OP)]
    wins = []
    for L in (4, 5, 6):
        dl = c2task.rows(P, c2task.programs(L))
        okl = gen_eval.fwd(p, dl['ops'], s=dl['s'])[1][-1].argmax(1) == dl['states'][:, -1]
        any_held = np.zeros(len(okl), bool)
        for end in range(2, L):
            any_held |= held_window(dl['ops'], held, end)
        for flag in (True, False):
            sel = any_held == flag
            wins.append(dict(run_id=e['run_id'], task_id=e['task_id'], init_seed=e['init_seed'], length=L,
                             contains_held_out_window=flag, n_rows=int(sel.sum()), accuracy=float(okl[sel].mean())))
    return summary, rows, wins


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    entries = [e for e in gen_eval.recovered_entries() if e['condition'] == 'inter'] + gen_eval.run_dir_entries(REP_INTER, 10000)
    assert len(entries) == 11 and all(e['condition'] == 'inter' for e in entries)
    summaries, rows, wins = {}, [], []
    for e in entries:
        s, r, w = anatomy(e)
        summaries[e['run_id']] = dict(s, sha256=e['sha256']); rows += r; wins += w
    verdict = {f: sum(s[f] for s in summaries.values()) for f in ('F1_top4_ge_0p5', 'F2_rho_gt_0p5', 'F3_window_share_ge_0p375')}
    gen_eval.write_csv(args.out/'anatomy.csv', rows); gen_eval.write_csv(args.out/'windows.csv', wins)
    (args.out/'summary.json').write_text(json.dumps(dict(models_meeting=verdict, of=len(summaries), per_model=summaries), indent=1))


if __name__ == '__main__':
    main()
