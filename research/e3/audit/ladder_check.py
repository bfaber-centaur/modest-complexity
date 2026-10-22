#!/usr/bin/env python3
"""Fifth e3 audit investigation (research/e3/audit/QUESTION-5.md): the dynamics lane's ladder
measures recomputed from the published checkpoints (T1), the tests of its claims (rules 1-4), and
two post-hoc yardsticks for rule 4 (labelled as such in the output).

The measures are written from the definitions in research/e3/dynamics/README.md and the
ladder.py docstring, not from its code: per run, on its ladder's evaluation pairs (the
complement of the ladder's 50% training set), accuracy; P(a+b over N), P(a+b over S),
P(S over N) from pairwise.py; the median over pairs with N non-empty of
(a+b logit - max over N) / sigma, sigma the population standard deviation of the pair's 30
wrong-label logits; and the mean over pairs of the fraction of the run's own-absent wrong labels
(not seen with either operand in the run's own training set) whose logit is above a+b.

uv run --locked python research/e3/audit/ladder_check.py OUT_DIR
"""
import csv
import itertools
import json
from pathlib import Path
import sys
import numpy as np

root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(root/'research/audit'))
import independent as ind  # noqa: E402
from pairwise import ladder_sets, operand_labels, preference  # noqa: E402

P, CONTROL = 31, 32000
SHARES = (.3, .4, .45, .5)
E3 = (41001, 41002, 41003)
LADDERS = E3 + (31000,)
INITS = (30000, 30001)
KEYS = ('acc', 'P_yN', 'P_yS', 'P_SN', 'margin_N_sigma', 'rank_absent')
HIGHER_IS_BETTER = dict(acc=True, P_yN=True, P_yS=True, margin_N_sigma=True, rank_absent=False)
DY = root/'research/e3/dynamics/runs'
REFERENCE = {(.3, 30000): 'research/coordinator/runs/e1-co-add-w128-i30000-a1',
             (.3, 30001): 'research/size-time/runs/e1-st-add-w128-i30001-a1',
             **{(s, i): f'research/e2/dynamics/runs/e2-dy-add-w128-i{i}-tf{round(100*s)}-a1' for s in (.4, .45) for i in INITS},
             **{(.5, i): f'research/coordinator/runs/e1f2-co-add-w128-i{i}-tf50-a1' for i in INITS}}


def run_dir(seed, share, init):
    return DY/f'e3-dy-add-w128-i{init}-tf{round(100*share)}-d{seed}-a1' if seed in E3 else root/REFERENCE[(share, init)]


def split(seed, share):
    a, b, y, _, sp = ind.regenerate_split(P, seed, CONTROL, share)
    return a, b, y, sp[0]


def weights(f):
    with np.load(f) as z:
        return [z[k] for k in 'UVW']


def measures(z, y, S, N, own_out):
    """z: logits (pairs, P) on evaluation pairs with labels y; S, N, own_out: boolean masks."""
    rows = np.arange(len(y))
    truth = np.eye(P, dtype=bool)[y]
    true = z[rows, y]
    sigma = z[~truth].reshape(len(y), P - 1).std(1)  # population standard deviation of the 30 wrong-label logits
    has_n, has_o = N.any(1), own_out.any(1)
    max_n = np.where(N, z, -np.inf).max(1)
    return dict(acc=float(np.mean(z.argmax(1) == y)),
                P_yN=float(np.nanmean(preference(z, truth, N)[0])), P_yS=float(np.nanmean(preference(z, truth, S)[0])),
                P_SN=float(np.nanmean(preference(z, S, N)[0])),
                margin_N_sigma=float(np.median(((true - max_n)/sigma)[has_n])),
                rank_absent=float(np.mean(((z > true[:, None]) & own_out).sum(1)[has_o]/own_out.sum(1)[has_o])),
                rank_all_wrong=float(np.mean(((z > true[:, None]) & ~truth).sum(1)/(P - 1))))


class Ladder:
    def __init__(self, seed, pairs=None):
        self.seed = seed
        self.a, self.b, self.y, small = split(seed, .3)
        self.train = {s: split(seed, s)[3] for s in SHARES}
        self.ev = np.array(sorted(set(range(P*P)) - set(self.train[.5].tolist()))) if pairs is None else pairs
        self.S, _, self.N, self.yy = ladder_sets(self.a, self.b, small, self.train[.5], self.ev, P)
        self.XA, self.XB = ind.one_hot_inputs(self.a[self.ev], self.b[self.ev], P)

    def score(self, w, share):
        own = operand_labels(self.a, self.b, self.train[share], P)
        ca, cb = self.a[self.ev], self.b[self.ev]
        own_out = ~(own[ca] | own[cb])
        own_out[np.arange(len(self.ev)), self.yy] = False
        return measures(ind.forward(w, self.XA, self.XB)[0], self.yy, self.S, self.N, own_out)


def t1_and_trajectories(lad):
    lane = {(r['run_id'], int(r['step'])): r for r in csv.DictReader(open(root/'research/e3/dynamics/figures/ladders.csv'))}
    rows, worst = [], {k: 0. for k in KEYS}
    for seed, share, init in itertools.product(LADDERS, SHARES, INITS):
        d = run_dir(seed, share, init)
        steps = sorted(int(f.stem.split('_')[1]) for f in d.glob('checkpoint_*.npz'))
        for step in steps if seed in E3 else [50000]:
            m = lad[seed].score(weights(d/f'checkpoint_{step:06d}.npz'), share)
            ref = lane[(d.name, step)]
            diff = {k: abs(m[k] - float(ref[k])) for k in KEYS}
            for k in KEYS:
                worst[k] = max(worst[k], diff[k])
            rows.append(dict(data_seed=seed, share=share, init_seed=init, run_id=d.name, step=step, **m,
                             max_diff_from_lane=max(diff.values())))
    return rows, worst


def rules_1_to_3(final, at30k):
    v = lambda seed, share, init, k: final[(seed, share, init)][k]
    r1 = {}
    for seed, init in itertools.product(LADDERS, INITS):
        bad = [(k, s0, s1) for k in HIGHER_IS_BETTER for s0, s1 in zip(SHARES, SHARES[1:])
               if not ((v(seed, s1, init, k) > v(seed, s0, init, k)) == HIGHER_IS_BETTER[k] and v(seed, s1, init, k) != v(seed, s0, init, k))]
        r1[f'{seed}/{init}'] = bad
    fall = {f'{seed}/{init}': v(seed, .3, init, 'P_SN') - v(seed, .5, init, 'P_SN') for seed, init in itertools.product(LADDERS, INITS)}
    rise45 = {f'{seed}/{init}': final[(seed, .45, init)]['acc'] - at30k[(seed, .45, init)]['acc'] for seed, init in itertools.product(E3, INITS)}
    return dict(
        rule1=dict(failed_rungs=r1, verdict='SUPPORTED LOCALLY' if not any(r1.values()) else 'see failed_rungs'),
        rule2=dict(fall_30_to_50=fall, verdict='SUPPORTED LOCALLY' if all(x >= .03 for x in fall.values()) else 'NOT SUPPORTED'),
        rule3=dict(acc_rise_30k_to_50k=rise45, verdict='SUPPORTED LOCALLY' if all(x >= .03 for x in rise45.values()) else 'NOT SUPPORTED'))


def null_criterion(rng, draws=1_000_000):
    """P(range of three two-init ladder means > all four within-ladder init differences) when every
    cell is independent N(0, 1)."""
    e = rng.standard_normal((draws, 4, 2))
    means, diffs = e[:, :3].mean(2), np.abs(e[:, :, 0] - e[:, :, 1])
    return float(np.mean(np.ptp(means, 1) > diffs.max(1)))


def two_way_f(x):
    """x: (ladders, inits), one value per cell. Ladder mean square over ladder x init mean square."""
    g = x.mean((-2, -1), keepdims=True)
    r, c = x.mean(-1, keepdims=True), x.mean(-2, keepdims=True)
    ms_l = x.shape[-1]*((r - g)**2).sum((-2, -1))/(x.shape[-2] - 1)
    ms_i = ((x - r - c + g)**2).sum((-2, -1))/((x.shape[-2] - 1)*(x.shape[-1] - 1))
    return ms_l/ms_i


def rule_4(final, traj, rng):
    out = dict(P0=null_criterion(rng))
    fired = []
    for k, share in itertools.product(KEYS, SHARES):
        x = np.array([[final[(s, share, i)][k] for i in INITS] for s in LADDERS])
        if np.ptp(x[:3].mean(1)) > np.abs(x[:, 0] - x[:, 1]).max():
            fired.append(f'{k} {share}')
    out['criterion_fires_in'] = fired
    x = np.array([[final[(s, .4, i)]['margin_N_sigma'] for i in INITS] for s in LADDERS])
    null = rng.standard_normal((1_000_000, 4, 2))
    out['two_way_40pct_sigma_margin'] = {
        name: dict(F=float(two_way_f(xx)), p=float(np.mean(two_way_f(nn) >= two_way_f(xx))))
        for name, xx, nn in (('three e3 ladders', x[:3], null[:, :3]), ('with 31000', x, null))}
    common = np.array(sorted(set.intersection(*(set(Ladder(s).ev.tolist()) for s in LADDERS))))
    fixed = {}
    for s, i in itertools.product(LADDERS, INITS):
        lc = Ladder(s, common)
        fixed[f'{s}/{i}'] = {k: v for k, v in lc.score(weights(run_dir(s, .4, i)/'checkpoint_050000.npz'), .4).items()
                             if k in ('margin_N_sigma', 'P_yN', 'rank_all_wrong', 'acc')}
    lead = {i: all(fixed[f'41002/{i}']['margin_N_sigma'] > fixed[f'{s}/{i}']['margin_N_sigma'] for s in (41001, 41003, 31000)) for i in INITS}
    out['fixed_pairs'] = dict(pairs=len(common), values=fixed, lead_41002_margin=lead,
                              lead_41002_P_yN={i: all(fixed[f'41002/{i}']['P_yN'] > fixed[f'{s}/{i}']['P_yN'] for s in (41001, 41003, 31000)) for i in INITS},
                              lead_41002_rank_all_wrong={i: all(fixed[f'41002/{i}']['rank_all_wrong'] < fixed[f'{s}/{i}']['rank_all_wrong']
                                                                for s in (41001, 41003, 31000)) for i in INITS},
                              verdict='SUPPORTED LOCALLY' if all(lead.values()) else 'CONTRADICTED' if not any(lead.values()) else 'AMBIGUOUS')
    by = {(r['data_seed'], r['share'], r['init_seed'], r['step']): r for r in traj}
    out['over_training'] = {step: {i: all(by[(41002, .4, i, step)]['margin_N_sigma'] > by[(s, .4, i, step)]['margin_N_sigma'] for s in (41001, 41003))
                                   for i in INITS} for step in (5000, 10000, 20000, 30000, 50000)}
    b = out['two_way_40pct_sigma_margin']
    out['verdict'] = ('CONTRADICTED' if out['fixed_pairs']['verdict'] == 'CONTRADICTED' else
                      'SUPPORTED LOCALLY' if all(v['p'] < .05 for v in b.values()) and out['fixed_pairs']['verdict'] == 'SUPPORTED LOCALLY'
                      else 'AMBIGUOUS')
    return out


def post_hoc(final, rng):
    """Not in QUESTION-5.md. (i) How much a 53-pair median sigma-margin moves with the choice of
    pairs: for each 40% run at 50k, the standard deviation over 2,000 random 53-pair subsets of
    its own evaluation pairs. (ii) The lane's criterion as a ratio, ladder-mean range over the
    largest init difference, per measure and share, with its null quantiles."""
    noise = {}
    for s, i in itertools.product(LADDERS, INITS):
        lad = Ladder(s)
        w = weights(run_dir(s, .4, i)/'checkpoint_050000.npz')
        z = ind.forward(w, lad.XA, lad.XB)[0]
        rows = np.arange(len(lad.ev))
        truth = np.eye(P, dtype=bool)[lad.yy]
        m = (z[rows, lad.yy] - np.where(lad.N, z, -np.inf).max(1))/z[~truth].reshape(len(rows), P - 1).std(1)
        noise[f'{s}/{i}'] = float(np.std([np.median(m[rng.choice(len(rows), 53, replace=False)]) for _ in range(2000)]))
    ratio = {f'{k} {share}': float(np.ptp([np.mean([final[(s, share, i)][k] for i in INITS]) for s in E3])
                                   / max(abs(final[(s, share, 30000)][k] - final[(s, share, 30001)][k]) for s in LADDERS))
             for k, share in itertools.product(KEYS, SHARES)}
    e = rng.standard_normal((1_000_000, 4, 2))
    null = np.ptp(e[:, :3].mean(2), 1)/np.abs(e[:, :, 0] - e[:, :, 1]).max(1)
    return dict(subset53_sd_of_median_sigma_margin=noise, criterion_ratio=ratio,
                null_ratio_quantiles={q: float(np.quantile(null, q)) for q in (.05, .25, .5, .75, .95)},
                null_ratio_above_1=float(np.mean(null > 1)), observed_above_1=sum(r > 1 for r in ratio.values()))


def main(out_dir):
    out = Path(out_dir)
    lad = {s: Ladder(s) for s in LADDERS}
    rows, worst = t1_and_trajectories(lad)
    final = {(r['data_seed'], r['share'], r['init_seed']): r for r in rows if r['step'] == 50000}
    at30k = {(r['data_seed'], r['share'], r['init_seed']): r for r in rows if r['step'] == 30000}
    res = dict(T1=dict(rows=len(rows), worst_abs_diff=worst, PASS=max(worst.values()) <= 1e-9))
    res.update(rules_1_to_3(final, at30k))
    res['rule4'] = rule_4(final, rows, np.random.default_rng(20261022))
    res['post_hoc'] = post_hoc(final, np.random.default_rng(20261023))
    with open(out/'ladder-check.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, list(rows[0])); wr.writeheader()
        wr.writerows({k: (float(f'{v:.10g}') if isinstance(v, float) else v) for k, v in r.items()} for r in rows)
    (out/'ladder-check.json').write_text(json.dumps(res, indent=1, default=str) + '\n')
    print(json.dumps({k: (v if k != 'rule4' else {q: v[q] for q in v if q != 'fixed_pairs'}) for k, v in res.items()}, indent=1, default=str))
    print(json.dumps(res['rule4']['fixed_pairs'], indent=1))
    print(json.dumps(res['post_hoc'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1])
