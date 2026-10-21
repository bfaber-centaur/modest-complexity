#!/usr/bin/env python3
"""Audit of the mechanism lane's first deliverable with independently written code.

Recomputes the frequency spectrum, every retention/removal row, a sample of the
random-subset rows, the targeted rescaling, every random rescaling draw (the
lane's documented gain procedure: c = exp(sigma z), z from default_rng(33200),
sigmas 0.1, 0.25, 0.5, 1 with 16 draws each) and the codec, using
research/audit/independent.py only. Also measures per-unit input/output balance.

uv run --locked python research/audit/audit_mechanisms.py RUN_DIR LANE_FIG_DIR OUT.json
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import independent as ind  # noqa: E402

SPLITS = ('train', 'cal', 'audit')
METRICS = ('accuracy', 'cross_entropy', 'margin_min', 'margin_median', 'margin_p05')


def load(run_dir, step):
    with np.load(run_dir/f'checkpoint_{step:06d}.npz') as z:
        return [z['U'].copy(), z['V'].copy(), z['W'].copy()]


def metrics(params, Xa, Xb, d):
    logits = ind.forward(params, Xa, Xb)[0]
    out = {}
    for s in SPLITS:
        ev = ind.evaluate(logits[d[s]], d['y'][d[s]])
        out.update({f'{s}_accuracy': ev['acc'], **{f'{s}_{k}': ev[k] for k in METRICS[1:]}})
    return out, logits


def diff(mine, row):
    acc = max(abs(mine[f'{s}_accuracy'] - float(row[f'{s}_accuracy'])) for s in SPLITS)
    rest = max(abs(mine[f'{s}_{k}'] - float(row[f'{s}_{k}'])) / max(1., abs(float(row[f'{s}_{k}'])))
               for s in SPLITS for k in METRICS[1:])
    return acc, rest


def ranking(params):
    return list(np.argsort(-ind.input_energy(params), kind='stable') + 1)


def main(run_dir, lane, out):
    summary = json.loads((lane/'summary.json').read_text())
    step = summary['input']['step']
    manifest = json.loads((run_dir/'manifest.json').read_text())
    ck = run_dir/f'checkpoint_{step:06d}.npz'
    sha = hashlib.sha256(ck.read_bytes()).hexdigest()
    with np.load(run_dir/'data.npz') as z:
        d = {k: z[k] for k in ('a', 'b', 'y', 'train', 'cal', 'audit')}
    p = 31
    Xa, Xb = ind.one_hot_inputs(d['a'], d['b'], p)
    w = load(run_dir, step); w0 = load(run_dir, 0)
    rep = dict(checkpoint_sha_matches_manifest_and_summary=sha == manifest['files'][ck.name]['sha256'] == summary['input']['checkpoint_sha256'])

    # Spectrum and ranking.
    e = ind.input_energy(w); share = e / e.sum(); e0 = ind.input_energy(w0); share0 = e0 / e0.sum()
    order = ranking(w)
    rep['spectrum'] = dict(ranking_matches=[int(k) for k in order] == summary['ranking'],
                           max_share_diff=float(np.max(np.abs(share - summary['energy_share']))),
                           max_step0_share_diff=float(np.max(np.abs(share0 - summary['reference_energy_share']['share']))),
                           share_range=(float(share.min()), float(share.max())), top4=[int(k) for k in order[:4]],
                           top4_share=float(share[np.array(order[:4]) - 1].sum()),
                           step0_share_range=(float(share0.min()), float(share0.max())),
                           step0_top4=[int(k) for k in ranking(w0)[:4]])

    # Every retention/removal row.
    rows = list(csv.DictReader(open(lane/'retention.csv')))
    worst_acc = worst_rest = 0.; prefix_ok = True; curve = {}
    for r in rows:
        freqs = [int(f) for f in r['frequencies'].split()]
        prefix_ok &= freqs == [int(k) for k in order[:int(r['K'])]]
        params = ind.keep_frequencies(w, freqs) if r['mode'] == 'retain' else ind.drop_frequencies(w, freqs)
        mine, _ = metrics(params, Xa, Xb, d)
        a, b = diff(mine, r); worst_acc = max(worst_acc, a); worst_rest = max(worst_rest, b)
        curve[(r['mode'], int(r['K']))] = mine
    orig, base_logits = metrics(w, Xa, Xb, d)
    rep['retention'] = dict(
        rows=len(rows), frequencies_are_topK_prefixes=bool(prefix_ok), max_acc_diff=worst_acc, max_rel_other_diff=worst_rest,
        full_retain_equals_original=max(abs(curve[('retain', 15)][k] - orig[k]) for k in orig),
        retain_train_acc_ce={K: (curve[('retain', K)]['train_accuracy'], curve[('retain', K)]['train_cross_entropy']) for K in (4, 8, 12, 15)},
        first_K_retain_train_acc_1=min(K for K in range(16) if curve[('retain', K)]['train_accuracy'] == 1.),
        remove_train_acc_ce={K: (curve[('remove', K)]['train_accuracy'], curve[('remove', K)]['train_cross_entropy']) for K in (2, 4, 8)},
        max_audit_acc_any_K=max(c['audit_accuracy'] for c in curve.values()),
        audit_ce_retain=[curve[('retain', K)]['audit_cross_entropy'] for K in range(16)],
        audit_margin_median_max=max(c['audit_margin_median'] for c in curve.values()))
    rep['retention']['audit_ce_retain_monotone'] = bool(np.all(np.diff(rep['retention']['audit_ce_retain']) > 0))

    # Random subsets: every 4th row recomputed, all rows' energy fractions.
    rand = list(csv.DictReader(open(lane/'random_subsets.csv')))
    wa = wr = we = 0.
    for i, r in enumerate(rand):
        freqs = [int(f) for f in r['frequencies'].split()]
        we = max(we, abs(share[np.array(freqs) - 1].sum() - float(r['energy_fraction'])))
        if i % 4 == 0:
            params = ind.keep_frequencies(w, freqs) if r['mode'] == 'retain' else ind.drop_frequencies(w, freqs)
            a, b = diff(metrics(params, Xa, Xb, d)[0], r); wa = max(wa, a); wr = max(wr, b)
    r4 = [float(r['train_accuracy']) for r in rand if r['mode'] == 'retain' and r['K'] == '4']
    m8 = [float(r['train_accuracy']) for r in rand if r['mode'] == 'remove' and r['K'] == '8']
    e4 = [float(r['energy_fraction']) for r in rand if r['mode'] == 'retain' and r['K'] == '4']
    rep['random_subsets'] = dict(
        rows=len(rand), recomputed=len(range(0, len(rand), 4)), max_acc_diff=wa, max_rel_other_diff=wr, max_energy_fraction_diff=we,
        retain4_random_range=(min(r4), max(r4)), retain4_topK_beats=sum(curve[('retain', 4)]['train_accuracy'] > x for x in r4),
        remove8_random_range=(min(m8), max(m8)), remove8_topK_below=sum(curve[('remove', 8)]['train_accuracy'] < x for x in m8),
        remove8_topK_ties=sum(curve[('remove', 8)]['train_accuracy'] == x for x in m8),
        energy4_random_range=(min(e4), max(e4)), max_random_audit_acc=max(float(r['audit_accuracy']) for r in rand))

    # Codec at the checkpoint, per-unit balance.
    split2 = [d['train'], d['cal']]
    rep['codec_checkpoint'] = ind.codec_bits(w, Xa, Xb, split2)
    A = (w[0]**2).sum(0) + (w[1]**2).sum(0); B = (w[2]**2).sum(1)
    cstar = (B / A)**.25  # per-unit gain minimizing |c u|^2 + |c v|^2 + |w/c|^2
    bal = [w[0]*cstar, w[1]*cstar, w[2]/cstar[:, None]]
    norm = lambda P: float(np.sqrt(sum((x**2).sum() for x in P)))
    rep['balance'] = dict(ratio_in_over_out_quantiles=[float(x) for x in np.quantile(A / B, [0, .1, .5, .9, 1])],
                          norm_checkpoint=norm(w), norm_balanced=norm(bal),
                          balanced_logit_change=float(np.abs(ind.forward(bal, Xa, Xb)[0] - base_logits).max()),
                          balanced_codec=ind.codec_bits(bal, Xa, Xb, split2),
                          balanced_top4=[int(k) for k in ranking(bal)[:4]])

    # Targeted rescaling.
    t = summary['targeted_rescaling']
    per_unit = sum(np.abs(ind.token_fourier(X)[1:])**2 for X in w[:2])  # (15, h)
    units = per_unit.argmax(0) + 1 == t['target']
    res = {}
    for name, g in (('gain', t['gain']), ('previous_grid_gain', t['gain'] / (1000/1.01)**(1/694))):
        c = np.where(units, g, 1.)
        wt = [w[0]*c, w[1]*c, w[2]/c[:, None]]
        rk = ranking(wt)
        res[name] = dict(gain=g, target_in_top4=t['target'] in rk[:4], top4=sorted(int(k) for k in rk[:4]),
                         max_abs_logit_change=float(np.abs(ind.forward(wt, Xa, Xb)[0] - base_logits).max()),
                         codec=ind.codec_bits(wt, Xa, Xb, split2), norm=norm(wt),
                         retained_top4_train=(lambda m: (m['train_accuracy'], m['train_cross_entropy']))(
                             metrics(ind.keep_frequencies(wt, rk[:4]), Xa, Xb, d)[0]))
    rep['targeted'] = dict(units=int(units.sum()), max_abs_logit=float(np.abs(base_logits).max()), **res)

    # Every random rescaling draw, regenerated from the lane's documented procedure.
    sweep = list(csv.DictReader(open(lane/'rescaling_sweep.csv')))
    rng = np.random.default_rng(33200)
    checks = dict(ranking=0, codec=0, norm=0.); summ = {}
    for r in sweep:
        c = np.exp(float(r['sigma']) * rng.standard_normal(w[0].shape[1]))
        wt = [w[0]*c, w[1]*c, w[2]/c[:, None]]
        rk = ranking(wt); cb = ind.codec_bits(wt, Xa, Xb, split2)
        rel = float(np.abs(ind.forward(wt, Xa, Xb)[0] - base_logits).max() / np.abs(base_logits).max())
        checks['ranking'] += ' '.join(map(str, rk)) != r['ranking']
        checks['codec'] += cb['bits'] != int(r['codec_bits'])
        checks['norm'] = max(checks['norm'], abs(norm(wt) - float(r['param_norm'])) / float(r['param_norm']))
        s = summ.setdefault(r['sigma'], dict(top4_changed=0, codec_bits=[], max_rel_logit=0., norm_range=[np.inf, 0], codec_shorter=0))
        s['top4_changed'] += set(rk[:4]) != set(order[:4])
        s['codec_bits'].append(cb['bits']); s['codec_shorter'] += cb['bits'] < rep['codec_checkpoint']['bits']
        s['max_rel_logit'] = max(s['max_rel_logit'], rel)
        s['norm_range'] = [min(s['norm_range'][0], norm(wt)), max(s['norm_range'][1], norm(wt))]
    for s in summ.values():
        s['codec_bits'] = {str(b): s['codec_bits'].count(b) for b in sorted(set(s['codec_bits']))}
    rep['rescaling_sweep'] = dict(draws=len(sweep), ranking_mismatches=checks['ranking'], codec_mismatches=checks['codec'],
                                  max_norm_rel_diff=checks['norm'], by_sigma=summ)
    out.write_text(json.dumps(rep, indent=1, default=float) + '\n')
    print(json.dumps(rep, indent=1, default=float))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
