#!/usr/bin/env python3
"""Pairs never trained on by either the 30% (e1) or the 50% (e1f2) split, a check
that the training sets nest, and both conditions scored on that one set.

uv run --locked --group analysis python research/e2/coordinator/common_set.py
"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from experiment import array_sha256, forward, objective_metrics  # noqa: E402

RUNS = {'30%': ['research/coordinator/runs/e1-co-add-w128-i30000-a1', 'research/size-time/runs/e1-st-add-w128-i30001-a1',
                'research/size-time/runs/e1-st-add-w128-i30002-a1'],
        '50%': [f'research/coordinator/runs/e1f2-co-add-w128-i{i}-tf50-a1' for i in (30000, 30001, 30002)]}


def main():
    s30, s50 = (np.load(ROOT/f'research/data/{n}') for n in ('e1-split.npz', 'e1f2-split-tf50.npz'))
    for k in ('a', 'b', 'yadd'):
        assert np.array_equal(s30[k], s50[k]), k
    a, b, y = s30['a'], s30['b'], s30['yadd']
    p = int(y.max()) + 1
    t30, t50 = set(s30['train'].tolist()), set(s50['train'].tolist())
    common = np.array(sorted(set(range(p*p)) - t30 - t50))
    groups = {(min(i, j), max(i, j)) for i, j in zip(a[common], b[common])}
    report = dict(
        train_nested=t30 <= t50, train30=len(t30), train50=len(t50), added_by_50=len(t50 - t30),
        common_never_trained=len(common), common_unordered_groups=len(groups),
        common_transpose_closed=set((a[common]*p + b[common]).tolist()) == set((b[common]*p + a[common]).tolist()),
        common_class_counts=np.bincount(y[common], minlength=p).tolist(),
        common_equals_50pct_cal_plus_audit=set(common.tolist()) == set(s50['cal'].tolist()) | set(s50['audit'].tolist()),
        common_inside_30pct_audit=len(set(common.tolist()) & set(s30['audit'].tolist())),
        common_inside_30pct_cal=len(set(common.tolist()) & set(s30['cal'].tolist())),
        common_sha256=array_sha256(common))
    np.savez_compressed(ROOT/'research/e2/data/common-never-trained-30-50.npz', index=common)
    rows = []
    for share, dirs in RUNS.items():
        for d in map(lambda x: ROOT/x, dirs):
            data = np.load(d/'data.npz')
            assert np.array_equal(data['yadd'], y) and set(data['train'].tolist()) == (t30 if share == '30%' else t50)
            logged = {r['step']: r for r in map(json.loads, (d/'metrics.jsonl').read_text().splitlines())}
            for path in sorted(d.glob('checkpoint_*.npz')):
                ck = np.load(path); step = int(ck['step'])
                logits = forward([ck['U'], ck['V'], ck['W']], a, b)[0]
                obj = objective_metrics(logits, y, common)
                rows.append(dict(share=share, run_id=d.name, step=step,
                                 common_acc=float(np.mean(logits[common].argmax(1) == y[common])),
                                 common_ce=obj['cross_entropy'], common_margin_median=obj['margin_median'],
                                 own_audit_acc=logged[step]['audit'], own_cal_acc=logged[step]['cal'],
                                 own_audit_n=len(data['audit'])))
    with (ROOT/'research/e2/coordinator/common_set_scores.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (ROOT/'research/e2/coordinator/common_set.json').write_text(json.dumps(report, indent=1) + '\n')
    plot(rows, report)
    print(json.dumps(report))
    for r in rows:
        if r['step'] in (0, 2000, 5000, 10000, 20000, 50000):
            print(r['share'], r['run_id'][:30], r['step'], round(r['common_acc'], 4), round(r['own_audit_acc'], 4),
                  round(r['common_ce'], 3))


def plot(rows, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
    colors = {'30%': 'tab:gray', '50%': 'tab:blue'}
    for run in dict.fromkeys(r['run_id'] for r in rows):
        rs = [r for r in rows if r['run_id'] == run]
        share, init = rs[0]['share'], run.split('-i')[1][:5]
        # A run with only two published checkpoints gets markers only, not a misleading straight join.
        style = dict(color=colors[share], marker='os^'[int(init) - 30000], ms=4, ls='-' if len(rs) > 3 else 'none')
        steps = [r['step'] for r in rs]
        axes[0].plot(steps, [r['common_acc'] for r in rs], label=f'{share} train, init {init}', **style)
        axes[0].plot(steps, [r['own_audit_acc'] for r in rs], ':', color=colors[share], lw=1)
        axes[1].plot(steps, [r['common_ce'] for r in rs], **style)
    axes[0].axhline(1/31, color='k', lw=.8, ls='--', label='chance 1/31')
    axes[0].set(xlabel='update', ylabel='accuracy', ylim=(-.02, 1),
                title=f"Common never-trained set ({report['common_never_trained']} pairs)\nsolid: common set; dotted: own audit split")
    axes[1].set(xlabel='update', ylabel='cross-entropy (common set)', title='Cross-entropy on the common set')
    axes[0].legend(fontsize=7, loc='center right')
    fig.text(0, -.04, 'Width 128; published checkpoints only (markers). Common set = all pairs minus both training sets; '
             f"sha256 {report['common_sha256'][:12]}. Source: research/e2/coordinator/common_set_scores.csv", fontsize=7)
    out = ROOT/'research/e2/coordinator/figures'
    out.mkdir(exist_ok=True)
    for ext in ('png', 'svg'):
        fig.savefig(out/f'common-set.{ext}', dpi=150, bbox_inches='tight')


if __name__ == '__main__':
    main()
