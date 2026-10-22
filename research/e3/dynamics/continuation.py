#!/usr/bin/env python3
"""e3 dynamics follow-up: intermediate runs continued past 50,000 updates (DECISIONS D15, QUESTION.md Q2).

Each continuation resumed a packet-1 run from its published update-50,000 checkpoint (weights, Adam
moments, update count). Asserted per continuation: its resume file is the source's published
checkpoint_050000.npz (SHA-256 against config and source manifest), its data arrays equal the source's,
and its update-50,000 metrics row equals the source's except wall-clock fields and the optimizer-update
diagnostic, which a resumed row does not log.

Measures are ladder.py's (shared e3 definitions, D6) on the same per-ladder eval pairs, whose label sets
come from the packet-1 ladder runs passed with --ladder. Writes continuation.csv (every published
checkpoint of the continuations and of their sources), continuation_table.csv (selected updates), the
figure continuation.png/svg, the figure continuation_dense.png/svg (own held-out accuracy, train cross-entropy and
weight norm at every logged update, from metrics.jsonl) and packet1_settle.csv (for every --ladder run, the first
update from 20,000 after which own held-out accuracy stays within 0.01 of its update-50,000 value).

uv run --locked --group analysis python research/e3/dynamics/continuation.py CONT_DIR ... --ladder RUN_DIR ... --out DIR
"""
import argparse
import json
from pathlib import Path
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder  # noqa: E402
from ladder import LABELS, SHARE_COLORS, SEED_STYLES, save, write_csv  # noqa: E402

ROOT = ladder.ROOT
KEYS = ['acc', 'P_yS', 'margin_N_sigma', 'P_SN']
TABLE_STEPS = (50000, 100000, 150000)
KILO = FuncFormatter(lambda x, _: f'{x/1000:.0f}k')
WALL = ('seconds', 'elapsed_after_diagnostics', 'optimizer_update')  # the resumed row logs no update (D15)


def logged(run_dir, step):
    for line in (run_dir/'metrics.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['step'] == step:
            return {k: v for k, v in r.items() if k not in WALL}
    raise KeyError(f'{run_dir.name}: no metrics row at {step}')


def source_of(cont, by_id):
    cfg = json.loads((cont['dir']/'config.json').read_text())
    resume = ROOT/cfg['resume']
    src = by_id[resume.parent.name]
    start = int(resume.stem.split('_')[1])
    manifest = json.loads((src['dir']/'manifest.json').read_text())['files']
    assert cfg['resume_sha256'] == ladder.sha256(resume) == manifest[resume.name]['sha256'], cont['run_id']
    assert all((cont['data'][k] == src['data'][k]).all() for k in cont['data']), cont['run_id']
    assert logged(cont['dir'], start) == logged(src['dir'], start), f"{cont['run_id']}: resumed row differs from source"
    return src, start


def figure(rows, conts, ladders, refs, out):
    seeds = list(ladders)
    fig, axes = plt.subplots(len(seeds), len(KEYS), figsize=(3.9*len(KEYS), 2.8*len(seeds)), sharex=True, sharey='col',
                             layout='constrained', squeeze=False)
    for i, seed in enumerate(seeds):
        mine = [c for c in conts if c['data_seed'] == seed]
        inits = sorted({c['init_seed'] for c in conts})
        for c in sorted(mine, key=lambda c: (c['share'], c['init_seed'])):
            rs = sorted((r for r in rows if r['run_id'] in (c['run_id'], c['source']) and r['step'] > 0), key=lambda r: r['step'])
            style = dict(color=SHARE_COLORS[c['share']], lw=1, ls=SEED_STYLES[inits.index(c['init_seed']) % 4])
            for j, key in enumerate(KEYS):
                axes[i, j].plot([r['step'] for r in rs], [r[key] for r in rs], marker='.', ms=3,
                                label=f"{c['share']:.0%} ({c['n_train']}), init {c['init_seed']}", **style)
        for r in refs[seed]:  # the ladder's 50% runs at 50,000 updates, not continued
            for j, key in enumerate(KEYS):
                axes[i, j].scatter([r['step']], [r[key]], color=SHARE_COLORS[.5], marker='os^D'[inits.index(r['init_seed']) % 4],
                                   s=22, zorder=3, label=f"50% ({r['n_train']}), init {r['init_seed']}, 50k only")
        axes[i, 0].set_ylabel(f"data seed {seed}\n{len(ladders[seed]['ev'])} eval pairs", fontsize=8)
        axes[i, 0].legend(fontsize=6, frameon=False, loc='upper left')
    for j, key in enumerate(KEYS):
        axes[0, j].set_title(LABELS[key][0], fontsize=9)
        for ax in axes[:, j]:
            ax.grid(True, color='0.9', lw=.6); ax.axhline(LABELS[key][1], color='0.4', lw=.8, ls=(0, (4, 3)))
            ax.axvline(50000, color='0.6', lw=.8, ls=':')
        axes[-1, j].set_xlabel('update step')
        axes[-1, j].xaxis.set_major_locator(MultipleLocator(25000)); axes[-1, j].xaxis.set_major_formatter(KILO)
    axes[0, 0].set_ylim(-.02, 1.02)
    save(fig, out, 'continuation', 'Published checkpoints of the packet-1 runs (to 50,000) and their uninterrupted '
         'continuations (dotted line: resume point; 45% to 150,000, 40% to 100,000). Blue points: the ladder\'s 50% runs at '
         '50,000 updates, not continued. Color = nominal setting, line style and marker = init seed. Each ladder is scored on '
         'its own eval pairs; pairs within a ladder are not independent. Dashed: chance 1/31, no preference 0.5, zero margin. '
         'Source: continuation.csv.')


def dense(run_dir, data):
    """Own held-out accuracy (cal and audit splits pooled), train cross-entropy and weight norm, every logged update."""
    n_cal, n_audit = len(data['cal']), len(data['audit'])
    rows = []
    for line in (run_dir/'metrics.jsonl').read_text().splitlines():
        r = json.loads(line)
        rows.append(dict(step=r['step'], own_heldout=(r['cal']*n_cal + r['audit']*n_audit)/(n_cal + n_audit),
                         train_ce=r['objective']['train']['cross_entropy'], norm=r['norm']))
    return rows


def settle(rows, end=50000, tol=.01):
    """First update (on the 1,000 grid from 20,000) after which own held-out accuracy stays within tol of its value at end."""
    acc = {r['step']: r['own_heldout'] for r in rows if 20000 <= r['step'] <= end and r['step'] % 1000 == 0}
    return min(s for s in acc if all(abs(acc[t] - acc[end]) <= tol for t in acc if t >= s))


def dense_figure(conts, by_id, out):
    seeds = sorted({c['data_seed'] for c in conts})
    keys = [('own_heldout', 'own held-out accuracy (metrics)'), ('train_ce', 'train cross-entropy'), ('norm', 'weight norm')]
    fig, axes = plt.subplots(len(seeds), len(keys), figsize=(3.9*len(keys), 2.8*len(seeds)), sharex=True, sharey='col',
                             layout='constrained', squeeze=False)
    inits = sorted({c['init_seed'] for c in conts})
    for i, seed in enumerate(seeds):
        for c in sorted((c for c in conts if c['data_seed'] == seed), key=lambda c: (c['share'], c['init_seed'])):
            src = by_id[c['source']]
            rs = [r for r in dense(src['dir'], src['data']) if 20000 <= r['step'] <= c['start']] + \
                 [r for r in dense(c['dir'], c['data']) if r['step'] > c['start']]
            style = dict(color=SHARE_COLORS[c['share']], lw=.8, ls=SEED_STYLES[inits.index(c['init_seed']) % 4])
            for j, (key, _) in enumerate(keys):
                axes[i, j].plot([r['step'] for r in rs], [r[key] for r in rs],
                                label=f"{c['share']:.0%} ({c['n_train']}), init {c['init_seed']}", **style)
        axes[i, 0].set_ylabel(f'data seed {seed}', fontsize=8)
        axes[i, 0].legend(fontsize=6, frameon=False, loc='lower right')
    for j, (key, title) in enumerate(keys):
        axes[0, j].set_title(title, fontsize=9)
        for ax in axes[:, j]:
            ax.grid(True, color='0.9', lw=.6); ax.axvline(50000, color='0.6', lw=.8, ls=':')
            if key == 'train_ce':
                ax.set_yscale('log')
        axes[-1, j].set_xlabel('update step')
        axes[-1, j].xaxis.set_major_locator(MultipleLocator(25000)); axes[-1, j].xaxis.set_major_formatter(KILO)
    save(fig, out, 'continuation_dense', 'Every logged update (100-update spacing) from 20,000, from metrics.jsonl of the '
         'packet-1 runs and their continuations (dotted line: resume point). Own held-out accuracy pools the run\'s own cal and '
         'audit splits, so its pairs differ from the ladder eval pairs and differ between shares. Color = nominal setting, line '
         'style = init seed.')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('runs', type=Path, nargs='+', help='continuation run directories')
    ap.add_argument('--ladder', type=Path, nargs='+', required=True, help='packet-1 ladder run directories')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    base = [ladder.load(d) for d in args.ladder]
    by_id = {r['run_id']: r for r in base}
    seeds = sorted({r['data_seed'] for r in base})
    ladders = {s: ladder.ladder_sets([r for r in base if r['data_seed'] == s]) for s in seeds}
    conts = [ladder.load(d) for d in args.runs]
    rows = []
    for c in conts:
        src, start = source_of(c, by_id)
        c.update(source=src['run_id'], n_train=len(c['train']), start=start)
        rows += [dict(r, source=src['run_id']) for r in ladder.score(c, ladders[c['data_seed']])]
    for sid in sorted({c['source'] for c in conts}):
        rows += [dict(r, source='') for r in ladder.score(by_id[sid], ladders[by_id[sid]['data_seed']])]
    refs = {s: [r for b in base if b['data_seed'] == s and b['share'] == .5
                for r in ladder.score(b, ladders[s]) if r['step'] == 50000] for s in seeds}
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out/'continuation.csv', rows)
    table = []
    for c in sorted(conts, key=lambda c: (c['share'], c['data_seed'], c['init_seed'])):
        for step in TABLE_STEPS:
            hit = [r for r in rows if r['run_id'] in (c['run_id'], c['source']) and r['step'] == step]
            if hit:
                table.append({k: hit[0][k] for k in ('data_seed', 'share', 'n_train', 'init_seed', 'step', 'acc', 'ce',
                                                     'P_yN', 'P_yS', 'P_SN', 'margin_N_sigma', 'rank_absent')})
    write_csv(args.out/'continuation_table.csv', table)
    figure(rows, conts, {s: ladders[s] for s in sorted({c['data_seed'] for c in conts})}, refs, args.out)
    dense_figure(conts, by_id, args.out)
    settled = []
    for b in sorted(base, key=lambda b: (b['share'], b['data_seed'], b['init_seed'])):
        d = {r['step']: r for r in dense(b['dir'], b['data'])}
        settled.append(dict(run_id=b['run_id'], data_seed=b['data_seed'], share=b['share'], init_seed=b['init_seed'],
                            **{f'own_heldout_{s//1000}k': d[s]['own_heldout'] for s in (30000, 40000, 50000)},
                            settled_within_001=settle(list(d.values()))))
    write_csv(args.out/'packet1_settle.csv', settled)
    for t in settled:
        print(t['data_seed'], t['share'], t['init_seed'], ' '.join(f"{t[k]:.3f}" for k in t if k.startswith('own_')),
              t['settled_within_001'])
    for t in table:
        print(t['data_seed'], f"{t['share']:.2f}", t['init_seed'], t['step'],
              ' '.join(f'{k}={t[k]:.3f}' for k in ('acc', 'P_yS', 'P_SN', 'margin_N_sigma', 'rank_absent')))


if __name__ == '__main__':
    main()
