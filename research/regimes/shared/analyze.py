#!/usr/bin/env python3
"""R2 publishing, tables and figures.

publish: copy small run files from the local root into runs/<ID>/ (metrics
  gzipped, checkpoints only for the IDs' saved states) and write MANIFEST.sha256.
figures: read only the published files and write figures/*.png|svg|csv.

uv run --locked python research/regimes/shared/analyze.py publish --root runs/regimes --ids ...
uv run --locked python research/regimes/shared/analyze.py figures
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
RUNS, FIGS = HERE/'runs', HERE/'figures'
ARMS = ['fresh', 'rel', 'part', 'unrel', 'same']
JOINT = ['joint-rel', 'joint-unrel', 'alone-y1']
FOLLOWUP = ['split']  # the one adaptive follow-up (QUESTION.md), drawn apart from the initial comparison
COLORS = {'split': '#882255', 'fresh': '#555555', 'rel': '#0072B2', 'part': '#E69F00', 'unrel': '#009E73', 'same': '#CC79A7',
          'joint-rel': '#0072B2', 'joint-unrel': '#009E73', 'alone-y1': '#D55E00'}
LABELS = {'fresh': 'fresh init', 'rel': 'rel: y1 = x1^x2^x3', 'part': 'part: x2^x3^x4 (no z)',
          'unrel': 'unrel: x5^x6^x7', 'same': 'same: y2 (positive control)',
          'split': 'follow-up split: (x1^x3^x5, x2^x6^x7), 4 classes'}
REP_STYLE = ['-', '--', ':']
UPDATE_NS = (32, 64)  # the two n of the against-updates panels, fixed in QUESTION.md's spirit before the main runs


def parse_id(run_id):
    # r1-share-<arm>-s<init>-a<attempt>
    body, attempt = run_id[len('r1-share-'):].rsplit('-a', 1)
    arm, _, seed = body.rpartition('-s')
    return (arm, int(seed), int(attempt)) if arm else (body, None, int(attempt))


def publish(args):
    lines = []
    for run_id in args.ids:
        src, dst = args.root/run_id, RUNS/run_id
        dst.mkdir(parents=True, exist_ok=True)
        for name in ('config.json', 'summary.json', 'runtime.json'):
            if (src/name).exists():
                shutil.copy2(src/name, dst/name)
        if (src/'metrics.jsonl').exists():
            with open(src/'metrics.jsonl', 'rb') as f, gzip.GzipFile(dst/'metrics.jsonl.gz', 'wb', mtime=0) as g:
                g.write(f.read())
        for ck in sorted(src.glob('*.npz')):
            if ck.name in args.checkpoints:
                shutil.copy2(ck, dst/ck.name)
        # Hashes of every published file, and of every produced file that stays local.
        for f in sorted(dst.iterdir()):
            lines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {run_id}/{f.name}')
        for f in sorted(src.iterdir()):
            if not (dst/f.name).exists():
                lines.append(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {run_id}/{f.name}  (local only)')
    manifest = RUNS/'MANIFEST.sha256'
    old = [l for l in (manifest.read_text().splitlines() if manifest.exists() else [])
           if l.split()[1].split('/')[0] not in args.ids]
    manifest.write_text('\n'.join(sorted(old + lines))+'\n')


def load_run(run_id):
    d = RUNS/run_id
    rows = [json.loads(l) for l in gzip.open(d/'metrics.jsonl.gz', 'rt')]
    cfg = json.loads((d/'config.json').read_text())
    summ = json.loads((d/'summary.json').read_text()) if (d/'summary.json').exists() else None
    return cfg, summ, rows


def registry_status():
    status = {}
    for l in (HERE/'registry.jsonl').read_text().splitlines():
        e = json.loads(l)
        status[e['run_id']] = e['status']
    return status


def collect():
    """Final-step table and trajectories for every planned main run (status kept for missing ones)."""
    plan = [r['run_id'] for r in json.loads((HERE/'plan.json').read_text())['runs']
            if parse_id(r['run_id'])[0] in ARMS + JOINT + FOLLOWUP and parse_id(r['run_id'])[1] in (52200, 52201, 52202)]
    status = registry_status(); final, traj = [], []
    for run_id in plan:
        arm, seed, attempt = parse_id(run_id)
        st = status.get(run_id, 'not run')
        if st != 'complete' or not (RUNS/run_id).exists():
            final.append(dict(run_id=run_id, arm=arm, init_seed=seed, status=st)); continue
        cfg, summ, rows = load_run(run_id)
        for r in rows:
            traj.append(dict(run_id=run_id, arm=arm, init_seed=seed, data_seed=cfg['data_seed'], **r))
        last = {}
        for r in rows:
            if r['tag'] in ('train', 'readout'):
                last[(r['tag'], r['n'])] = r
        for (tag, n), r in sorted(last.items()):
            final.append(dict(run_id=run_id, arm=arm, init_seed=seed, data_seed=cfg['data_seed'], status=st,
                              branch=tag, coverage_y2=cfg['coverage'][str(n)]['y2'], **r))
    return final, traj


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, 'w') as f:
        f.write(','.join(keys)+'\n')
        for r in rows:
            f.write(','.join('' if r.get(k) is None else (f'{r[k]:.6g}' if isinstance(r[k], float) else str(r[k]))
                             for k in keys)+'\n')


def figures(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    final, traj = collect()
    write_csv(FIGS/'final.csv', final)
    missing = sorted({(r['run_id'], r['status']) for r in final if r['status'] != 'complete'})
    seeds = [52200, 52201, 52202]

    def per_n(arm, branch, key, seed):
        rows = sorted((r for r in final if r['arm'] == arm and r.get('branch') == branch and r['init_seed'] == seed),
                      key=lambda r: r['n'])
        return [r['n'] for r in rows], [r.get(key) for r in rows]

    def per_step(arm, branch, key, seed, n):
        rows = sorted((r for r in traj if r['arm'] == arm and r['tag'] == branch and r['init_seed'] == seed
                       and r['n'] == n), key=lambda r: r['step'])
        return [r['step'] for r in rows], [r.get(key) for r in rows]

    # Figure 1: transfer curves (fine-tune row, frozen-readout row).
    fig, axes = plt.subplots(2, 4, figsize=(17, 8.2), sharey=False)
    sub = []
    for row, branch in enumerate(('train', 'readout')):
        ax = axes[row, 0]
        for arm in ARMS:
            for i, s in enumerate(seeds):
                n, y = per_n(arm, branch, 'new_test_acc_x4is1', s)
                if n:
                    ax.plot(np.array(n)*(1+0.04*(i-1)), y, REP_STYLE[i], marker='o', ms=3, lw=1.1,
                            color=COLORS[arm], label=LABELS[arm] if i == 0 else None)
        ax.axhline(.5, color='k', lw=.8, ls='--'); ax.text(8, .41, 'chance and z-blind heuristics = 0.5', fontsize=7)
        ax.set_ylabel('held-out accuracy, x4 = 1 half (y2 = z)')
        ax = axes[row, 1]
        for arm in ARMS:
            for i, s in enumerate(seeds):
                n, y = per_n(arm, branch, 'new_test_acc', s)
                n2, yt = per_n(arm, branch, 'new_train_acc', s)
                if n:
                    ax.plot(np.array(n)*(1+0.04*(i-1)), y, REP_STYLE[i], marker='o', ms=3, lw=1.1, color=COLORS[arm])
                    ax.plot(np.array(n2)*(1+0.04*(i-1)), yt, 'x', ms=4, color=COLORS[arm], alpha=.6)
        ax.axhline(.75, color='k', lw=.8, ls='--'); ax.text(90, .725, 'majority (always 0) = 0.75', fontsize=7)
        ax.set_ylabel('accuracy, all held-out (lines) / train (x)')
        for ax in axes[row, :2]:
            ax.set_xscale('log', base=2); ax.set_xlabel('training inputs n (D_n, nested)'); ax.set_ylim(.4, 1.02)
        for col, n in enumerate(UPDATE_NS, 2):
            ax = axes[row, col]
            for arm in ARMS:
                for i, s in enumerate(seeds):
                    st, y = per_step(arm, branch, 'new_test_acc_x4is1', s, n)
                    st2, yt = per_step(arm, branch, 'new_train_acc', s, n)
                    if st:
                        ax.plot(st, y, REP_STYLE[i], lw=1.1, color=COLORS[arm])
                        ax.plot(st2, yt, REP_STYLE[i], lw=.7, color=COLORS[arm], alpha=.35)
            ax.axhline(.5, color='k', lw=.8, ls='--')
            ax.set_xscale('symlog', linthresh=1); ax.set_xlim(0, 2200); ax.set_ylim(.2, 1.02)
            ax.set_xlabel('updates in phase 2 (symlog; 0 = phase start)')
            ax.set_ylabel('held-out acc, x4 = 1 half (bold); train acc (faint)')
            ax.set_title(f'n = {n}', fontsize=9)
        axes[row, 0].set_title(('fine-tune trunk + new head' if branch == 'train' else
                                'frozen features, new head only') + ': final step (2,000 updates)', fontsize=9, loc='left')
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='lower center', ncol=5, fontsize=8, frameon=False)
    reps = 'replicates: solid r0 (init 52200, data 52100), dashed r1 (52201/52101), dotted r2 (52202/52102)'
    miss = '; '.join(f'{i}: {s}' for i, s in missing if parse_id(i)[0] in ARMS) or 'none'
    fig.suptitle('R2: acquisition of y2 = (x1^x2) AND x4 after related, partly related, unrelated, same-task or no pretraining\n'
                 f'{reps}. Missing/capped/failed transfer runs: {miss}', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .95))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'transfer.{ext}', dpi=130)
    plt.close(fig)
    write_csv(FIGS/'transfer_trajectories.csv',
              [{k: r.get(k) for k in ('run_id', 'arm', 'init_seed', 'data_seed', 'tag', 'n', 'step', 'new_train_acc',
                                      'new_test_acc', 'new_test_acc_x4is1', 'new_test_acc_x4is0', 'new_train_loss',
                                      'new_test_loss', 'pre_test_acc')}
               for r in traj if r['tag'] in ('train', 'readout') and r['arm'] in ARMS + JOINT + FOLLOWUP])

    # Figure 2: joint versus separate training.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for arm, color, label in (('fresh', COLORS['fresh'], 'separate: y2 alone'),
                              ('joint-rel', COLORS['rel'], 'joint: y1 + y2'),
                              ('joint-unrel', COLORS['unrel'], 'joint: x5^x6^x7 + y2')):
        for i, s in enumerate(seeds):
            n, y = per_n(arm, 'train', 'new_test_acc_x4is1', s)
            if n:
                axes[0].plot(np.array(n)*(1+.04*(i-1)), y, REP_STYLE[i], marker='o', ms=3, color=color,
                             label=label if i == 0 else None)
            st, y = per_step(arm, 'train', 'new_test_acc_x4is1', s, UPDATE_NS[0])
            if st:
                axes[2].plot(st, y, REP_STYLE[i], color=color)
    for arm, key, color, label in (('alone-y1', 'new_test_acc', COLORS['alone-y1'], 'separate: y1 alone'),
                                   ('joint-rel', 'pre_test_acc', COLORS['rel'], 'joint: y1 head of y1 + y2'),
                                   ('joint-unrel', 'pre_test_acc', COLORS['unrel'], 'joint: x5^x6^x7 head')):
        for i, s in enumerate(seeds):
            n, y = per_n(arm, 'train', key, s)
            if n:
                axes[1].plot(np.array(n)*(1+.04*(i-1)), y, REP_STYLE[i], marker='o', ms=3, color=color,
                             label=label if i == 0 else None)
    for ax in axes[:2]:
        ax.set_xscale('log', base=2); ax.set_xlabel('training inputs n (both tasks labeled on the same D_n)')
        ax.axhline(.5, color='k', lw=.8, ls='--'); ax.set_ylim(.4, 1.02); ax.legend(fontsize=7, loc='lower right')
    axes[0].set_ylabel('y2 held-out accuracy, x4 = 1 half'); axes[0].set_title('new task y2', fontsize=9)
    axes[1].set_ylabel('auxiliary task held-out accuracy'); axes[1].set_title('auxiliary task (balanced)', fontsize=9)
    axes[2].set_xscale('symlog', linthresh=1); axes[2].set_xlim(0, 2200); axes[2].set_ylim(.2, 1.02)
    axes[2].axhline(.5, color='k', lw=.8, ls='--'); axes[2].set_xlabel('updates (symlog)')
    axes[2].set_ylabel('y2 held-out accuracy, x4 = 1 half'); axes[2].set_title(f'n = {UPDATE_NS[0]}', fontsize=9)
    miss = '; '.join(f'{i}: {s}' for i, s in missing if parse_id(i)[0] in JOINT) or 'none'
    fig.suptitle(f'R2 joint vs separate training at equal width and equal per-task exposure (n inputs x 2,000 updates). '
                 f'{reps}.\nMissing/capped/failed: {miss}', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .92))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'joint.{ext}', dpi=130)
    plt.close(fig)

    # Follow-up figure: the split control next to the initial arms it is compared with (faded).
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    for ax, branch, title in ((axes[0], 'train', 'fine-tune trunk + new head'),
                              (axes[1], 'readout', 'frozen phase-1 features, new head only')):
        for arm in ['fresh', 'rel', 'part'] + FOLLOWUP:
            for i, s in enumerate(seeds):
                n, y = per_n(arm, branch, 'new_test_acc_x4is1', s)
                if n:
                    ax.plot(np.array(n)*(1+.04*(i-1)), y, REP_STYLE[i], marker='o', ms=3 if arm in FOLLOWUP else 2,
                            lw=1.6 if arm in FOLLOWUP else .9, alpha=1 if arm in FOLLOWUP else .45, color=COLORS[arm],
                            label=(LABELS[arm] + ('' if arm in FOLLOWUP else ' (initial comparison)')) if i == 0 else None)
        ax.axhline(.5, color='k', lw=.8, ls='--'); ax.set_xscale('log', base=2); ax.set_ylim(.4, 1.02)
        ax.set_xlabel('training inputs n (D_n, nested)'); ax.set_ylabel('y2 held-out accuracy, x4 = 1 half')
        ax.set_title(title + ': final step', fontsize=9)
    axes[0].legend(fontsize=6.5, loc='lower right')
    mech = FIGS/'mechanism.csv'
    if mech.exists():
        import csv
        rows = [r for r in csv.DictReader(open(mech)) if r['kind'] == 'probe' and r['layer'] == '2']
        feats = ['x1^x2', 'x1^x3', 'x2^x3', 'x1^x4', 'x5^x6']
        groups = [('init', 'fresh', 'init', '#999999'), ('rel', 'rel', 'pretrained', COLORS['rel']),
                  ('part', 'part', 'pretrained', COLORS['part']), ('unrel', 'unrel', 'pretrained', COLORS['unrel']),
                  ('split (follow-up)', 'split', 'pretrained', COLORS['split'])]
        ax = axes[2]
        for g, (label, arm, net, color) in enumerate(groups):
            for i, s in enumerate(seeds):
                v = {r['feature']: float(r['value']) for r in rows if r['arm'] == arm and r['net'] == net
                     and int(r['init_seed']) == s}
                if v:
                    ax.plot(np.arange(len(feats)) + (g-2)*.14, [v[f] for f in feats], ls='none', marker='os^'[i],
                            mfc='none', color=color, label=label if i == 0 else None)
        ax.set_xticks(range(len(feats))); ax.set_xticklabels(['z = x1^x2', 'x1^x3', 'x2^x3', 'x1^x4', 'x5^x6'])
        ax.axhline(.5, color='k', lw=.8, ls='--'); ax.set_ylim(.4, 1.02); ax.legend(fontsize=6.5, loc='lower left')
        ax.set_ylabel('ridge-probe test accuracy, hidden layer 2'); ax.set_title(
            'linear decodability after phase 1 (init = untrained network)', fontsize=9)
    fig.suptitle('R2 follow-up (adaptive, after M2): pretraining that shares the pair {x1, x2} with y2 but has no x1^x2. '
                 'Markers/linestyles o/solid, square/dashed, triangle/dotted = replicates r0, r1, r2.', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'followup.{ext}', dpi=130)
    plt.close(fig)

    # Figure 3: input dependence (flip sensitivity) of y2 predictions and of the pretraining head.
    flip_rows = []
    for arm in ARMS + ['joint-rel', 'joint-unrel']:
        for s in seeds:
            run_id = f'r1-share-{arm}-s{s}-a1'
            if not (RUNS/run_id/'summary.json').exists():
                continue
            summ = json.loads((RUNS/run_id/'summary.json').read_text())
            if 'pretrained' in summ:
                flip_rows.append(dict(run_id=run_id, arm=arm, init_seed=s, when='after phase 1', head='pre', n='',
                                      **{f'x{k+1}': v for k, v in enumerate(summ['pretrained']['pre_flip'])}))
            for n, row in summ['per_n'].items():
                flip_rows.append(dict(run_id=run_id, arm=arm, init_seed=s, when='after phase 2', head='new', n=n,
                                      **{f'x{k+1}': v for k, v in enumerate(row['new_flip'])}))
                if 'pre_flip' in row:
                    flip_rows.append(dict(run_id=run_id, arm=arm, init_seed=s, when='after phase 2', head='pre', n=n,
                                          **{f'x{k+1}': v for k, v in enumerate(row['pre_flip'])}))
    write_csv(FIGS/'flip_sensitivity.csv', flip_rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
    bits = np.arange(1, 11)
    for ax, n in zip(axes, ('16', '64', '512')):
        for j, arm in enumerate(ARMS + ['joint-rel']):
            for i, s in enumerate(seeds):
                r = [x for x in flip_rows if x['arm'] == arm and x['init_seed'] == s and x['head'] == 'new' and x['n'] == n]
                if r:
                    ax.plot(bits + (j-2.5)*.1, [r[0][f'x{b}'] for b in bits], ls='none', marker='os^'[i], ms=3.5, mfc='none',
                            color=COLORS[arm] if arm != 'joint-rel' else '#56B4E9',
                            label=(LABELS.get(arm, 'joint-rel: y1 + y2 on D_n') if i == 0 else None))
        ax.plot(bits, [.5, .5, 0, .5, 0, 0, 0, 0, 0, 0], 'k_', ms=14, mew=2, label='exact y2')
        ax.set_xticks(bits); ax.set_xlabel('flipped input bit'); ax.set_title(f'y2 head after phase 2, n = {n}', fontsize=9)
    axes[0].set_ylabel('fraction of 1,024 inputs whose y2 prediction flips'); axes[0].legend(fontsize=6.5)
    fig.suptitle('R2 input dependence of the trained y2 predictions (x3, x5-x10 are irrelevant to y2). Markers o, square, triangle = '
                 'replicates r0, r1, r2; arms offset horizontally.', fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93))
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'flip.{ext}', dpi=130)
    plt.close(fig)


def ridge_probe(H, y, train, test, lam=1e-2):
    """Linear decodability: ridge regression to +-1 on standardized features of the
    pool rows, scored by sign accuracy on the test rows."""
    mu, sd = H[train].mean(0), H[train].std(0) + 1e-8
    A = np.c_[(H-mu)/sd, np.ones(len(H))]
    w = np.linalg.solve(A[train].T @ A[train] + lam*len(train)*np.eye(A.shape[1]), A[train].T @ (2.*y[train]-1))
    return float(np.mean((A[test] @ w > 0) == (y[test] == 1)))


def mean_diff(H, y, rows):
    return H[rows][y[rows] == 1].mean(0) - H[rows][y[rows] == 0].mean(0)


def mechanism(args):
    """Probes and class-mean steering on the initial, phase-1 and n = 512 phase-2 networks."""
    sys.path.insert(0, str(HERE.parent/'common')); sys.path.insert(0, str(HERE))
    import nets
    import task
    out, seeds = [], [52200, 52201, 52202]
    flips = {'x1^x2': task.Z, 'x1^x3': task.PROBES['x1^x3'], 'x5^x6': task.PROBES['x5^x6']}
    x4 = task.B[:, 3] == 1
    for arm in ARMS + FOLLOWUP:
        for s in seeds:
            run_id = f'r1-share-{arm}-s{s}-a1'; d = RUNS/run_id
            if not any((d/f).exists() for f in ('final_n512.npz', 'pretrained.npz')):
                continue
            cfg = json.loads((d/'config.json').read_text())
            pool, test = task.split(cfg['data_seed'])
            net = {'init': nets.init(task.N_BITS, cfg['hidden'], {'pre': 2, 'new': 2}, s)}  # trunk only is used
            if (d/'final_n512.npz').exists():
                with np.load(d/'final_n512.npz') as z:
                    net['final512'] = {k: z[k] for k in z.files}
            if (d/'pretrained.npz').exists():
                net['pretrained'] = nets.load(d/'pretrained.npz')[0]
            acts = {k: nets.forward(p, task.X)[0] for k, p in net.items()}
            base = dict(run_id=run_id, arm=arm, init_seed=s, data_seed=cfg['data_seed'])
            for k, A in acts.items():
                for layer in (1, 2):
                    for name, y in task.PROBES.items():
                        out.append(dict(base, kind='probe', net=k, layer=layer, feature=name,
                                        value=ridge_probe(A[layer], y, pool, test)))
            if 'final512' not in net:  # published without its phase-2 network: probes only
                continue
            # Steering: move each input to the other class mean along a mean-difference direction.
            p_fin = net['final512']
            pred = {'new': nets.forward(p_fin, task.X)[1]['new'].argmax(1)}
            rng = np.random.default_rng(52300 + s - 52200)
            sources = ['final512', 'init'] + (['pretrained'] if 'pretrained' in net else [])
            for layer in (1, 2):
                for src in sources:
                    for name, y in flips.items():
                        dvec = mean_diff(acts[src][layer], y, pool)
                        dirs = {name: dvec}
                        if name == 'x1^x2':
                            r = rng.normal(size=dvec.shape); dirs['random'] = r/np.linalg.norm(r)*np.linalg.norm(dvec)
                        for dn, dv in dirs.items():
                            yy = y if dn != 'random' else task.Z
                            H = acts['final512'][layer] + (1-2*yy)[:, None]*dv
                            new = nets.forward_from(p_fin, layer, H)[1]['new'].argmax(1)
                            out.append(dict(base, kind='steer', net='final512', direction_from=src, layer=layer,
                                            feature=dn, norm=float(np.linalg.norm(dv)),
                                            cos_with_final=float(dv @ mean_diff(acts['final512'][layer], yy, pool) /
                                                                 (np.linalg.norm(dv)*np.linalg.norm(mean_diff(acts['final512'][layer], yy, pool))+1e-12)),
                                            y2_flip_x4is1=float(np.mean(new[x4] != pred['new'][x4])),
                                            y2_flip_x4is0=float(np.mean(new[~x4] != pred['new'][~x4]))))
                    if src == 'pretrained':
                        # The y1/pre head inside the phase-1 network, steered along its own directions.
                        pp = net['pretrained']; pr = nets.forward(pp, task.X)[1]['pre'].argmax(1)
                        for name, y in flips.items():
                            dv = mean_diff(acts['pretrained'][layer], y, pool)
                            H = acts['pretrained'][layer] + (1-2*y)[:, None]*dv
                            new = nets.forward_from(pp, layer, H)[1]['pre'].argmax(1)
                            out.append(dict(base, kind='steer-pre', net='pretrained', direction_from='pretrained',
                                            layer=layer, feature=name, norm=float(np.linalg.norm(dv)),
                                            pre_flip_all=float(np.mean(new != pr))))
    FIGS.mkdir(exist_ok=True)
    write_csv(FIGS/'mechanism.csv', out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    p = sp.add_parser('publish'); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--checkpoints', nargs='*', default=['pretrained.npz', 'final_n64.npz', 'final_n512.npz'])
    sp.add_parser('figures')
    sp.add_parser('mechanism')
    args = ap.parse_args()
    {'publish': publish, 'figures': figures, 'mechanism': mechanism}[args.cmd](args)


if __name__ == '__main__':
    main()
