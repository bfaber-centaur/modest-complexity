"""c2 time lane: publish run records, check continuations, measure, and draw the first figure.

  uv run --locked python research/composition-c2/time/time_analyze.py publish --root runs/c2 --ids ID... [--commit-checkpoints ID/FILE...]
  uv run --locked python research/composition-c2/time/time_analyze.py check     # verify runs and joins -> checks.json
  uv run --locked python research/composition-c2/time/time_analyze.py measure   # CSVs in figures/
  uv run --locked python research/composition-c2/time/time_analyze.py figure    # figures/time_main and time_followup .{png,svg} from those CSVs
  uv run --locked python research/composition-c2/time/time_analyze.py lengths   # F1 100k states: h1/h2 readout, lengths 4-6

Everything reads the committed records: this lane's runs/, r1's published metrics of the
source runs, and the committed checkpoints (../recovered/ for 10k, runs/ for 20k-50k).
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(HERE.parent/'common'), str(REPO/'research'/'regimes'/'common')]
import c2task  # noqa: E402
import nets  # noqa: E402

RUNS, FIGS, RECOVERED = HERE/'runs', HERE/'figures', HERE.parent/'recovered'
ARMS = ('tt-inter', 'tu-inter', 'tt-final')
SEEDS = (51200, 51201, 51202)
SOURCE = {'tt-inter': 'research/regimes/composition/runs/r1-comp-rnn-inter-s{s}-a1',
          'tu-inter': 'research/regimes/review/runs/r1-rev-comp-rnn-untied-s{s}-a1',
          'tt-final': 'research/regimes/composition/runs/r1-comp-rnn-final-s{s}-a1'}
R1_KEYS = ('train', 'held', 'held_final_pred')
SMALL = ('config.json', 'metrics.jsonl', 'runtime.json')
EVERY, THRESHOLD, SUSTAIN, END = 50, 0.95, 5, 50000
CKPT_STEPS = (10000, 20000, 30000, 40000, 50000)


def run_id(arm, seed, attempt=1):
    return f'c2-time-{arm}-s{seed}-a{attempt}'


def source_dir(arm, seed):
    return REPO/SOURCE[arm].format(s=seed)


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines()] if Path(path).exists() else None


def publish(root, ids, committed_checkpoints=()):
    for rid in ids:
        src, dst = root/rid, RUNS/rid
        dst.mkdir(parents=True, exist_ok=True)
        manifest = []
        for f in sorted(src.iterdir()):
            committed = f.name in SMALL or f'{rid}/{f.name}' in committed_checkpoints
            if committed:
                shutil.copy2(f, dst/f.name)
            manifest.append(dict(file=f.name, bytes=f.stat().st_size, sha256=hashlib.sha256(f.read_bytes()).hexdigest(),
                                 committed=committed))
        log = root/f'{rid}.log'
        if log.exists():
            shutil.copy2(log, dst/'stdout.log')
            manifest.append(dict(file='stdout.log', bytes=log.stat().st_size, sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
                                 committed=True))
        (dst/'MANIFEST.json').write_text(json.dumps(manifest, indent=1)+'\n')
        print(rid, len(manifest), 'files')


def same_r1_keys(rows, published):
    """Steps of rows whose r1 keys differ from the published record at the same step (missing steps count)."""
    pub = {r['step']: r for r in published}
    return [r['step'] for r in rows if r['step'] not in pub or any(r[k] != pub[r['step']][k] for k in R1_KEYS)]


def recipe_vs_source(cfg, src):
    """Fields of a c2 continuation's recipe that differ from its r1 source run's config.json (reviewer C1:
    --resume checks only the hash and the tying scheme)."""
    want = dict(task_id=c2task.task_id(src['task_seed'], src['split_seed']), transition='tied',
                readout='untied' if src.get('reviewer', {}).get('mode') == 'untied' else 'tied',
                step_weights=[1., 1., 1.] if src['condition'] == 'inter' else [0., 0., 1.],
                hidden=src['hidden'][0], lr=src['lr'], wd=src['wd'], eval_every=src['eval_every'], init_seed=src['init_seed'])
    return {k: [cfg.get(k), v] for k, v in want.items() if cfg.get(k) != v}


def check():
    out = {}
    for arm in ARMS:
        pub = read_jsonl(source_dir(arm, 51200)/'metrics.jsonl')
        rows = read_jsonl(RUNS/f'c2-time-verify-{arm}-s51200-a1'/'metrics.jsonl')
        if rows is not None:
            vid = f'c2-time-verify-{arm}-s51200-a1'
            out[vid] = dict(source=SOURCE[arm].format(s=51200), steps=[rows[0]['step'], rows[-1]['step']], logged=len(rows),
                            differing_steps=same_r1_keys(rows, pub),
                            recipe_differs=recipe_vs_source(json.loads((RUNS/vid/'config.json').read_text()),
                                                            json.loads((source_dir(arm, 51200)/'config.json').read_text())))
        for s in SEEDS:
            rows = read_jsonl(RUNS/run_id(arm, s)/'metrics.jsonl')
            if rows is None:
                continue
            pub = read_jsonl(source_dir(arm, s)/'metrics.jsonl')
            out[run_id(arm, s)] = dict(source=SOURCE[arm].format(s=s), first_step=rows[0]['step'], last_step=rows[-1]['step'],
                                       logged=len(rows), join_step_differs=bool(same_r1_keys(rows[:1], pub)),
                                       subset_recomputed_equal=subsets(rows[0]['held_final_pred']) == rows[0]['held_subset'],
                                       recipe_differs=recipe_vs_source(json.loads((RUNS/run_id(arm, s)/'config.json').read_text()),
                                                                       json.loads((source_dir(arm, s)/'config.json').read_text())))
    for s in SEEDS:  # F1 extensions: joined to their 50k run, recipe still the r1 source's
        rows, base = read_jsonl(RUNS/ext_id('tt-final', s)/'metrics.jsonl'), read_jsonl(RUNS/run_id('tt-final', s)/'metrics.jsonl')
        if rows is None:
            continue
        out[ext_id('tt-final', s)] = dict(source=run_id('tt-final', s), first_step=rows[0]['step'], last_step=rows[-1]['step'], logged=len(rows),
                                          join_step_differs=bool(same_r1_keys(rows[:1], base[-1:])) or rows[0]['held_subset'] != base[-1]['held_subset'],
                                          recipe_differs=recipe_vs_source(json.loads((RUNS/ext_id('tt-final', s)/'config.json').read_text()),
                                                                          json.loads((source_dir('tt-final', s)/'config.json').read_text())))
    (HERE/'checks.json').write_text(json.dumps(out, indent=1)+'\n')
    print(json.dumps(out, indent=1))


# ---- measures -------------------------------------------------------------------------------

_T = c2task.load()
HELD = c2task.rows(_T['P'], _T['held'])
NBR, _ALL_TRAIN = c2task.neighbour_answerable(_T['P'], _T['train'], _T['held'])
assert _ALL_TRAIN and NBR.sum() == 86


def neighbour_labels(P, train, held):
    """Per held-out row: final labels of all one-op-slot training neighbours, and of the same-prefix ones (op 3 differs)."""
    tr = c2task.rows(P, train)
    lookup = {(int(s), *map(int, q)): int(y) for s, q, y in zip(tr['s'], tr['ops'], tr['states'][:, -1])}
    he = c2task.rows(P, held)
    every, prefix = [], []
    for s, q in zip(he['s'], he['ops']):
        q = list(map(int, q)); a, p = set(), set()
        for j in range(len(q)):
            for v in range(c2task.N_OP):
                if v != q[j]:
                    y = lookup[(int(s), *q[:j], v, *q[j+1:])]
                    a.add(y)
                    if j == len(q) - 1:
                        p.add(y)
        every.append(a); prefix.append(p)
    return every, prefix


EVERY_NBR, PREFIX_NBR = neighbour_labels(_T['P'], _T['train'], _T['held'])
Y = HELD['states'][:, -1]


def preds(s):
    return np.array([int(c) for c in s])


def subsets(pred_string):
    ok = preds(pred_string) == Y
    return dict(N=float(ok[NBR].mean()), notN=float(ok[~NBR].mean()), n_N=int(NBR.sum()), n_notN=int((~NBR).sum()))


def compatibility(pred_string, every=None, prefix=None, y=None):
    """Wrong held-out predictions: count, fraction equal to any one-slot neighbour's label (and the uniform
    expectation over the 7 wrong symbols), and the same for same-prefix neighbours."""
    every, prefix, y = every or EVERY_NBR, prefix or PREFIX_NBR, Y if y is None else y
    p = preds(pred_string); wrong = np.flatnonzero(p != y)
    if not len(wrong):
        return dict(n_wrong=0, any_nbr=None, any_nbr_uniform=None, same_prefix=None, same_prefix_uniform=None)
    f = lambda sets: (float(np.mean([p[i] in sets[i] for i in wrong])), float(np.mean([len(sets[i] - {int(y[i])})/7 for i in wrong])))
    a, au = f(every); b, bu = f(prefix)
    return dict(n_wrong=int(len(wrong)), any_nbr=a, any_nbr_uniform=au, same_prefix=b, same_prefix_uniform=bu)


def first_sustained(steps, acc, threshold=THRESHOLD, k=SUSTAIN, every=EVERY):
    """First logged step t with acc >= threshold at t and the next k-1 logged evaluations, which must be
    consecutive at the logging interval. None means censored at the last logged step."""
    ok = np.asarray(acc) >= threshold - 1e-12
    for i in range(len(steps) - k + 1):
        if ok[i:i+k].all() and steps[i+k-1] - steps[i] == (k-1)*every:
            return int(steps[i])
    return None


def ext_id(arm, seed, attempt=1):
    return f'c2-time-{arm}-ext-s{seed}-a{attempt}'


def trajectory(arm, seed, ext=False):
    """r1's published 0-10,000 record joined to the continuation's 10,050-50,000 record and, with ext, the F1
    extension's 50,050-100,000 record. Returns (rows, joined); joined is False when a part is absent."""
    r1 = read_jsonl(source_dir(arm, seed)/'metrics.jsonl')
    cont = read_jsonl(RUNS/run_id(arm, seed)/'metrics.jsonl')
    if cont is None:
        return r1, False
    assert not same_r1_keys(cont[:1], r1), f'{run_id(arm, seed)}: step-10000 row differs from r1; not joined'
    rows = r1 + cont[1:]
    if not ext:
        return rows, True
    more = read_jsonl(RUNS/ext_id(arm, seed)/'metrics.jsonl')
    if more is None:
        return rows, False
    assert not same_r1_keys(more[:1], cont[-1:]) and more[0]['held_subset'] == cont[-1]['held_subset'], \
        f'{ext_id(arm, seed)}: step-50000 row differs from its 50k run; not joined'
    return rows + more[1:], True


def label(arm, seed, step):
    if step <= 10000:
        return SOURCE[arm].format(s=seed).rsplit('/', 1)[1]
    return run_id(arm, seed) if step <= 50000 else ext_id(arm, seed)


def measure():
    FIGS.mkdir(exist_ok=True)
    measure_set(ARMS, False, 'time', (10000, 20000, 30000, 40000, 50000))
    if any((RUNS/ext_id('tt-final', s)).exists() for s in SEEDS):
        measure_set(('tt-final',), True, 'time_followup', (50000, 75000, 100000))


def measure_set(arms, ext, prefix, ckpt_steps):
    traj_rows, thr_rows, comp_rows = [], [], []
    for arm in arms:
        for s in SEEDS:
            rows, joined = trajectory(arm, s, ext)
            prev = None
            for r in rows:
                sub = r.get('held_subset') or subsets(r['held_final_pred'])
                p = preds(r['held_final_pred'])
                d = r.get('diag', {})
                traj_rows.append(dict(
                    arm=arm, seed=s, run_id=label(arm, s, r['step']),
                    step=r['step'], train_final_acc=r['train']['final']['acc'], train_final_ce=r['train']['final']['ce'],
                    held_final_acc=r['held']['final']['acc'], held_final_ce=r['held']['final']['ce'],
                    train_step1_acc=r['train']['step1']['acc'], train_step2_acc=r['train']['step2']['acc'],
                    held_N_acc=sub['N'], held_notN_acc=sub['notN'],
                    churn='' if prev is None else float(np.mean(p != prev)),
                    grad_norm=d.get('grad_norm', ''), update_norm=d.get('update_norm', ''),
                    sqrt_vhat_median=d.get('adam_sqrt_vhat', {}).get('median', ''),
                    sqrt_vhat_max=d.get('adam_sqrt_vhat', {}).get('max', ''),
                    frac_sqrt_vhat_below_eps=d.get('adam_sqrt_vhat', {}).get('frac_below_eps', ''),
                    train_margin_min=d.get('train_final_margin', {}).get('min', ''),
                    train_margin_median=d.get('train_final_margin', {}).get('median', '')))
                prev = p
                if r['step'] % 1000 == 0:
                    comp_rows.append(dict(arm=arm, seed=s, step=r['step'], **compatibility(r['held_final_pred'])))
            steps = [r['step'] for r in rows]
            last = steps[-1]
            for name, acc in (('held_final', [r['held']['final']['acc'] for r in rows]),
                              ('held_notN', [(r.get('held_subset') or subsets(r['held_final_pred']))['notN'] for r in rows])):
                t = first_sustained(steps, acc)
                after = [a for st, a in zip(steps, acc) if t is not None and st >= t]
                at = {st: r for st, r in zip(steps, rows)}
                start = at[last - 10000]
                thr_rows.append(dict(arm=arm, seed=s, measure=name, joined=joined, last_step=last,
                                     crossing_step=t if t is not None else '', censored=t is None,
                                     min_after_crossing=min(after) if after else '', stays_above=bool(after) and min(after) >= THRESHOLD - 1e-12,
                                     value_at_10k=acc[steps.index(10000)], value_at_end=acc[-1],
                                     change_last_10k=acc[-1] - acc[steps.index(last - 10000)],
                                     # the "gap persists" tolerances (QUESTION.md R1, R2), over the last 10,000 / 1,000 updates
                                     preds_differ_last_10k=int(np.sum(preds(rows[-1]['held_final_pred']) != preds(start['held_final_pred']))),
                                     pred_changes_last_1k=int(sum(np.sum(preds(at[st]['held_final_pred']) != preds(at[st - EVERY]['held_final_pred']))
                                                                  for st in range(last - 1000 + EVERY, last + 1, EVERY))),
                                     held_ce_rel_change_last_10k=rows[-1]['held']['final']['ce']/start['held']['final']['ce'] - 1))
    main = prefix == 'time'
    write_csv(FIGS/f'{prefix}_main.csv' if main else FIGS/f'{prefix}.csv', traj_rows)
    write_csv(FIGS/f'{prefix}_threshold.csv', thr_rows)
    write_csv(FIGS/f'{prefix}_error_compatibility.csv', comp_rows)
    write_csv(FIGS/f'{prefix}_checkpoints.csv', checkpoint_rows(traj_rows, arms, ckpt_steps))


def load_ckpt(arm, seed, step):
    if step == 10000:
        src = SOURCE[arm].format(s=seed).rsplit('/', 1)[1]
        path = RECOVERED/src/'ckpt-10000.npz'
    else:
        path = RUNS/(run_id(arm, seed) if step <= 50000 else ext_id(arm, seed))/f'ckpt-{step}.npz'
    return nets.load(path)[0] if path.exists() else None


def checkpoint_rows(traj_rows, arms=ARMS, ckpt_steps=CKPT_STEPS):
    """Parameter norm per checkpoint, distance to the previous one, and the logged update norms between them
    scaled by the logging interval (an estimate of path length: only every 50th update is logged)."""
    out = []
    for arm in arms:
        for s in SEEDS:
            upd = {r['step']: r['update_norm'] for r in traj_rows if r['arm'] == arm and r['seed'] == s and r['update_norm'] != ''}
            prev, prev_step = None, None
            for step in ckpt_steps:
                p = load_ckpt(arm, s, step)
                if p is None:
                    continue
                flat = np.concatenate([v.ravel() for _, v in sorted(p.items())])
                row = dict(arm=arm, seed=s, step=step, param_norm=float(np.linalg.norm(flat)), dist_prev='', path_est_prev='')
                if prev is not None:
                    row['dist_prev'] = float(np.linalg.norm(flat - prev))
                    row['path_est_prev'] = EVERY*sum(u for st, u in upd.items() if prev_step <= st < step)
                out.append(row); prev, prev_step = flat, step
    return out


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print(path.relative_to(REPO), len(rows), 'rows')


def lengths():
    """F1 states at 100k: the shared readout on h1, h2 of held-out rows, and final accuracy on every
    program of length 4-6 (tied scheme, so step_map [1]*L) -> figures/time_followup_lengths.csv."""
    import c2rnn
    out = []
    for s in SEEDS:
        p = nets.load(RUNS/ext_id('tt-final', s)/'ckpt-100000.npz')[0]
        logits = c2rnn.forward(p, HELD['ops'], s=HELD['s'])[1]
        row = dict(run_id=ext_id('tt-final', s), step=100000,
                   **{f'held_h{t+1}_readout_acc': float(np.mean(logits[t].argmax(1) == HELD['states'][:, t])) for t in (0, 1)})
        for L in (4, 5, 6):
            d = c2task.rows(_T['P'], c2task.programs(L))
            row[f'all_programs_L{L}_acc'] = float(np.mean(c2rnn.forward(p, d['ops'], s=d['s'], step_map=[1]*L)[1][-1].argmax(1) == d['states'][:, -1]))
        out.append(row)
    write_csv(FIGS/'time_followup_lengths.csv', out)


# ---- figure ---------------------------------------------------------------------------------

COLOR = {'tt-inter': '#eb6834', 'tu-inter': '#1baf7a', 'tt-final': '#2a78d6'}  # r1's inter/final colours; dataviz slots 1-3
LABEL = {'tt-inter': 'tied transition, tied readout, intermediate (tt-inter)',
         'tu-inter': 'tied transition, per-step readouts, intermediate (tu-inter)',
         'tt-final': 'tied transition, tied readout, final only (tt-final)'}
MARKER = dict(zip(SEEDS, 'osD'))


def read_csv(path):
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [{k: (float(v) if v not in ('', 'True', 'False') and k not in ('arm', 'run_id', 'measure') else v) for k, v in r.items()} for r in rows]


def figure():
    draw('time_main', 'time_threshold', ARMS, END,
         'c2 time lane: three RNN arms continued from r1 ckpt-10000 to 50,000 updates, unchanged recipe; logged every 50 updates')
    if (FIGS/'time_followup.csv').exists():
        draw('time_followup', 'time_followup_threshold', ('tt-final',), 100000,
             'c2 time lane follow-up F1: tt-final continued from ckpt-50000 to 100,000 updates, unchanged recipe; logged every 50 updates',
             marks=(10000, 50000))


def draw(name, thr_name, arms, end_x, title, marks=(10000,)):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    traj, thr = read_csv(FIGS/f'{name}.csv'), read_csv(FIGS/f'{thr_name}.csv')
    series = lambda arm, s, key: [(r['step'], r[key]) for r in traj if r['arm'] == arm and r['seed'] == s and r[key] != '']
    fig = plt.figure(figsize=(12.5, 14.5))
    gs = fig.add_gridspec(4, 2, height_ratios=[1.25, 1, 1, 1], hspace=.42, wspace=.2, top=.9)
    top = fig.add_subplot(gs[0, :])
    ax = {k: fig.add_subplot(gs[i, j], sharex=top) for k, (i, j) in
          dict(notN=(1, 0), N=(1, 1), ce=(2, 0), margin=(2, 1), norms=(3, 0), vhat=(3, 1)).items()}

    def lines(a, key, arm, s, **kw):
        xy = series(arm, s, key)
        if xy:
            x, y = zip(*xy)
            a.plot(x, y, color=COLOR[arm], marker=MARKER[s], markevery=max(1, len(x)//10), ms=4, **kw)
        return xy

    missing = sorted({f"{r['arm']} s{int(r['seed'])}" for r in thr if r['joined'] != 'True'})
    for arm in arms:
        for s in SEEDS:
            lines(top, 'held_final_acc', arm, s, lw=1.4)
            lines(top, 'train_final_acc', arm, s, lw=.8, ls=':', alpha=.8)
            lines(ax['notN'], 'held_notN_acc', arm, s, lw=1.3)
            lines(ax['N'], 'held_N_acc', arm, s, lw=1.3)
            lines(ax['ce'], 'held_final_ce', arm, s, lw=1.3)
            lines(ax['ce'], 'train_final_ce', arm, s, lw=.8, ls=':')
            lines(ax['margin'], 'train_margin_median', arm, s, lw=1.3)
            lines(ax['margin'], 'train_margin_min', arm, s, lw=.8, ls=':')
            lines(ax['norms'], 'grad_norm', arm, s, lw=1.1)
            lines(ax['norms'], 'update_norm', arm, s, lw=.8, ls=':')
            lines(ax['vhat'], 'sqrt_vhat_median', arm, s, lw=1.3)
            lines(ax['vhat'], 'sqrt_vhat_max', arm, s, lw=.8, ls=':')
    end = max(r['step'] for r in traj)
    for r in thr:
        a = top if r['measure'] == 'held_final' else ax['notN']
        if r['crossing_step'] != '':
            a.scatter([r['crossing_step']], [THRESHOLD], marker='v', s=46, color=COLOR[r['arm']], edgecolor='k', lw=.5, zorder=5)
        else:
            a.scatter([r['last_step']], [r['value_at_end']], marker='>', s=46, facecolor='white', edgecolor=COLOR[r['arm']], lw=1.4, zorder=5, clip_on=False)
    for a in (top, ax['notN'], ax['N']):
        a.axhline(THRESHOLD, color='0.45', lw=1, ls=(0, (4, 3)))
        a.axhline(1/8, color='0.65', lw=.8, ls=':')
        a.set_ylim(-.02, 1.04)
    def first_below(key):
        out = []
        for arm in arms:
            for s in SEEDS:
                t = next((int(x) for x, y in series(arm, s, key) if y < 1e-8), None)
                if t is not None:
                    out.append(t)
        return out
    med, mx = first_below('sqrt_vhat_median'), first_below('sqrt_vhat_max')
    ax['vhat'].axhline(1e-8, color='0.3', lw=1, ls=(0, (4, 3)))
    ax['vhat'].text(.99, .96, 'eps = 1e-8 (dashed)' + (f'\nmedian √v̂ < eps from {min(med):,}–{max(med):,}' if med else '') +
                    (f'\nevery coordinate < eps from {min(mx):,}–{max(mx):,}\n({len(mx)} of {len(arms)*len(SEEDS)} runs; first logged update)' if mx else ''),
                    transform=ax['vhat'].transAxes, ha='right', va='top', fontsize=8, color='0.25')
    for k in ('ce', 'norms', 'vhat'):
        ax[k].set_yscale('log')
    for a in (top, *ax.values()):
        for m in marks:
            a.axvline(m, color='0.55', lw=.9)
        a.set_xlim(0, max(end_x, end)); a.grid(alpha=.25, lw=.5)
    names = ['r1 record', 'continuation to 50k', 'F1 extension to 100k']
    for k, m in enumerate(marks):
        if k == 0:
            top.text(m - .006*end_x, .03, names[0], ha='right', fontsize=8, color='0.35', transform=top.get_xaxis_transform())
        top.text(m + .006*end_x, .03, names[k+1], ha='left', fontsize=8, color='0.35', transform=top.get_xaxis_transform())
    if missing:
        top.text(.995, .1, 'not run or not joined: ' + ', '.join(missing), transform=top.transAxes, ha='right', fontsize=7, color='crimson', wrap=True)
    top.set_title('Held-out final accuracy (solid) and train final accuracy (dotted), 128 held-out / 384 train rows', loc='left', fontsize=10)
    top.set_ylabel('accuracy')
    ax['notN'].set_title('Held-out ¬N: 42 rows no one-op-slot neighbour copy answers', loc='left', fontsize=10)
    ax['N'].set_title('Held-out N: 86 rows a neighbour copy can answer', loc='left', fontsize=10)
    ax['ce'].set_title('Final-step cross-entropy: held-out (solid), train (dotted)', loc='left', fontsize=10)
    ax['margin'].set_title('Train final margin: median (solid), min (dotted); from 10k', loc='left', fontsize=10)
    ax['norms'].set_title('Global L2: gradient (solid), update incl. decay (dotted); from 10k', loc='left', fontsize=10)
    ax['vhat'].set_title('Adam √v̂: median (solid), max (dotted); from 10k', loc='left', fontsize=10)
    for k in ('norms', 'vhat'):
        ax[k].set_xlabel('update')
    handles = [Line2D([], [], color=COLOR[a], lw=2, label=LABEL[a]) for a in arms]
    handles += [Line2D([], [], color='0.3', marker=MARKER[s], lw=0, label=f'init seed {s}') for s in SEEDS]
    handles += [Line2D([], [], color='0.3', marker='v', lw=0, label='first of 5 logged evals ≥ 0.95'),
                Line2D([], [], color='0.3', marker='>', markerfacecolor='white', lw=0, label='censored (never sustained ≥ 0.95)'),
                Line2D([], [], color='0.45', ls=(0, (4, 3)), label='0.95 threshold'), Line2D([], [], color='0.65', ls=':', label='chance 1/8')]
    fig.legend(handles=handles, loc='upper center', ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(.5, .975))
    fig.suptitle(title, y=.99, fontsize=11)
    for ext in ('png', 'svg'):
        fig.savefig(FIGS/f'{name}.{ext}', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('wrote', (FIGS/f'{name}.png').relative_to(REPO))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('publish'); p.add_argument('--root', type=Path, required=True); p.add_argument('--ids', nargs='+', required=True)
    p.add_argument('--commit-checkpoints', nargs='*', default=[])
    sub.add_parser('check'); sub.add_parser('measure'); sub.add_parser('figure'); sub.add_parser('lengths')
    a = ap.parse_args()
    if a.cmd == 'publish':
        publish(a.root, a.ids, set(a.commit_checkpoints))
    elif a.cmd == 'check':
        check()
    elif a.cmd == 'measure':
        measure()
    elif a.cmd == 'lengths':
        lengths()
    else:
        figure()


if __name__ == '__main__':
    main()
